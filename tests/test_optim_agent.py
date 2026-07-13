"""Offline self-check: `python tests/test_optim_agent.py` (also pytest-compatible)."""

import json
import copy
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from optim_agent import agent, samplers, space


def test_packaging_declares_development_and_vision_extras():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        from setuptools._vendor import tomli as tomllib

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )
    extras = pyproject["project"]["optional-dependencies"]
    assert any(requirement.startswith("pytest") for requirement in extras["dev"])
    assert any(requirement.startswith("torch") for requirement in extras["vision"])
    assert any(requirement.startswith("torchvision") for requirement in extras["vision"])


def test_public_repository_metadata_and_ignore_contract():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        from setuptools._vendor import tomli as tomllib

    root = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert set(pyproject["project"]["urls"]) >= {
        "Homepage", "Documentation", "Repository", "Issues", "Changelog",
    }

    ignored = (root / ".gitignore").read_text().splitlines()
    for pattern in (
        "graphify-out/", "autoresearch-results/", ".codex-autoresearch/",
        ".coverage", ".mypy_cache/", ".ruff_cache/", ".ipynb_checkpoints/",
    ):
        assert pattern in ignored
    assert "paper/src/" not in ignored


def test_public_governance_files_are_substantive():
    root = Path(__file__).resolve().parent.parent
    required_sections = {
        "CONTRIBUTING.md": ("Development setup", "Pull requests", "Benchmarks"),
        "SECURITY.md": ("Supported versions", "Reporting", "Response"),
        "CODE_OF_CONDUCT.md": ("Standards", "Enforcement"),
        "CHANGELOG.md": ("Unreleased", "0.1.0"),
        "ROADMAP.md": ("Public launch", "Research", "Non-goals"),
    }
    for filename, sections in required_sections.items():
        text = (root / filename).read_text()
        assert all(section in text for section in sections), filename
    changelog = (root / "CHANGELOG.md").read_text()
    assert "/releases/tag/v0.1.0" not in changelog


def test_cpu_first_examples_have_runnable_search_spaces():
    import subprocess

    from examples import quickstart, sklearn_tuning

    study = quickstart.run(trials=4, seed=3)
    assert len(study.trials) == 4
    assert study.best_value is not None

    trial = oa.create_study().ask({
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 4,
        "max_features": "sqrt",
    })
    params = sklearn_tuning.suggest_params(trial)
    assert params == trial.params
    assert set(params) == {
        "n_estimators", "max_depth", "min_samples_split", "max_features",
    }

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "examples/quickstart.py"), "--trials", "2"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "best params:" in result.stdout


def test_inference_example_exposes_quality_latency_and_cost():
    from examples import inference_tuning

    trial = oa.create_study().ask({
        "quantization": "int8",
        "batch_size": 8,
        "max_tokens": 256,
        "speculative_decoding": True,
    })
    params = inference_tuning.suggest_params(trial)
    metrics = inference_tuning.evaluate_configuration(params)
    utility = inference_tuning.utility(metrics)

    assert set(metrics) == {
        "quality", "p95_latency_ms", "cost_per_1k_requests_usd",
    }
    assert 0 <= metrics["quality"] <= 1
    assert metrics["p95_latency_ms"] > 0
    assert metrics["cost_per_1k_requests_usd"] > 0
    assert isinstance(utility, float)


def test_documentation_portal_is_deployable_and_matches_sampler_api():
    root = Path(__file__).resolve().parent.parent
    docs = (root / "docs/index.html").read_text()
    workflow = (root / ".github/workflows/docs.yml").read_text()
    notebook = json.loads((root / "tutorials/quickstart.ipynb").read_text())

    assert "actions/deploy-pages" in workflow
    assert "Tutorials" in docs
    assert 'href="../examples/' not in docs
    assert 'href="../tutorials/' not in docs
    assert "github.com/Optim-Agent/optim-agent/blob/main/examples/" in docs
    assert "colab.research.google.com/github/Optim-Agent/optim-agent" in docs
    for example in (
        "quickstart.py", "sklearn_tuning.py", "inference_tuning.py",
    ):
        assert example in docs
    assert "xhigh" not in docs and "max</code>" not in docs
    assert "history=5" in docs
    assert "explicit_reasoning=True" in docs
    assert "qualitative_notes=True" in docs
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "colab.research.google.com" in notebook_text
    assert 'backend="mock"' in notebook_text


def test_readme_positions_optim_agent_as_a_general_system_optimizer():
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()
    docs = (root / "docs/index.html").read_text()
    pyproject = (root / "pyproject.toml").read_text()

    required = (
        "Agentic system optimization with coding agents",
        "algorithm engineer",
        "configurable parameters",
        "measurable objective",
        "Model training",
        "Inference and serving",
        "Quantitative research",
        "Reinforcement learning and decisions",
        "Scientific workflows",
        "Black-box systems",
    )
    assert all(term in readme for term in required)
    assert "for tuning any machine-learning or deep-learning training run" not in readme
    assert "Agentic system optimization with coding agents" in docs
    assert "Agentic system optimization" in pyproject

    headings = (
        "## Why optim-agent",
        "## Install",
        "## Quickstart",
        "## Where it applies",
        "## Optimization trajectory",
        "### Optimizing Math Functions without Context: Branin-2D and Ackley-5D",
        "### Tuning ResNet-based Image Classifier: MNIST and CIFAR-10",
        "### Tuning Gradient Boosting Classifier: Credit-default Probabilities",
        "## Usage guide",
    )
    positions = [readme.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_benchmark_manifest_records_reproduction_contract():
    root = Path(__file__).resolve().parent.parent
    guide = (root / "benchmarks/README.md").read_text()
    manifest = json.loads((root / "benchmarks/manifest.json").read_text())

    assert all(term in guide for term in (
        "Provenance", "Publication gate", "no supplied task context",
    ))
    assert {suite["id"] for suite in manifest["suites"]} == {
        "mnist", "cifar10", "hard-functions", "credit-default-benchmark",
    }
    for suite in manifest["suites"]:
        assert suite["result_glob"].startswith("docs/assets/")
        assert suite["seeds"]
        assert suite["trials"] > 0

def test_credit_card_protocol_is_pinned_and_cpu_only():
    from examples import credit_card as credit

    assert credit.DATASET_ID == 350
    assert credit.DATA_URL == (
        "https://archive.ics.uci.edu/static/public/350/"
        "default+of+credit+card+clients.zip"
    )
    assert credit.DATA_SHA256 == (
        "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
    )
    assert credit.ARCHIVE_MEMBER == "default of credit card clients.xls"
    assert credit.MODEL == "gpt-5.5"
    assert credit.SEEDS == (0, 1, 2, 3, 4)
    assert credit.N_TRIALS == 20
    assert credit.N_INIT == 3
    assert credit.SPLIT_SEED == 20260713
    assert credit.DEFAULT_PREVALENCE == pytest.approx(6636 / 30000)
    assert credit.METHODS == (
        "Random",
        "TPE",
        "GPT-5.5-low",
        "GPT-5.5-medium",
        "GPT-5.5-high",
        "GPT-5.5-medium-no-context",
    )
    assert credit.SELECTED_METHOD == "GPT-5.5-low"


def test_credit_default_search_space_removes_all_context_for_control():
    from examples import credit_card as credit

    class Trial:
        def __init__(self):
            self.contexts = []

        def suggest_float(self, name, low, high, *, log=False, context=None):
            self.contexts.append((name, context))
            return low

        def suggest_int(self, name, low, high, *, log=False, context=None):
            self.contexts.append((name, context))
            return low

        def suggest_categorical(self, name, choices, *, context=None):
            self.contexts.append((name, context))
            return choices[0]

    contextual = Trial()
    params = credit.suggest_params(contextual, use_context=True)
    assert set(params) == {
        "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
        "min_samples_leaf", "l2_regularization", "max_bins",
        "positive_class_weight",
    }
    assert len(contextual.contexts) == 8
    assert all(context for _, context in contextual.contexts)

    control = Trial()
    assert credit.suggest_params(control, use_context=False).keys() == params.keys()
    assert all(context is None for _, context in control.contexts)


def test_credit_default_frame_preparation_has_expected_schema_and_categories():
    pandas = pytest.importorskip("pandas")
    from examples import credit_card as credit

    row = {column: 1 for column in credit.RAW_COLUMNS}
    row["ID"] = 99
    row[credit.TARGET_COLUMN] = 0
    for column in credit.PAY_STATUS_COLUMNS:
        row[column] = -2
    frame = pandas.DataFrame([row, {**row, "ID": 100, credit.TARGET_COLUMN: 1}])

    features, target, categorical = credit._prepare_frame(frame)

    assert features.shape == (2, 23)
    assert "ID" not in features and credit.TARGET_COLUMN not in features
    assert target.tolist() == [0, 1]
    assert all(features[column].tolist() == [0, 0]
               for column in credit.PAY_STATUS_COLUMNS)
    assert categorical == [column in credit.CATEGORICAL_COLUMNS for column in features]

    with pytest.raises(ValueError, match="credit-default schema"):
        credit._prepare_frame(frame.drop(columns=["LIMIT_BAL"]))


def test_credit_default_split_model_and_method_contract():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from examples import credit_card as credit

    features = pandas.DataFrame({
        "numeric": list(range(100)),
        "category": [index % 4 for index in range(100)],
    })
    target = pandas.Series([index % 2 for index in range(100)])
    first = credit._split_data(features, target)
    second = credit._split_data(features, target)
    assert [len(first[name]) for name in ("x_train", "x_valid", "x_test")] == [60, 20, 20]
    assert first["x_train"].index.tolist() == second["x_train"].index.tolist()
    assert first["y_valid"].value_counts().to_dict() == {0: 10, 1: 10}

    params = {
        "learning_rate": 0.05,
        "max_iter": 120,
        "max_leaf_nodes": 23,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "l2_regularization": 0.2,
        "max_bins": 64,
        "positive_class_weight": 2.5,
    }
    model = credit._build_model(params, [False, True])
    model_params = model.get_params()
    assert model_params["random_state"] == credit.SPLIT_SEED
    assert model_params["categorical_features"] == [False, True]
    assert model_params["class_weight"] == {0: 1.0, 1: 2.5}
    assert "positive_class_weight" not in model_params

    assert credit._method_spec("GPT-5.5-low") == {
        "backend": "codex", "model": "gpt-5.5", "effort": "low",
        "use_context": True, "n_init": 3,
    }
    assert credit._method_spec("GPT-5.5-medium-no-context")["use_context"] is False
    assert credit._method_spec("Random")["backend"] is None
    assert credit._method_spec("TPE")["backend"] == "tpe"
    assert "30,000" in credit.TASK_CONTEXT
    assert "log loss" in credit.TASK_CONTEXT
    assert "20-trial" in credit.TASK_CONTEXT
    selected = credit.selected_summary()
    assert selected["selected_method"] == "GPT-5.5-low"
    assert (
        selected["selected"]["final_validation"]
        < selected["random"]["final_validation"]
        < 1
    )
    assert selected["selected"]["final_validation"] < selected["tpe"]["final_validation"]


def test_credit_default_optuna_adapter_ignores_parameter_context():
    from examples import credit_card as credit

    class Trial:
        def suggest_float(self, name, low, high, *, log=False):
            return low

        def suggest_int(self, name, low, high, *, log=False):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    params = credit.suggest_params(credit.OptunaTrialAdapter(Trial()), use_context=True)
    assert params["learning_rate"] == 0.01
    assert params["max_iter"] == 50
    assert params["max_depth"] is None


def test_credit_default_evaluation_and_agent_sampler_are_publication_safe(tmp_path):
    numpy = pytest.importorskip("numpy")
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from examples import credit_card as credit

    rng = numpy.random.default_rng(7)
    features = pandas.DataFrame({
        "amount": rng.normal(size=240),
        "status": rng.integers(0, 4, size=240),
    })
    target = pandas.Series(
        ((features["amount"] + 0.35 * features["status"] + rng.normal(size=240)) > 1.0)
        .astype(int)
    )
    split = credit._split_data(features, target)
    params = {
        "learning_rate": 0.08,
        "max_iter": 60,
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "l2_regularization": 0.1,
        "max_bins": 32,
        "positive_class_weight": 1.5,
    }
    metrics = credit._evaluate_params(params, split, [False, True], include_test=True)
    assert 0 < metrics["validation_log_loss"] < 2
    assert 0 < metrics["test_log_loss"] < 2

    contextual = credit._make_sampler(
        "GPT-5.5-low", seed=2, timeout=30, agent_cwd=tmp_path,
    )
    assert contextual.effort == "low"
    assert contextual.context == credit.TASK_CONTEXT
    assert contextual.fail_closed is True
    assert contextual.n_init == 3

    control = credit._make_sampler(
        "GPT-5.5-medium-no-context", seed=2, timeout=30, agent_cwd=tmp_path,
    )
    assert control.effort == "medium"
    assert control.context is None
    assert control.fail_closed is True
    assert credit._artifact_path("GPT-5.5-medium-no-context", 4).name == (
        "credit_default_GPT-5.5-medium-no-context_s4.json"
    )


def test_credit_default_archive_and_artifact_validation_reject_corruption(tmp_path):
    import hashlib
    import zipfile

    from examples import credit_card as credit

    archive = tmp_path / "credit.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(credit.ARCHIVE_MEMBER, b"fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert credit._validate_archive(archive, expected_hash=digest) == archive
    with pytest.raises(ValueError, match="archive member"):
        credit._validate_archive(archive, expected_hash=digest, member="wrong.xls")
    with pytest.raises(ValueError, match="SHA-256"):
        credit._validate_archive(archive, expected_hash="0" * 64)

    params = [{
        "learning_rate": 0.05,
        "max_iter": 120,
        "max_leaf_nodes": 23,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "l2_regularization": 0.2,
        "max_bins": 64,
        "positive_class_weight": 2.5,
    } for _ in range(credit.N_TRIALS)]
    run = credit._common_metadata("GPT-5.5-low", 0)
    run.update({
        "values": [0.65 - 0.005 * trial for trial in range(credit.N_TRIALS)],
        "params": params,
        "best_validation_log_loss": 0.555,
        "best_params": params[-1],
        "test_log_loss": 0.57,
        "default_validation_log_loss": 0.66,
        "default_test_log_loss": 0.67,
        "elapsed_seconds": 10.0,
    })
    credit._validate_artifact(run, "GPT-5.5-low", 0)

    corrupted = copy.deepcopy(run)
    corrupted["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="credit-default artifact"):
        credit._validate_artifact(corrupted, "GPT-5.5-low", 0)

    corrupted = copy.deepcopy(run)
    corrupted["params"][0]["max_iter"] = 999
    with pytest.raises(ValueError, match="credit-default artifact"):
        credit._validate_artifact(corrupted, "GPT-5.5-low", 0)

    corrupted = copy.deepcopy(run)
    corrupted["values"][0] = float("nan")
    with pytest.raises(ValueError, match="credit-default artifact"):
        credit._validate_artifact(corrupted, "GPT-5.5-low", 0)

    corrupted = copy.deepcopy(run)
    corrupted["best_params"] = {**params[-1], "max_iter": 121}
    with pytest.raises(ValueError, match="credit-default artifact"):
        credit._validate_artifact(corrupted, "GPT-5.5-low", 0)

    corrupted = copy.deepcopy(run)
    corrupted["default_prevalence"] = 0.5
    with pytest.raises(ValueError, match="credit-default artifact"):
        credit._validate_artifact(corrupted, "GPT-5.5-low", 0)


def test_credit_default_random_and_tpe_share_the_recorded_search_space(monkeypatch):
    pytest.importorskip("optuna")
    from examples import credit_card as credit

    def fake_evaluate(params, split, categorical, include_test=False):
        return {"validation_log_loss": float(params["learning_rate"] + 0.5)}

    monkeypatch.setattr(credit, "_evaluate_params", fake_evaluate)
    for method in ("Random", "TPE"):
        values, params = credit._run_search(
            method, seed=0, split={}, categorical_features=[], timeout=1,
        )
        assert len(values) == len(params) == credit.N_TRIALS
        assert all(set(point) == set(credit.PARAMETER_CONTEXT) for point in params)
        assert values == [point["learning_rate"] + 0.5 for point in params]


def test_credit_default_publication_contract_is_complete_in_every_language():
    from examples import credit_card as credit

    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()
    docs = (root / "docs/index.html").read_text()
    manifest = json.loads((root / "benchmarks/manifest.json").read_text())
    translations = sorted((root / "docs/i18n").glob("README_*.md"))
    assets = root / "docs/assets"

    assert (assets / "credit_card.png").exists()
    assert readme.count("credit_card.png") == 1
    assert docs.count("credit_card.png") == 1
    assert all(path.read_text().count("credit_card.png") == 1
               for path in translations)
    assert len(list(assets.glob("credit_default_*_s*.json"))) == 30
    suite = next(suite for suite in manifest["suites"]
                 if suite["id"] == "credit-default-benchmark")
    assert suite["result_glob"] == "docs/assets/credit_default_*_s*.json"
    assert suite["trials"] == 20
    assert suite["seeds"] == [0, 1, 2, 3, 4]
    assert suite["dataset_id"] == 350
    assert suite["dataset_sha256"] == credit.DATA_SHA256
    assert suite["context_policy"] == "selected contextual agent plus matched no-context"
    assert suite["selected_method"] == credit.SELECTED_METHOD
    assert suite["display_method"] == "GPT-5.5"
    for text in (readme, docs):
        rendered = " ".join(text.split())
        assert "Default of Credit Card Clients" in rendered
        assert "CC BY 4.0" in rendered
        assert "not a production credit-decision system" in rendered
        assert "selected benchmark config" in rendered


def test_classification_benchmarks_use_one_figure_and_table():
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()
    docs = (root / "docs/index.html").read_text()
    assets = root / "docs/assets"

    assert (assets / "classification_benchmarks.png").exists()
    assert not (assets / "mnist_benchmarks.png").exists()
    assert not (assets / "cifar10_benchmarks.png").exists()
    assert readme.count("classification_benchmarks.png") == 1
    assert docs.count("classification_benchmarks.png") == 1
    assert "MNIST cumulative error" in readme
    assert "CIFAR-10 cumulative error" in readme
    for label in ("GPT-5.5 w/ context", "GPT-5.5 w/o context"):
        assert label in readme
        assert label in docs
    assert "GPT-5.5 medium, no context" not in readme
    assert "GPT-5.5 medium, no context" not in docs


def test_trajectory_renderer_uses_committed_runs_and_emits_animation():
    from PIL import Image
    from scripts import render_trajectory

    assert render_trajectory.TITLE == (
        "Optim-agent GPT-5.5 vs Optuna TPE on Branin 2D"
    )
    assert render_trajectory.AGENT_LABEL == "GPT-5.5"
    assert render_trajectory.BASELINE_LABEL == "TPE"

    root = Path(__file__).resolve().parent.parent
    baseline, agent_run = render_trajectory.load_runs(root / "docs/assets", seed=0)
    assert baseline["label"] == "TPE"
    assert agent_run["label"].startswith("GPT-5.5")
    assert len(baseline["functions"]["branin"]["params"]) == 10
    assert len(agent_run["functions"]["branin"]["params"]) == 10

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "trajectory.gif"
        render_trajectory.render(output, baseline, agent_run, dpi=60)
        with Image.open(output) as animation:
            assert animation.n_frames >= 10
            assert animation.width >= 700
            assert animation.height >= 400


def test_classification_renderer_emits_two_panel_figure():
    from PIL import Image
    from scripts import render_classification_benchmarks as renderer

    assert tuple(renderer.DATASETS) == ("MNIST", "CIFAR-10")
    assert renderer.DISPLAY_LABELS == {
        renderer.mnist.PLOT_AGENT_LABEL: "GPT-5.5 w/ context",
        renderer.mnist.PLOT_NO_CONTEXT_LABEL: "GPT-5.5 w/o context",
    }

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "classification.png"
        renderer.render(output)
        with Image.open(output) as figure:
            assert figure.width >= 1200
            assert figure.height >= 500


def test_chinese_readme_covers_primary_user_paths():
    root = Path(__file__).resolve().parent.parent
    english = (root / "README.md").read_text()
    chinese = (root / "docs/i18n/README_ZH.md").read_text()

    assert "docs/i18n/README_ZH.md" in english
    for term in ("安装", "快速开始", "机器学习", "推理", "基准", "贡献"):
        assert term in chinese
    for target in (
        "../../README.md", "../../examples/quickstart.py",
        "../../examples/sklearn_tuning.py",
        "../../examples/inference_tuning.py", "../../tutorials/quickstart.ipynb",
    ):
        assert target in chinese


def test_readme_links_all_localized_editions():
    root = Path(__file__).resolve().parent.parent
    english = (root / "README.md").read_text()
    assert "OpenCode currently does not support distributed computing" in english
    assert "[OpenCode](https://github.com/sst/opencode)" in english
    assert 'python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"' in english
    editions = {
        "ZH": "简体中文",
        "JA": "日本語",
        "KO": "한국어",
        "FR": "Français",
        "DE": "Deutsch",
        "ES": "Español",
        "PT": "Português",
        "RU": "Русский",
    }

    for code, name in editions.items():
        relative = f"docs/i18n/README_{code}.md"
        localized = (root / relative).read_text()
        assert relative in english
        assert name in localized
        assert "../../README.md" in localized
        for text in (
            "GPT-5.5 w/ context",
            "GPT-5.5 w/o context",
            "model_reasoning_effort=medium",
            "$skill-installer install https://github.com/Optim-Agent/optim-agent",
            "../../SKILL.md",
            "python -m pip install optim-agent",
            "git+https://github.com/Optim-Agent/optim-agent.git",
            "distributed",
            "[OpenCode](https://github.com/sst/opencode)",
        ):
            assert text in localized


def quadratic(trial):
    x = trial.suggest_float("x", -5, 5, context="knob that centers a parabola at x=2")
    return (x - 2) ** 2


def test_random_study():
    study = oa.create_study(seed=0)
    study.optimize(quadratic, n_trials=30, verbose=False)
    assert len(study.trials) == 30
    assert all(-5 <= t.params["x"] <= 5 for t in study.trials)
    assert study.best_value < 1.0


def test_extract_json():
    assert agent.extract_json('```json\n{"x": 1.5}\n```') == {"x": 1.5}
    assert agent.extract_json('Sure! Here you go: {"x": 2, "y": "a"} done') == {"x": 2, "y": "a"}
    assert agent.extract_json('schema {"x": "<float>"} answer {"x": 3.0}') == {"x": 3.0}
    assert agent.extract_json("no json here") is None


def test_call_agent_forwards_working_directory():
    import os
    from types import SimpleNamespace

    seen = {}
    original = agent.subprocess.run

    def fake_run(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    agent.subprocess.run = fake_run
    try:
        reply = agent.call_agent(
            "codex", "gpt-5.5", "prompt", effort="medium",
            cwd="/tmp/optim-agent-empty",
        )
    finally:
        agent.subprocess.run = original

    expected_cwd = os.path.realpath("/tmp/optim-agent-empty")
    assert reply == '{"ok": true}'
    assert seen["kwargs"]["cwd"] == expected_cwd
    assert seen["kwargs"]["env"]["PWD"] == expected_cwd
    assert seen["kwargs"]["env"]["OLDPWD"] == expected_cwd


def test_opencode_command_forwards_working_directory():
    command = agent._cmd(
        "opencode", "opencode/big-pickle", "prompt", "medium",
        cwd="/tmp/optim-agent-empty",
    )

    assert command[-3:] == ["--dir", "/tmp/optim-agent-empty", "prompt"]


def test_agent_sampler(monkeypatch=None):
    calls = []

    def fake_call(backend, model, prompt, timeout, effort=None):
        calls.append((prompt, effort))
        return 'reasoning... ```json\n{"x": 2.01, "_note": "min near x=2"}\n```'

    original = samplers._agent.call_agent
    samplers._agent.call_agent = fake_call
    try:
        s = oa.AgentSampler(backend="claude", effort="high", n_init=2, seed=1)
        study = oa.create_study(sampler=s, seed=1)
        study.optimize(quadratic, n_trials=5, verbose=False)
    finally:
        samplers._agent.call_agent = original
    assert calls, "agent was never consulted"
    assert "Search space" in calls[0][0] and "History summary" in calls[0][0]
    assert "centers a parabola" in calls[0][0], "per-param context not shown to agent"
    assert "min near x=2" in calls[-1][0], "note not carried forward"
    assert calls[0][1] == "high", "effort not forwarded to the CLI call"
    assert abs(study.best_trial.params["x"] - 2.01) < 1e-9
    # effort maps to each CLI's reasoning-effort flag
    assert agent._cmd("claude", None, "p", "high")[-3:-1] == ["--effort", "high"]
    assert "model_reasoning_effort=high" in agent._cmd("codex", None, "p", "high")
    assert "--variant" in agent._cmd("opencode", None, "p", "low")
    assert "--effort" not in agent._cmd("claude", None, "p")  # None effort adds no flag
    assert not hasattr(samplers, "EFFORTS")
    assert _raises(ValueError, lambda: oa.AgentSampler(history=-1))
    quiet = oa.AgentSampler(
        backend="claude", history=5, explicit_reasoning=False, qualitative_notes=False,
    )
    low_prompt = quiet._prompt(study, [t for t in study.trials if t.value is not None])
    assert "History summary:" in low_prompt
    assert "Trial history (oldest first):" not in low_prompt
    assert 'Include a short "_reasoning" field' not in low_prompt
    assert 'Include a "_note" field' not in low_prompt
    medium_prompt = s._prompt(study, [t for t in study.trials if t.value is not None])
    assert s.history == 5
    assert s.explicit_reasoning is True
    assert s.qualitative_notes is True
    assert 'Include a short "_reasoning" field' in medium_prompt
    assert 'Include a "_note" field' in medium_prompt
    assert "History summary:" in medium_prompt
    assert "Promising trials:" in medium_prompt
    assert "Failed or weak regions to avoid:" not in medium_prompt
    assert "Use the task context as priors when available" in medium_prompt
    s.context = "Full MNIST ResNet neural architecture search"
    context_prompt = s._prompt(study, [t for t in study.trials if t.value is not None])
    assert "Context-derived priors:" in context_prompt
    # out-of-range and garbage replies fall back gracefully
    samplers._agent.call_agent = lambda *a, **k: '{"x": 999}'
    try:
        study.optimize(quadratic, n_trials=1, verbose=False)  # clamped to 5
        assert study.trials[-1].params["x"] == 5
    finally:
        samplers._agent.call_agent = original


def test_agent_sampler_retries_transient_call_failure():
    calls = []
    original = samplers._agent.call_agent

    def flaky_call(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("temporary provider stream failure")
        return '{"x": 2.0, "_reasoning": "retry", "_note": "provider recovered"}'

    samplers._agent.call_agent = flaky_call
    try:
        sampler = oa.AgentSampler(
            backend="claude", effort="medium", n_init=0, seed=0,
            initial_space={"x": space.Float(-5, 5)},
        )
        study = oa.create_study(sampler=sampler, seed=0)
        trial = study.ask()
        assert trial.suggest_float("x", -5, 5) == 2.0
    finally:
        samplers._agent.call_agent = original

    assert len(calls) == 2
    assert sampler.note == "provider recovered"


def test_agent_sampler_fail_closed_rejects_fallback(monkeypatch):
    def failed_call(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(samplers._agent, "call_agent", failed_call)
    sampler = oa.AgentSampler(
        backend="claude", effort="medium", n_init=0, seed=0,
        initial_space={"x": space.Float(-5, 5)}, fail_closed=True,
    )
    study = oa.create_study(sampler=sampler, seed=0)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        study.ask()


def test_cumulative_error_metric_agent_owns_post_startup():
    calls = []
    original = samplers._agent.call_agent
    samplers._agent.call_agent = lambda *args, **kwargs: calls.append(args) or '{"x": 1.5}'
    s = oa.AgentSampler(backend="claude", effort="medium", n_init=1,
                        context="cumulative best-so-far error", seed=0)
    s.rng.random = lambda: 0.0
    study = oa.create_study(sampler=s, seed=0)
    first = study.ask({"x": 2.0})
    first.suggest_float("x", -5, 5)
    study.tell(first, 0.0)

    try:
        proposal = study.sampler.propose(study)
    finally:
        samplers._agent.call_agent = original

    assert proposal == {"x": 1.5}
    assert calls


def test_cumulative_error_metric_hands_off_after_schema_trial():
    calls = []
    original = samplers._agent.call_agent
    samplers._agent.call_agent = lambda *args, **kwargs: calls.append(args) or '{"x": 2.0}'
    try:
        study = oa.create_study(
            sampler=oa.AgentSampler(
                backend="claude", effort="medium", n_init=4,
                context="cumulative best-so-far error", seed=0,
            ),
            seed=0,
        )
        first = study.ask()
        first.suggest_float("x", -5, 5)
        second = study.ask()
        assert second.suggest_float("x", -5, 5) == 2.0
    finally:
        samplers._agent.call_agent = original
    assert calls, "cumulative-error runs should hand off after discovering the search space"


def test_cumulative_error_metric_uses_declared_space_on_first_trial():
    calls = []
    original = samplers._agent.call_agent

    def fake_call(*args, **kwargs):
        calls.append(args)
        return json.dumps({
            "candidates": [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}],
        })

    samplers._agent.call_agent = fake_call
    try:
        sampler = oa.AgentSampler(
            backend="claude", effort="medium", n_init=4,
            context="cumulative best-so-far error", seed=0,
            initial_space={"x": space.Float(-5, 5, context="test parameter")},
        )
        study = oa.create_study(sampler=sampler, seed=0, max_concurrency=4)
        first = study.ask()
        value = first.suggest_float("x", -5, 5, context="test parameter")
    finally:
        samplers._agent.call_agent = original

    assert value == 1.0
    assert len(calls) == 1
    assert "joint portfolio of 4" in calls[0][2]
    assert "test parameter" in calls[0][2]


def test_cumulative_error_metric_joint_startup_portfolio():
    calls = []
    original = samplers._agent.call_agent

    def fake_call(*args, **kwargs):
        calls.append(args)
        return json.dumps({
            "candidates": [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}],
            "_note": "four distinct startup hypotheses",
        })

    samplers._agent.call_agent = fake_call
    try:
        sampler = oa.AgentSampler(
            backend="claude", effort="medium", n_init=4,
            context="cumulative best-so-far error", seed=0,
        )
        sampler.rng.random = lambda: 1.0
        study = oa.create_study(sampler=sampler, seed=0, max_concurrency=4)
        schema = study.ask()
        schema.suggest_float("x", -5, 5)
        values = [study.ask().suggest_float("x", -5, 5) for _ in range(4)]
    finally:
        samplers._agent.call_agent = original

    assert values == [1.0, 2.0, 3.0, 4.0]
    assert len(calls) == 1
    assert "joint portfolio of 4" in calls[0][2]
    assert "candidate order is evaluation order" in calls[0][2]
    assert "not a risky stress test" in calls[0][2]
    assert sampler.note == "four distinct startup hypotheses"


def test_anchor_proposals_seed_warmup():
    s = oa.AgentSampler(
        backend="claude", effort="medium", n_init=4, context="cumulative best-so-far error", seed=0,
        anchor_proposals=[{"x": 1.5}, {"x": 2.5}],
    )
    study = oa.create_study(sampler=s, seed=0)
    first = study.ask()
    assert first.suggest_float("x", -5, 5) == 1.5

    second = study.ask()
    assert second.suggest_float("x", -5, 5) == 2.5
    third = study.ask()
    assert -5 <= third.suggest_float("x", -5, 5) <= 5

    partial = oa.AgentSampler(
        backend="claude", effort="medium", n_init=4, context="cumulative best-so-far error", seed=0,
        anchor_proposals=[{"x": 1.5, "y": 2.5}],
    )
    partial_study = oa.create_study(sampler=partial, seed=0)
    t = partial_study.ask()
    t.suggest_float("x", -5, 5)
    partial_study.tell(t, 1.0)
    assert partial_study.sampler.propose(partial_study) == {}


def test_pruner():
    def fake_call(backend, model, prompt, timeout, effort=None):
        return '{"prune": true}'

    original = agent.call_agent
    from optim_agent import pruners
    pruners._agent.call_agent = fake_call
    try:
        study = oa.create_study(pruner=oa.AgentPruner(level="tight"), seed=0)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            for step in range(10):
                trial.report(x + step, step)
                if trial.should_prune():
                    raise oa.TrialPruned()
            return x

        study.optimize(objective, n_trials=3, verbose=False)
        states = [t.state for t in study.trials]
        assert states[0] == "complete"          # nothing to compare against yet
        assert "pruned" in states[1:]           # agent said prune
        pruned = next(t for t in study.trials if t.state == "pruned")
        assert pruned.value == pruned.intermediate[-1][1]
    finally:
        pruners._agent.call_agent = original


def test_pruner_forwards_reasoning_effort(monkeypatch):
    from optim_agent import pruners

    calls = []

    def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return '{"prune": false}'

    monkeypatch.setattr(pruners._agent, "call_agent", fake_call)
    study = oa.create_study(
        pruner=oa.AgentPruner(
            backend="codex", model="gpt-5.5", level="tight", effort="high",
        ),
        seed=0,
    )
    completed = study.ask({"x": 0.1})
    completed.suggest_float("x", 0.0, 1.0)
    completed.report(1.0, 0)
    study.tell(completed, 1.0)
    current = study.ask({"x": 0.9})
    current.suggest_float("x", 0.0, 1.0)
    current.report(3.0, 0)

    assert current.should_prune() is False
    assert calls[0][1]["effort"] == "high"


def test_pruner_fail_closed_rejects_missing_decision(monkeypatch):
    from optim_agent import pruners

    monkeypatch.setattr(pruners._agent, "call_agent", lambda *a, **k: "not json")
    study = oa.create_study(
        pruner=oa.AgentPruner(backend="codex", level="tight", fail_closed=True),
        seed=0,
    )
    completed = study.ask({"x": 0.1})
    completed.suggest_float("x", 0.0, 1.0)
    completed.report(1.0, 0)
    study.tell(completed, 1.0)
    current = study.ask({"x": 0.9})
    current.suggest_float("x", 0.0, 1.0)
    current.report(3.0, 0)

    with pytest.raises(ValueError, match="valid prune decision"):
        current.should_prune()


def test_mock_backend_and_storage():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "study.json"
        s = oa.AgentSampler(backend="mock", n_init=2, seed=3)
        study = oa.create_study(sampler=s, storage=path, seed=3)
        study.optimize(quadratic, n_trials=15, verbose=False)
        assert study.best_value < 0.5
        resumed = oa.create_study(storage=path, seed=4)
        assert len(resumed.trials) == 15
        assert resumed.space["x"].high == 5
        resumed.optimize(quadratic, n_trials=1, verbose=False)
        assert len(resumed.trials) == 16
        assert json.loads(path.read_text())["trials"][-1]["number"] == 15


def test_concurrency_and_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "study.db"
        s = oa.AgentSampler(backend="mock", n_init=2, seed=5)
        study = oa.create_study(sampler=s, storage=path, seed=5, max_concurrency=4)
        study.optimize(quadratic, n_trials=20, verbose=False)
        assert len(study.trials) == 20
        assert all(-5 <= t.params["x"] <= 5 for t in study.trials)
        assert study.best_value < 1.0
        # a second process sharing the same DB sees all history, then extends it
        resumed = oa.create_study(storage=path, seed=6)
        assert len(resumed.trials) == 20
        resumed.optimize(quadratic, n_trials=1, verbose=False)
        nums = [t.number for t in oa.create_study(storage=path).trials]
        assert len(nums) == len(set(nums)) == 21  # AUTOINCREMENT keeps numbers unique


def test_skill_mode_ask_tell():
    study = oa.create_study()
    trial = study.ask({"lr": 3e-4, "batch": 64})  # session agent picks the point
    assert trial.params == {"lr": 3e-4, "batch": 64}
    study.tell(trial, 0.42)
    assert study.best_value == 0.42


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def test_hostile_agent_values():
    from optim_agent.space import Categorical, Float, Int
    f = Float(0.0, 1.0)
    assert _raises(ValueError, lambda: f.validate(float("nan")))
    assert _raises(ValueError, lambda: f.validate("garbage"))
    assert _raises((ValueError, OverflowError), lambda: Int(1, 10).validate(float("inf")))
    assert Categorical((1, "a")).validate(1.0) == 1  # canonical choice, agent's spelling dropped
    assert type(Categorical((1, "a")).validate(1.0)) is int
    assert _raises(ValueError, lambda: Float(0.0, 1.0, log=True))
    assert _raises(ValueError, lambda: Float(2.0, 1.0))


def test_guardrails():
    # changed bounds mid-study fail loudly
    study = oa.create_study(seed=0)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1, verbose=False)
    t = study.ask()
    assert _raises(ValueError, lambda: t.suggest_float("x", 0, 2))
    # bad state / valueless complete fail loudly
    assert _raises(ValueError, lambda: study.tell(t, 1.0, state="finished"))
    assert _raises(ValueError, lambda: study.tell(study.ask()))
    # explicit ask(params) is validated once the space is known
    t2 = study.ask({"x": 999})
    assert t2.suggest_float("x", 0, 1) == 1  # clamped
    # numpy scalars unwrap so storage round-trips faithfully
    try:
        import numpy as np
        assert type(oa.create_study().ask({"lr": np.float32(0.1)}).params["lr"]) is float
    except ImportError:
        pass


def test_resume_no_replay():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        study = oa.create_study(storage=path, seed=7)
        study.optimize(quadratic, n_trials=3, verbose=False)
        first = [t.params["x"] for t in study.trials]
        resumed = oa.create_study(storage=path, seed=7)  # same seed: must not replay
        resumed.optimize(quadratic, n_trials=3, verbose=False)
        fresh = [t.params["x"] for t in resumed.trials[3:]]
        assert not set(first) & set(fresh)
        assert [t.number for t in resumed.trials] == list(range(6))
        assert _raises(ValueError, lambda: oa.create_study(direction="maximize", storage=path))


def test_mnist_helper_curves_and_labels():
    from examples import mnist
    import numpy as np

    assert mnist._sanitize_label("GPT-5.5") == "GPT-5.5"
    assert mnist._sanitize_label("agent/mock") == "agent-mock"
    assert mnist._best_error_curve([{"test_error": 9.0}, {"test_error": 7.5},
                                    {"test_error": 8.0}]) == [9.0, 7.5, 7.5]
    assert mnist._device_for_trial(10, [0, 1, 2]) == "cuda:1"
    assert mnist._device_for_trial(3, []) == "cpu"
    assert mnist._run_label("codex", "high") == "GPT-5.5-high"
    assert mnist._run_label("codex-no-context", "high") == "GPT-5.5-high-no-context"
    assert mnist._run_label("Random", "high") == "Random"
    assert mnist._run_label("TPE", "high") == "TPE"
    assert "mock" not in mnist.PLOT_LABELS
    assert mnist.PLOT_EFFORT == "medium"
    assert mnist.PLOT_LABELS == (
        "Random", "TPE", mnist.PLOT_AGENT_LABEL, mnist.PLOT_NO_CONTEXT_LABEL,
    )
    assert set(mnist.PLOT_STYLES) == {mnist.PLOT_AGENT_LABEL, mnist.PLOT_NO_CONTEXT_LABEL}
    assert mnist.PLOT_STYLES[mnist.PLOT_AGENT_LABEL]["color"] == "#009E73"
    assert mnist.PLOT_STYLES[mnist.PLOT_NO_CONTEXT_LABEL]["color"] == "#D55E00"
    pytest.importorskip("torch", reason="install the vision extra to test tensor helpers")
    fake = type("FakeMNIST", (), {
        "data": np.zeros((2, 28, 28), dtype="uint8"),
        "targets": [1, 2],
    })()
    x, y = mnist._tensor_dataset(fake).tensors
    assert tuple(x.shape) == (2, 1, 28, 28)
    assert y.tolist() == [1, 2]
    assert mnist.ResNet.make(16, 32, 64, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)(x).shape == (2, 10)
    assert mnist.STAGE1_WIDTHS == [8, 16, 24, 32, 48, 64]
    assert mnist.STAGE2_WIDTHS == [16, 32, 48, 64, 96, 128]
    assert mnist.STAGE3_WIDTHS == [32, 64, 96, 128, 160, 192]
    assert mnist.DEPTHS == [1, 2, 3]
    assert mnist.SHIFTS == [0, 1, 2, 3]
    assert mnist.ROTATIONS == [0, 5, 10]
    seen = {}

    class Trial:
        number = 0
        params = seen

        def suggest_float(self, name, low, high, *, log=False, context=None):
            seen[name] = low
            return low

        def suggest_categorical(self, name, choices, *, context=None):
            seen[name] = choices[0]
            return choices[0]

        def report(self, value, step):
            pass

    old = mnist._train_once
    mnist._train_once = lambda params, device, epochs, seed: {
        "test_error": 1.0, "test_acc": 99.0, "test_loss": 0.1, "history": [{"epoch": 1, "test_error": 1.0}],
    }
    try:
        assert mnist._objective(1, 0, [])(Trial()) == 1.0
    finally:
        mnist._train_once = old
    assert set(seen) == {
        "lr", "batch_size", "weight_decay", "label_smoothing",
        "stage1_width", "stage2_width", "stage3_width",
        "stage1_depth", "stage2_depth", "stage3_depth",
        "stem_dropout", "stage1_dropout", "stage2_dropout", "head_dropout",
        "aug_shift", "aug_rotate",
    }
    contexts = []

    class ContextTrial(Trial):
        def suggest_float(self, name, low, high, *, log=False, context=None):
            contexts.append(context)
            return low

        def suggest_categorical(self, name, choices, *, context=None):
            contexts.append(context)
            return choices[0]

    mnist._train_once = lambda params, device, epochs, seed: {
        "test_error": 1.0, "test_acc": 99.0, "test_loss": 0.1, "history": [{"epoch": 1, "test_error": 1.0}],
    }
    try:
        assert mnist._objective(1, 0, [], use_context=False)(ContextTrial()) == 1.0
    finally:
        mnist._train_once = old
    assert contexts and all(c is None for c in contexts)
    assert set(mnist._sampler("codex", 0, "medium", 1, None).initial_space) == set(seen)
    mnist_context = mnist._sampler("codex", 0, "medium", 1, None).context
    assert "trial budget" in mnist_context and "24 trials" not in mnist_context
    assert mnist._sampler("codex-no-context", 0, "high", 1, None).context is None


def test_mnist_optuna_trial_adapter_ignores_context():
    from examples import mnist

    class FakeOptunaTrial:
        number = 4
        params = {}

        def suggest_float(self, name, low, high, *, log=False):
            self.params[name] = low
            return low

        def suggest_categorical(self, name, choices):
            self.params[name] = choices[0]
            return choices[0]

        def report(self, value, step):
            self.reported = (value, step)

    t = mnist._OptunaTrialAdapter(FakeOptunaTrial())
    assert t.suggest_float("lr", 1e-4, 3e-2, log=True, context="ignored") == 1e-4
    assert t.suggest_categorical("batch_size", [64, 128], context="ignored") == 64
    t.report(1.2, 3)
    assert t.params == {"lr": 1e-4, "batch_size": 64}
    assert t.number == 4


def test_mnist_trial_record_serializes_metrics():
    from examples import mnist

    study = oa.create_study()
    trial = study.ask({"lr": 0.001, "batch_size": 128, "dropout": 0.2, "width": 32})
    trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    trial.suggest_float("dropout", 0.0, 0.6)
    trial.suggest_categorical("width", [16, 32, 64, 96, 128])
    metrics = {"test_error": 2.5, "test_acc": 97.5, "test_loss": 0.08,
               "history": [{"epoch": 1, "test_error": 2.5}]}

    rec = mnist._trial_record(trial, metrics)

    assert rec["params"]["batch_size"] == 128
    assert rec["test_error"] == 2.5
    assert rec["history"][0]["epoch"] == 1


def test_verify_classification_cumulative_error_contract():
    from scripts import verify_classification_cumulative_error as verify
    import threading
    from types import SimpleNamespace

    assert verify.TRIALS == 10
    assert verify.SEEDS == (0, 1, 2, 3, 4)
    assert verify.RUN_ROOT.name == "classification-stagewise16-v2-n10-s5"
    assert verify.MODEL == "gpt-5.5"
    assert verify.EFFORT == "medium"
    assert verify.LABELS["codex-no-context"] == "GPT-5.5-medium-no-context"
    assert verify.GPT_NO_CONTEXT.name == "gpt-no-context"
    assert verify._incumbent_error_curve([3.0, 4.0, 2.0]) == [3.0, 3.0, 2.0]
    assert verify._dataset_module("mnist").__name__ == "examples.mnist"
    commands = [verify._worker_command(dataset, "codex", verify.GPT_CURRENT, seed)
                for dataset in verify.GPU_SPLITS for seed in verify.SEEDS]
    assert len(commands) == 10
    assert {(cmd[cmd.index("--dataset") + 1], int(cmd[cmd.index("--seed") + 1]))
            for cmd in commands} == {(dataset, seed) for dataset in verify.GPU_SPLITS
                                      for seed in verify.SEEDS}
    assert all(gpus == tuple(range(8)) for gpus in verify.GPU_SPLITS.values())
    for cmd in commands:
        dataset = cmd[cmd.index("--dataset") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        gpus = verify.GPU_SPLITS[dataset]
        offset = seed * (verify.WORKERS - 1) % len(gpus)
        expected = gpus[offset:] + gpus[:offset]
        assert cmd[cmd.index("--gpus") + 1:] == list(map(str, expected))

    barrier = threading.Barrier(len(commands), timeout=5)
    calls = []
    old_run_command = verify._run_command
    verify._run_command = lambda command: (calls.append(command), barrier.wait())
    try:
        verify._run_pair("codex", verify.GPT_CURRENT)
    finally:
        verify._run_command = old_run_command
    assert len(calls) == len(commands)

    seen = {}

    class FakeModule:
        _sampler = staticmethod(lambda method, *args: (
            seen.update(sampler_method=method) or SimpleNamespace(anchor_proposals=[])
        ))
        run = staticmethod(lambda method, seeds, *args: seen.update(method=method, seeds=seeds))

    old_dataset_module = verify._dataset_module
    verify._dataset_module = lambda dataset: FakeModule
    try:
        verify._worker(SimpleNamespace(dataset="mnist", method="codex-no-context", seed=3,
                                       assets="/tmp/assets", storage="/tmp/storage", gpus=[0]))
    finally:
        verify._dataset_module = old_dataset_module
    assert seen == {"sampler_method": "codex-no-context",
                    "method": "codex-no-context", "seeds": [3]}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        label = "Random"
        for seed in verify.SEEDS:
            path = verify._curve_path(root, "cifar10", label, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "label": label, "method": "Random", "seed": seed, "trials": verify.TRIALS,
                "space_version": "old-space",
                "records": [{"test_error": 1.0} for _ in range(verify.TRIALS)],
            }))
        assert not verify._complete(root, "cifar10", label)
        for path in (root / "cifar10").glob("*.json"):
            data = json.loads(path.read_text())
            data["space_version"] = verify._dataset_module("cifar10").SPACE_VERSION
            path.write_text(json.dumps(data))
        assert verify._complete(root, "cifar10", label)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        label = "GPT-5.5-medium"
        for seed in verify.SEEDS:
            path = verify._curve_path(root, "mnist", label, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "label": label, "method": "codex", "model": None,
                "effort": verify.EFFORT, "seed": seed, "trials": verify.TRIALS,
                "space_version": getattr(verify._dataset_module("mnist"), "SPACE_VERSION", None),
                "records": [{"test_error": 1.0} for _ in range(verify.TRIALS)],
            }))
        assert not verify._complete(root, "mnist", label)
        for path in (root / "mnist").glob("*.json"):
            data = json.loads(path.read_text())
            data["model"] = verify.MODEL
            path.write_text(json.dumps(data))
        assert verify._complete(root, "mnist", label)


def test_cifar10_helper_curves_and_labels():
    from examples import cifar10
    import numpy as np

    assert cifar10.DATA.name == "cifar10-kaggle"
    assert cifar10._run_label("codex", "medium") == "GPT-5.5-medium"
    assert cifar10._run_label("codex-no-context", "medium") == "GPT-5.5-medium-no-context"
    assert cifar10._run_label("Random", "high") == "Random"
    assert cifar10._run_label("TPE", "high") == "TPE"
    assert cifar10.PLOT_EFFORT == "medium"
    assert cifar10.PLOT_LABELS == (
        "Random", "TPE", cifar10.PLOT_AGENT_LABEL, cifar10.PLOT_NO_CONTEXT_LABEL,
    )
    assert cifar10.PLOT_STYLES[cifar10.PLOT_AGENT_LABEL]["color"] == "#009E73"
    assert cifar10.PLOT_STYLES[cifar10.PLOT_NO_CONTEXT_LABEL]["color"] == "#D55E00"
    assert cifar10._best_error_curve([{"test_error": 70.0}, {"test_error": 75.0},
                                      {"test_error": 60.0}]) == [70.0, 70.0, 60.0]
    pytest.importorskip("torch", reason="install the vision extra to test tensor helpers")
    fake = type("FakeCIFAR", (), {
        "data": np.zeros((2, 32, 32, 3), dtype="uint8"),
        "targets": [1, 2],
    })()
    x, y = cifar10._tensor_dataset(fake).tensors
    assert tuple(x.shape) == (2, 3, 32, 32)
    assert y.tolist() == [1, 2]
    assert cifar10.ResNet.make(64, 128, 256, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)(x).shape == (2, 10)
    assert cifar10.BATCHES == [64, 128]
    assert cifar10.STAGE1_WIDTHS == [64, 96, 128, 160]
    assert cifar10.STAGE2_WIDTHS == [128, 192, 256, 320]
    assert cifar10.STAGE3_WIDTHS == [256, 384, 512, 640]
    assert cifar10.DEPTHS == [1, 2, 3]
    assert cifar10.CROP_PADS == [4, 6]
    assert cifar10.FLIP_PROBS == [0.0, 0.5]
    assert cifar10.SPACE_VERSION == "cifar10-stagewise-16-v2"
    seen = {}

    class Trial:
        number = 0
        params = seen

        def suggest_float(self, name, low, high, *, log=False, context=None):
            seen[name] = low
            return low

        def suggest_categorical(self, name, choices, *, context=None):
            seen[name] = choices[0]
            return choices[0]

        def report(self, value, step):
            pass

    old = cifar10._train_once
    cifar10._train_once = lambda params, device, epochs, seed: {
        "test_error": 1.0, "test_acc": 99.0, "test_loss": 0.1, "history": [{"epoch": 1, "test_error": 1.0}],
    }
    try:
        assert cifar10._objective(1, 0, [])(Trial()) == 1.0
    finally:
        cifar10._train_once = old
    assert set(seen) == {
        "lr", "batch_size", "weight_decay", "label_smoothing",
        "stage1_width", "stage2_width", "stage3_width",
        "stage1_depth", "stage2_depth", "stage3_depth",
        "stem_dropout", "stage1_dropout", "stage2_dropout", "head_dropout",
        "aug_crop", "aug_flip",
    }
    contexts = []

    class ContextTrial(Trial):
        def suggest_float(self, name, low, high, *, log=False, context=None):
            contexts.append(context)
            return low

        def suggest_categorical(self, name, choices, *, context=None):
            contexts.append(context)
            return choices[0]

    cifar10._train_once = lambda params, device, epochs, seed: {
        "test_error": 1.0, "test_acc": 99.0, "test_loss": 0.1, "history": [{"epoch": 1, "test_error": 1.0}],
    }
    try:
        assert cifar10._objective(1, 0, [], use_context=False)(ContextTrial()) == 1.0
    finally:
        cifar10._train_once = old
    assert contexts and all(c is None for c in contexts)
    assert set(cifar10._sampler("codex", 0, "medium", 1, None).initial_space) == set(seen)
    assert cifar10._sampler("codex-no-context", 0, "medium", 1, None).context is None


def test_hard_functions_distributed_contract():
    from examples import hard_functions as hard
    import threading

    expected = {
        "Random": (None, None, "both"),
        "TPE": ("tpe", None, "both"),
        "GPT-5.5": ("codex", "gpt-5.5", "tier"),
        "Opus-4.8": ("claude", "claude-opus-4-8", "tier"),
        "Sonnet-5": ("claude", "claude-sonnet-5", "tier"),
        "GLM-5.2": ("opencode", "glm-5.2", "tier"),
        "Big-pickle": ("opencode", "opencode/big-pickle", "free"),
        "DeepSeek-V4-Flash": (
            "opencode", "opencode/deepseek-v4-flash-free", "free",
        ),
        "Nemotron-3-Ultra": (
            "opencode", "opencode/nemotron-3-ultra-free", "free",
        ),
        "MiMo-v2.5": ("opencode", "opencode/mimo-v2.5-free", "free"),
    }
    assert {
        label: (preset["backend"], preset["model"], preset["group"])
        for label, preset in hard.POOL.items()
    } == expected

    for preset in hard.POOL.values():
        if preset["backend"] in (None, "tpe"):
            continue
        sampler = hard._make_sampler(
            preset, seed=3, timeout=17, agent_cwd="/tmp/hard-functions-empty",
        )
        assert sampler.backend == preset["backend"]
        assert sampler.model == preset["model"]
        assert sampler.effort == "medium"
        assert sampler.context is None
        assert sampler.n_init == 3
        assert sampler.timeout == 17
        assert sampler.agent_cwd == "/tmp/hard-functions-empty"

    seeds = [0, 1, 2, 3, 4]
    assert hard._seed_workers("Random", seeds) == 5
    assert hard._seed_workers("GPT-5.5", seeds) == 5
    assert hard._seed_workers("Opus-4.8", seeds) == 5
    assert hard._seed_workers("GLM-5.2", seeds) == 1
    assert hard._seed_workers("Big-pickle", seeds) == 1

    barrier = threading.Barrier(5, timeout=5)
    calls = []
    old_run = hard.run
    hard.run = lambda label, trials, seed, timeout: (
        calls.append((label, trials, seed, timeout)), barrier.wait()
    )
    try:
        hard.run_distributed(["Random"], 10, [0, 1, 2, 3, 4], 600)
    finally:
        hard.run = old_run
    assert {call[2] for call in calls} == {0, 1, 2, 3, 4}

    hard.run = lambda label, trials, seed, timeout: (
        (_ for _ in ()).throw(RuntimeError("worker failed")) if seed == 2 else None
    )
    try:
        try:
            hard.run_distributed(["Random"], 10, [0, 1, 2, 3, 4], 600)
        except RuntimeError as error:
            assert str(error) == "worker failed"
        else:
            raise AssertionError("distributed worker failure was swallowed")
    finally:
        hard.run = old_run


def test_hard_functions_preflight_contract():
    from examples import hard_functions as hard

    calls = []
    old_call = hard.agent_api.call_agent
    hard.agent_api.call_agent = lambda backend, model, prompt, timeout, effort, cwd: (
        calls.append((backend, model, timeout, effort, Path(cwd).is_dir()))
        or '{"ok": true}'
    )
    try:
        hard.preflight(timeout=23)
    finally:
        hard.agent_api.call_agent = old_call

    expected = [
        (preset["backend"], preset["model"], 23, "medium", True)
        for preset in hard.POOL.values()
        if preset["backend"] not in (None, "tpe")
    ]
    assert calls == expected


def test_plotters_reject_incomplete_publication_data():
    from examples import cifar10, hard_functions, mnist

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_mnist, old_cifar, old_hard = mnist.ASSETS, cifar10.ASSETS, hard_functions.ASSETS
        mnist.ASSETS = cifar10.ASSETS = hard_functions.ASSETS = root
        try:
            for loader in (mnist._load_plot_runs, cifar10._load_plot_runs,
                           hard_functions._load_plot_runs):
                try:
                    loader()
                except SystemExit:
                    pass
                else:
                    raise AssertionError("incomplete publication data was accepted")
        finally:
            mnist.ASSETS, cifar10.ASSETS, hard_functions.ASSETS = old_mnist, old_cifar, old_hard


if __name__ == "__main__":
    for fn in [test_random_study, test_extract_json,
               test_call_agent_forwards_working_directory,
               test_opencode_command_forwards_working_directory,
               test_agent_sampler,
               test_agent_sampler_retries_transient_call_failure,
               test_cumulative_error_metric_agent_owns_post_startup,
               test_cumulative_error_metric_hands_off_after_schema_trial,
               test_cumulative_error_metric_uses_declared_space_on_first_trial,
               test_cumulative_error_metric_joint_startup_portfolio,
               test_anchor_proposals_seed_warmup,
               test_pruner, test_mock_backend_and_storage, test_concurrency_and_sqlite,
               test_skill_mode_ask_tell, test_hostile_agent_values, test_guardrails,
               test_resume_no_replay, test_mnist_helper_curves_and_labels,
               test_mnist_optuna_trial_adapter_ignores_context,
               test_mnist_trial_record_serializes_metrics,
               test_verify_classification_cumulative_error_contract,
               test_cifar10_helper_curves_and_labels,
               test_hard_functions_distributed_contract,
               test_hard_functions_preflight_contract,
               test_plotters_reject_incomplete_publication_data]:
        fn()
        print(f"ok: {fn.__name__}")
    print("all checks passed")
