"""Offline self-check: `python tests/test_optim_agent.py` (also pytest-compatible)."""

import json
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
    assert _raises(ValueError, lambda: oa.AgentSampler(effort="xhigh"))
    assert _raises(ValueError, lambda: oa.AgentSampler(effort="max"))
    assert samplers.EFFORTS == {
        "low": dict(history=5, reasoning=False, notes=False),
        "medium": dict(history=10, reasoning=True, notes=True),
        "high": dict(history=20, reasoning=True, notes=True),
    }
    low_prompt = s._prompt(study, [t for t in study.trials if t.value is not None],
                           samplers.EFFORTS["low"])
    assert "History summary:" in low_prompt
    assert "Trial history (oldest first):" not in low_prompt
    medium_prompt = s._prompt(study, [t for t in study.trials if t.value is not None],
                              samplers.EFFORTS["medium"])
    assert 'Include a short "_reasoning" field' in medium_prompt
    assert 'Include a "_note" field' in medium_prompt
    assert "History summary:" in medium_prompt
    assert "Promising trials:" in medium_prompt
    assert "Failed or weak regions to avoid:" not in medium_prompt
    assert "Use the task context as priors when available" in medium_prompt
    s.context = "Full MNIST ResNet neural architecture search"
    context_prompt = s._prompt(study, [t for t in study.trials if t.value is not None],
                               samplers.EFFORTS["medium"])
    assert "Context-derived priors:" in context_prompt
    # out-of-range and garbage replies fall back gracefully
    samplers._agent.call_agent = lambda *a, **k: '{"x": 999}'
    try:
        study.optimize(quadratic, n_trials=1, verbose=False)  # clamped to 5
        assert study.trials[-1].params["x"] == 5
    finally:
        samplers._agent.call_agent = original


def test_early_reward_agent_owns_post_startup():
    calls = []
    original = samplers._agent.call_agent
    samplers._agent.call_agent = lambda *args, **kwargs: calls.append(args) or '{"x": 1.5}'
    s = oa.AgentSampler(backend="claude", effort="medium", n_init=1,
                        context="early reward", seed=0)
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


def test_early_reward_hands_off_after_schema_trial():
    calls = []
    original = samplers._agent.call_agent
    samplers._agent.call_agent = lambda *args, **kwargs: calls.append(args) or '{"x": 2.0}'
    try:
        study = oa.create_study(
            sampler=oa.AgentSampler(
                backend="claude", effort="medium", n_init=4,
                context="early reward", seed=0,
            ),
            seed=0,
        )
        first = study.ask()
        first.suggest_float("x", -5, 5)
        second = study.ask()
        assert second.suggest_float("x", -5, 5) == 2.0
    finally:
        samplers._agent.call_agent = original
    assert calls, "early-reward runs should hand off after discovering the search space"


def test_early_reward_uses_declared_space_on_first_trial():
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
            context="early reward", seed=0,
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


def test_early_reward_joint_startup_portfolio():
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
            context="early reward", seed=0,
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
        backend="claude", effort="medium", n_init=4, context="early reward", seed=0,
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
        backend="claude", effort="medium", n_init=4, context="early reward", seed=0,
        anchor_proposals=[{"x": 1.5, "y": 2.5}],
    )
    partial_study = oa.create_study(sampler=partial, seed=0)
    t = partial_study.ask()
    t.suggest_float("x", -5, 5)
    partial_study.tell(t, 1.0)
    assert partial_study.sampler.propose(partial_study) == {}


def test_pruner():
    def fake_call(backend, model, prompt, timeout):
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
    assert mnist.PLOT_LABELS == ("Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context")
    assert set(mnist.PLOT_STYLES) == {"GPT-5.5-medium", "GPT-5.5-medium-no-context"}
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


def test_verify_mnist_prompting_scores_and_prunes():
    from scripts import verify_mnist_prompting as verify

    metrics = verify._score({
        "Random": [0.60, 0.60, 0.60],
        "TPE": [0.56, 0.57, 0.56],
        "GPT-5.5-medium": [0.40, 0.46, 0.50],
        "GPT-5.5-medium-no-context": [0.50, 0.52, 0.51],
    })
    assert metrics["margin_vs_best_random_tpe"] > 0.10
    assert metrics["context_margin_vs_no_context"] > 0.05
    assert metrics["success"] == 1.0

    target1_metrics = verify._score({
        "Random": [0.60, 0.60, 0.60],
        "TPE": [0.56, 0.57, 0.56],
        "GPT-5.5-medium": [0.53, 0.60, 0.50],
    })
    assert verify._target1_failed(target1_metrics)
    assert not verify._seed_is_close_enough([0.431], [0.40])
    assert verify._seed_is_close_enough([0.43], [0.40])
    assert verify._reference_context_vals_from_metrics(
        {"context_s0": 0.40, "context_s1": 0.46, "context_s2": 0.53},
        {"context_s0": 0.56},
    ) == [0.56, 0.46, 0.53]
    assert verify._previous_context_metrics(
        {"last_status": "refine", "last_trial_metrics": {"context_s0": 0.40}},
        [0.56],
    ) == {"context_s0": 0.56}


def test_verify_mnist_reward_safe_label():
    from scripts import verify_mnist_reward as verify

    assert verify._safe("GPT-5.5 medium/no ctx") == "GPT-5.5-medium-no-ctx"


def test_verify_classification_reward_contract():
    from scripts import verify_classification_reward as verify
    import threading
    from types import SimpleNamespace

    assert verify.TRIALS == 10
    assert verify.SEEDS == (0, 1, 2, 3, 4)
    assert verify.RUN_ROOT.name == "classification-stagewise16-v2-n10-s5"
    assert verify.MODEL == "gpt-5.5"
    assert verify.EFFORT == "medium"
    assert verify.LABELS["codex-no-context"] == "GPT-5.5-medium-no-context"
    assert verify.GPT_NO_CONTEXT.name == "gpt-no-context"
    assert verify._reward_curve([3.0, 4.0, 2.0]) == [3.0, 3.0, 2.0]
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
    assert cifar10.PLOT_LABELS == ("Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context")
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

    assert tuple(hard.POOL) == (
        "Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context",
    )
    assert hard.POOL["GPT-5.5-medium"]["model"] == "gpt-5.5"
    assert hard.POOL["GPT-5.5-medium-no-context"]["no_context"] is True

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
    for fn in [test_random_study, test_extract_json, test_agent_sampler,
               test_early_reward_agent_owns_post_startup,
               test_early_reward_hands_off_after_schema_trial,
               test_early_reward_uses_declared_space_on_first_trial,
               test_early_reward_joint_startup_portfolio,
               test_anchor_proposals_seed_warmup,
               test_pruner, test_mock_backend_and_storage, test_concurrency_and_sqlite,
               test_skill_mode_ask_tell, test_hostile_agent_values, test_guardrails,
               test_resume_no_replay, test_mnist_helper_curves_and_labels,
               test_mnist_optuna_trial_adapter_ignores_context,
               test_mnist_trial_record_serializes_metrics,
               test_verify_mnist_prompting_scores_and_prunes,
               test_verify_mnist_reward_safe_label,
               test_verify_classification_reward_contract,
               test_cifar10_helper_curves_and_labels,
               test_hard_functions_distributed_contract,
               test_plotters_reject_incomplete_publication_data]:
        fn()
        print(f"ok: {fn.__name__}")
    print("all checks passed")
