import subprocess
import sys
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from scripts import check_public_readiness
from scripts import verify_classification_cumulative_error as classification_cumulative_error


ROOT = Path(__file__).resolve().parent.parent


def test_root_skill_covers_installation_and_study_workflow():
    skill_path = ROOT / "SKILL.md"
    readme = (ROOT / "README.md").read_text()
    docs = (ROOT / "docs/index.html").read_text()

    assert skill_path.exists()
    assert not (ROOT / "skills/optim-agent/SKILL.md").exists()
    skill = skill_path.read_text()
    normalized_skill = " ".join(skill.split())
    for text in (
        "$skill-installer install https://github.com/Optim-Agent/optim-agent",
        "Claude Code",
        "OpenCode/OpenClaw",
        "python -m pip install optim-agent",
        "git+https://github.com/Optim-Agent/optim-agent.git",
        ".optim-agent-runs/",
        "git check-ignore -q .optim-agent-runs/",
        "study.ask",
        "study.tell",
        'state="failed"',
        'state="pruned"',
    ):
        assert text in skill
    assert "does not depend on Codex-only APIs" in normalized_skill
    assert "[optim-agent skill](SKILL.md)" in readme
    assert "/blob/main/SKILL.md" in docs


def test_quickstart_notebook_cells_execute_offline():
    notebook = json.loads((ROOT / "tutorials/quickstart.ipynb").read_text())
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    assert "trials after resume: 15" in result.stdout


def test_classification_worker_accepts_random_baseline(monkeypatch, tmp_path):
    calls = []
    module = SimpleNamespace(
        _sampler=lambda *args: oa.RandomSampler(),
        run=lambda *args: calls.append(args),
        ASSETS=None,
        STORAGE=None,
    )
    monkeypatch.setattr(classification_cumulative_error, "_dataset_module", lambda _: module)

    args = SimpleNamespace(
        dataset="mnist",
        method="Random",
        seed=0,
        assets=str(tmp_path / "assets"),
        storage=str(tmp_path / "storage"),
        gpus=[],
    )

    classification_cumulative_error._worker(args)

    assert len(calls) == 1
    assert module.ASSETS == tmp_path / "assets"
    assert module.STORAGE == tmp_path / "storage"


def test_local_generated_and_draft_outputs_are_ignored():
    local_outputs = (
        ".superpowers/brainstorm/local-screen.html",
        "study.json",
        "hpo_study.json",
        "local.db",
        "local.sqlite",
        "local.sqlite3",
        "docs/superpowers/plans/local-plan.md",
        "paper/README.md",
        "paper/src/main.aux",
        "paper/src/main.log",
        "paper/src/main.pdf",
    )

    for path in local_outputs:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"local output is not ignored: {path}"

    tracked_fixtures = (
        "docs/assets/benchmark_study.json",
        "tests/fixtures/study.db",
    )
    for path in tracked_fixtures:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 1, f"repository fixture would be ignored: {path}"


def test_readiness_checker_uses_project_commands(monkeypatch):
    commands = []
    monkeypatch.setattr(
        check_public_readiness,
        "_command_passes",
        lambda *command: commands.append(command) or True,
    )

    check_public_readiness.evaluate()

    assert any(command[1:] == ("-m", "pytest", "-q") for command in commands)
    assert any(
        command[1:] == ("-m", "build", "--wheel", "--sdist")
        for command in commands
    )


def test_readiness_checker_selects_python_with_build(monkeypatch):
    commands = []
    monkeypatch.setattr(
        check_public_readiness,
        "sys",
        SimpleNamespace(executable="/tools/current-python"),
        raising=False,
    )
    monkeypatch.setattr(
        check_public_readiness,
        "shutil",
        SimpleNamespace(
            which=lambda name: f"/tools/{name}" if name in {"python3", "python3.10"} else None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        check_public_readiness,
        "_command_passes",
        lambda *command: commands.append(command) or command[0] == "/tools/python3.10",
    )

    assert check_public_readiness._python_with_module("build") == "/tools/python3.10"
    assert commands[-1][-1] == "import build.__main__"


def test_readiness_checker_prefers_running_python(monkeypatch):
    monkeypatch.setattr(
        check_public_readiness,
        "sys",
        SimpleNamespace(executable="/venv/python"),
        raising=False,
    )
    monkeypatch.setattr(
        check_public_readiness,
        "shutil",
        SimpleNamespace(which=lambda name: None),
        raising=False,
    )
    monkeypatch.setattr(
        check_public_readiness,
        "_command_passes",
        lambda *command: command[0] == "/venv/python",
    )

    assert check_public_readiness._python_with_module("pytest") == "/venv/python"


def test_readiness_checker_does_not_mask_module_command_failure(monkeypatch):
    monkeypatch.setattr(
        check_public_readiness,
        "sys",
        SimpleNamespace(executable="/tools/current-python"),
    )
    monkeypatch.setattr(
        check_public_readiness,
        "shutil",
        SimpleNamespace(
            which=lambda name: f"/tools/{name}" if name == "python3.10" else None,
        ),
    )

    def command_passes(*command):
        if command[1:3] == ("-c", "import pytest.__main__"):
            return True
        return command[0] == "/tools/python3.10"

    monkeypatch.setattr(check_public_readiness, "_command_passes", command_passes)

    assert check_public_readiness._python_with_passing_module_command(
        "pytest", "-q"
    ) is None


def test_readiness_command_failure_is_a_failed_gate(monkeypatch):
    def missing_command(*args, **kwargs):
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(check_public_readiness.subprocess, "run", missing_command)

    assert check_public_readiness._command_passes("missing-tool", "--version") is False


def test_public_quality_gates_are_configured(monkeypatch):
    pyproject = (ROOT / "pyproject.toml").read_text()
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    precommit = (ROOT / ".pre-commit-config.yaml").read_text()
    dependabot = (ROOT / ".github/dependabot.yml").read_text()

    for dependency in ("build", "pre-commit", "pytest-cov", "ruff"):
        assert dependency in pyproject
    assert "[tool.ruff]" in pyproject
    assert "ruff check" in ci
    assert "--cov-fail-under=85" in ci
    assert "ruff-pre-commit" in precommit
    assert 'package-ecosystem: "pip"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot

    monkeypatch.setattr(check_public_readiness, "_command_passes", lambda *args: True)
    assert check_public_readiness.evaluate()["dependency_scan"] is True
