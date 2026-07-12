import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from scripts import check_public_readiness
from scripts import verify_classification_reward as classification_reward


ROOT = Path(__file__).resolve().parent.parent


def test_classification_worker_accepts_random_baseline(monkeypatch, tmp_path):
    calls = []
    module = SimpleNamespace(
        _sampler=lambda *args: oa.RandomSampler(),
        run=lambda *args: calls.append(args),
        ASSETS=None,
        STORAGE=None,
    )
    monkeypatch.setattr(classification_reward, "_dataset_module", lambda _: module)

    args = SimpleNamespace(
        dataset="mnist",
        method="Random",
        seed=0,
        assets=str(tmp_path / "assets"),
        storage=str(tmp_path / "storage"),
        gpus=[],
    )

    classification_reward._worker(args)

    assert len(calls) == 1
    assert module.ASSETS == tmp_path / "assets"
    assert module.STORAGE == tmp_path / "storage"


def test_local_study_and_paper_outputs_are_ignored():
    local_outputs = (
        "study.json",
        "hpo_study.json",
        "local.db",
        "local.sqlite",
        "local.sqlite3",
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


def test_readiness_checker_uses_project_commands(monkeypatch):
    commands = []
    monkeypatch.setattr(
        check_public_readiness,
        "_command_passes",
        lambda *command: commands.append(command) or True,
    )

    check_public_readiness.evaluate()

    assert ("pytest", "-q") in commands
    assert any(
        command[1:] == ("-m", "build", "--wheel", "--sdist")
        for command in commands
    )


def test_readiness_checker_selects_python_with_build(monkeypatch):
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
        lambda *command: command[0] == "/tools/python3.10",
    )

    assert check_public_readiness._python_with_module("build") == "/tools/python3.10"


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
