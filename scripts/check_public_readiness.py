#!/usr/bin/env python3
"""Emit the number of mechanically verifiable public-readiness gates passed."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _text(path):
    candidate = ROOT / path
    return candidate.read_text(encoding="utf-8", errors="replace") if candidate.exists() else ""


def _command_passes(*command):
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except OSError:
        return False


def _ignored(path):
    return _command_passes("git", "check-ignore", "--no-index", "--quiet", path)


def _python_candidates():
    candidates = [sys.executable]
    candidates.extend(shutil.which(name) for name in (
        "python", "python3", "python3.14", "python3.13", "python3.12",
        "python3.11", "python3.10", "python3.9",
    ))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _python_with_module(module):
    for candidate in _python_candidates():
        if candidate and _command_passes(candidate, "-c", f"import {module}.__main__"):
            return candidate
    return None


def _python_with_passing_module_command(module, *args):
    for candidate in _python_candidates():
        if not _command_passes(candidate, "-c", f"import {module}.__main__"):
            continue
        return candidate if _command_passes(candidate, "-m", module, *args) else None
    return None


def _has_cost_latency_evidence():
    for path in (ROOT / "docs" / "assets").glob("*curves*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        serialized = json.dumps(data)
        if "token_cost" in serialized and "latency" in serialized:
            return True
    return False


def evaluate():
    pyproject = _text("pyproject.toml")
    readme = _text("README.md")
    ci = _text(".github/workflows/ci.yml")
    docs_ci = _text(".github/workflows/docs.yml")
    verifier = _text("scripts/verify_classification_cumulative_error.py")
    pytest_python = _python_with_passing_module_command("pytest", "-q")
    build_python = _python_with_module("build")
    github_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / ".github").rglob("*")
        if path.is_file()
    )

    return {
        "license": (ROOT / "LICENSE").exists(),
        "package_name": 'name = "optim-agent"' in pyproject,
        "package_version": bool(re.search(r'version\s*=\s*"\d+\.\d+\.\d+"', pyproject)),
        "python_floor": 'requires-python = ">=3.9"' in pyproject,
        "project_urls": "[project.urls]" in pyproject and "Documentation =" in pyproject,
        "dev_extra": "dev =" in pyproject,
        "vision_extra": "vision =" in pyproject,
        "ml_extra": "ml =" in pyproject,
        "ci_workflow": (ROOT / ".github/workflows/ci.yml").exists(),
        "ci_python39": '"3.9"' in ci,
        "ci_python13": '"3.13"' in ci,
        "docs_workflow": (ROOT / ".github/workflows/docs.yml").exists() and "deploy-pages" in docs_ci,
        "release_workflow": any((ROOT / ".github/workflows").glob("*release*")),
        "precommit": (ROOT / ".pre-commit-config.yaml").exists(),
        "lint_gate": bool(re.search(r"\[tool\.(ruff|flake8|pylint)", pyproject)
                          or re.search(r"\b(ruff|flake8|pylint)\b", ci)),
        "type_gate": "[tool.mypy]" in pyproject or bool(re.search(r"\b(mypy|pyright)\b", ci)),
        "coverage_gate": bool(re.search(r"coverage|pytest-cov", pyproject + ci)),
        "dependency_scan": (ROOT / ".github/dependabot.yml").exists()
        or bool(re.search(r"pip-audit|dependabot|dependency-review", github_text)),
        "readme_quickstart": "## Quickstart" in readme,
        "offline_mode": "### Offline testing" in readme,
        "troubleshooting": "## Troubleshooting" in readme,
        "trajectory_animation": "optimization_trajectory.gif" in readme
        and (ROOT / "docs/assets/optimization_trajectory.gif").exists(),
        "benchmark_section": "## Benchmarks:" in readme,
        "chinese_readme": (ROOT / "docs/i18n/README_ZH.md").exists()
        and "docs/i18n/README_ZH.md" in readme,
        "notebook_tutorial": (ROOT / "tutorials/quickstart.ipynb").exists(),
        "one_click_notebook": bool(re.search(r"colab|binder", readme, re.IGNORECASE)),
        "api_reference": (ROOT / "docs/reference/api.md").exists(),
        "backend_extension_guide": (ROOT / "docs/reference/backends.md").exists(),
        "quant_example": (ROOT / "examples/quant_walk_forward.py").exists(),
        "sklearn_example": (ROOT / "examples/sklearn_tuning.py").exists(),
        "inference_example": (ROOT / "examples/inference_tuning.py").exists(),
        "vision_examples": (ROOT / "examples/mnist.py").exists()
        and (ROOT / "examples/cifar10.py").exists(),
        "benchmark_manifest": (ROOT / "benchmarks/manifest.json").exists(),
        "benchmark_contract": (ROOT / "benchmarks/README.md").exists(),
        "classification_verifier_safe": 'getattr(sampler, "anchor_proposals", ())' in verifier,
        "cost_latency_evidence": _has_cost_latency_evidence(),
        "broader_classical_baselines": bool(
            re.search(r"CMA-ES|Gaussian process|GP baseline", readme, re.IGNORECASE)
        ),
        "governance": all((ROOT / name).exists() for name in (
            "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md")),
        "issue_pr_templates": any((ROOT / ".github/ISSUE_TEMPLATE").glob("*"))
        and (ROOT / ".github/pull_request_template.md").exists(),
        "release_research_metadata": all((ROOT / name).exists() for name in (
            "CHANGELOG.md", "ROADMAP.md", "CITATION.cff")),
        "ignore_json_studies": _ignored("study.json") and _ignored("hpo_study.json"),
        "ignore_sqlite_studies": all(_ignored(name) for name in (
            "local.db", "local.sqlite", "local.sqlite3")),
        "ignore_paper_build": all(_ignored(name) for name in (
            "paper/src/main.aux", "paper/src/main.log", "paper/src/main.pdf")),
        "tests_pass": bool(pytest_python),
        "package_builds": bool(build_python) and _command_passes(
            build_python, "-m", "build", "--wheel", "--sdist"
        ),
    }


def main():
    gates = evaluate()
    failed = [name for name, passed in gates.items() if not passed]
    print(f"public readiness: {len(gates) - len(failed)}/{len(gates)}")
    print("failing gates: " + (", ".join(failed) if failed else "none"))
    print(len(gates) - len(failed))


if __name__ == "__main__":
    main()
