import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".html", ".json", ".md", ".py", ".tex", ".toml", ".yaml", ".yml"}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".worktrees",
    "autoresearch-results",
    "build",
    "dist",
    "graphify-out",
    "ship",
}


def test_classification_metric_uses_error_terminology_everywhere():
    old_term = "re" + "ward"
    pattern = re.compile(rf"\b{old_term}s?\b", re.IGNORECASE)
    violations = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        if old_term in path.name.lower():
            violations.append(str(path.relative_to(ROOT)))
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if pattern.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert not violations, "obsolete classification metric term:\n" + "\n".join(violations)
