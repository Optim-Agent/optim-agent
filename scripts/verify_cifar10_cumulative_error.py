#!/usr/bin/env python3
"""Run CIFAR-10 GPT-medium and print 12-trial cumulative-error-ratio metrics JSON."""

import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import cifar10


REFERENCE = ROOT / ".optim-agent-runs" / "cifar10-reference"
OUTPUT = ROOT / ".optim-agent-runs" / "cifar10-verify" / "assets"
STORAGE = ROOT / ".optim-agent-runs" / "cifar10-verify" / "storage"
SEEDS = (0, 1, 2)
TRIALS = 12
MODEL = "gpt-5.5"


def _safe(label):
    return "".join(c if c.isalnum() or c in "_.-" else "-" for c in label).strip("-")


def _clean_gpt():
    for root in (OUTPUT, STORAGE):
        for path in root.glob("cifar10_*GPT-5.5-medium_s*.json"):
            path.unlink()


def _cumulative_error(root, label):
    cumulative_errors = []
    safe = _safe(label)
    for seed in SEEDS:
        data = json.loads((root / f"cifar10_curves_{safe}_s{seed}.json").read_text())
        best, curve = math.inf, []
        for row in data["records"]:
            if row.get("state", "complete") == "complete" and row.get("test_error") is not None:
                best = min(best, float(row["test_error"]))
            curve.append(best)
        if len(curve) < TRIALS:
            raise SystemExit(f"{label} seed {seed} has {len(curve)} trials")
        cumulative_errors.append(sum(curve[:TRIALS]))
    return statistics.mean(cumulative_errors), cumulative_errors


def main():
    if not REFERENCE.is_dir():
        raise SystemExit(f"missing fixed Random/TPE reference curves in {REFERENCE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    _clean_gpt()
    random_cumulative_error, random_seeds = _cumulative_error(REFERENCE, "Random")
    tpe_cumulative_error, tpe_seeds = _cumulative_error(REFERENCE, "TPE")
    cifar10.ASSETS = OUTPUT
    cifar10.STORAGE = STORAGE
    cifar10.run("codex", list(SEEDS), TRIALS, 3, 8, list(range(8)), "medium", 600, MODEL)
    gpt_cumulative_error, gpt_seeds = _cumulative_error(OUTPUT, "GPT-5.5-medium")
    baseline = min(random_cumulative_error, tpe_cumulative_error)
    print(json.dumps({
        "ratio": gpt_cumulative_error / baseline,
        "gpt_cumulative_error": gpt_cumulative_error,
        "best_baseline_cumulative_error": baseline,
        "random_cumulative_error": random_cumulative_error,
        "tpe_cumulative_error": tpe_cumulative_error,
        **{f"random_s{i}": value for i, value in enumerate(random_seeds)},
        **{f"tpe_s{i}": value for i, value in enumerate(tpe_seeds)},
        **{f"gpt_s{i}": value for i, value in enumerate(gpt_seeds)},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
