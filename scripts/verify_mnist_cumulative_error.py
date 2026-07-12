#!/usr/bin/env python3
"""Run MNIST GPT-medium and print cumulative-error-ratio metrics JSON."""

import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
BASELINE = ROOT / "autoresearch-results" / "mnist-cumulative-error-baseline"
STORAGE = ROOT / ".optim-agent-runs" / "mnist"
SEEDS = ("0", "1", "2")


def _safe(label):
    return "".join(c if c.isalnum() or c in "_.-" else "-" for c in label).strip("-")


def _copy_baselines():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for path in BASELINE.glob("mnist_curves_*.json"):
        shutil.copy2(path, ASSETS / path.name)


def _clean_gpt():
    for root in (ASSETS, STORAGE):
        for path in root.glob("mnist_*GPT-5.5-medium_s*.json"):
            path.unlink()


def _run_gpt(seed):
    subprocess.run([
        sys.executable, "examples/mnist.py", "run",
        "--method", "codex", "--effort", "medium", "--seeds", seed,
        "--trials", "24", "--epochs", "3", "--workers", "8",
        "--gpus", "0", "1", "2", "3", "4", "5", "6", "7",
        "--timeout", "600",
    ], cwd=ROOT, check=True)


def _cumulative_error(label):
    cumulative_errors = []
    safe = _safe(label)
    for seed in SEEDS:
        data = json.loads((ASSETS / f"mnist_curves_{safe}_s{seed}.json").read_text())
        best, curve = math.inf, []
        for row in data["records"]:
            if row.get("state", "complete") == "complete" and row.get("test_error") is not None:
                best = min(best, float(row["test_error"]))
            curve.append(best)
        if len(curve) < 24:
            raise SystemExit(f"{label} seed {seed} has {len(curve)} trials")
        cumulative_errors.append(sum(curve[:24]))
    return statistics.mean(cumulative_errors), cumulative_errors


def main():
    _copy_baselines()
    _clean_gpt()
    random_cumulative_error, random_seeds = _cumulative_error("Random")
    tpe_cumulative_error, tpe_seeds = _cumulative_error("TPE")
    for seed in SEEDS:
        _run_gpt(seed)
    gpt_cumulative_error, gpt_seeds = _cumulative_error("GPT-5.5-medium")
    best = min(random_cumulative_error, tpe_cumulative_error)
    print(json.dumps({
        "ratio": gpt_cumulative_error / best,
        "random_cumulative_error": random_cumulative_error,
        "tpe_cumulative_error": tpe_cumulative_error,
        "gpt_cumulative_error": gpt_cumulative_error,
        "best_baseline_cumulative_error": best,
        "random_s0": random_seeds[0],
        "random_s1": random_seeds[1],
        "random_s2": random_seeds[2],
        "tpe_s0": tpe_seeds[0],
        "tpe_s1": tpe_seeds[1],
        "tpe_s2": tpe_seeds[2],
        "gpt_s0": gpt_seeds[0],
        "gpt_s1": gpt_seeds[1],
        "gpt_s2": gpt_seeds[2],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
