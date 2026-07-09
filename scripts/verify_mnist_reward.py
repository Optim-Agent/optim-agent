#!/usr/bin/env python3
"""Run MNIST GPT-medium and print reward-ratio metrics JSON."""

import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
BASELINE = ROOT / "autoresearch-results" / "mnist-reward-baseline"
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


def _reward(label):
    rewards = []
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
        rewards.append(sum(curve[:24]))
    return statistics.mean(rewards), rewards


def main():
    _copy_baselines()
    _clean_gpt()
    random_reward, random_seeds = _reward("Random")
    tpe_reward, tpe_seeds = _reward("TPE")
    for seed in SEEDS:
        _run_gpt(seed)
    gpt_reward, gpt_seeds = _reward("GPT-5.5-medium")
    best = min(random_reward, tpe_reward)
    print(json.dumps({
        "ratio": gpt_reward / best,
        "random_reward": random_reward,
        "tpe_reward": tpe_reward,
        "gpt_reward": gpt_reward,
        "best_baseline_reward": best,
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
