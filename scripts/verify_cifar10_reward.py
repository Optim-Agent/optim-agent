#!/usr/bin/env python3
"""Run CIFAR-10 GPT-medium and print 12-trial reward-ratio metrics JSON."""

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


def _reward(root, label):
    rewards = []
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
        rewards.append(sum(curve[:TRIALS]))
    return statistics.mean(rewards), rewards


def main():
    if not REFERENCE.is_dir():
        raise SystemExit(f"missing fixed Random/TPE reference curves in {REFERENCE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    _clean_gpt()
    random_reward, random_seeds = _reward(REFERENCE, "Random")
    tpe_reward, tpe_seeds = _reward(REFERENCE, "TPE")
    cifar10.ASSETS = OUTPUT
    cifar10.STORAGE = STORAGE
    cifar10.run("codex", list(SEEDS), TRIALS, 3, 8, list(range(8)), "medium", 600, MODEL)
    gpt_reward, gpt_seeds = _reward(OUTPUT, "GPT-5.5-medium")
    baseline = min(random_reward, tpe_reward)
    print(json.dumps({
        "ratio": gpt_reward / baseline,
        "gpt_reward": gpt_reward,
        "best_baseline_reward": baseline,
        "random_reward": random_reward,
        "tpe_reward": tpe_reward,
        **{f"random_s{i}": value for i, value in enumerate(random_seeds)},
        **{f"tpe_s{i}": value for i, value in enumerate(tpe_seeds)},
        **{f"gpt_s{i}": value for i, value in enumerate(gpt_seeds)},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
