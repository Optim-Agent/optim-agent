#!/usr/bin/env python3
"""Run the MNIST medium prompt comparison and print flat metrics JSON."""

import glob
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
STORAGE = ROOT / ".optim-agent-runs" / "mnist"
STATE = ROOT / "autoresearch-results" / "state.json"
SEEDS = ("0", "1", "2")
CONTEXT_LABEL = "GPT-5.5-medium"
NO_CONTEXT_LABEL = "GPT-5.5-medium-no-context"
SEED_REGRESSION_LIMIT = 0.03


def _clean(label):
    safe = label.replace("/", "-")
    for path in ASSETS.glob(f"mnist_curves_{safe}_s*.json"):
        path.unlink()
    for path in STORAGE.glob(f"mnist_{safe}_s*.json"):
        path.unlink()


def _run(method, seed):
    subprocess.run([
        sys.executable, "examples/mnist.py", "run",
        "--method", method,
        "--effort", "medium",
        "--seeds", seed,
        "--trials", "24",
        "--epochs", "3",
        "--workers", "8",
        "--gpus", "0", "1", "2", "3", "4", "5", "6", "7",
        "--timeout", "600",
    ], cwd=ROOT, check=True)


def _vals(label):
    vals = [float(json.loads(Path(p).read_text())["best_error"])
            for p in sorted(glob.glob(str(ASSETS / f"mnist_curves_{label}_s*.json")))]
    return vals


def _mean(label):
    vals = _vals(label)
    return _mean_vals(label, vals), vals


def _mean_vals(label, vals):
    if len(vals) != len(SEEDS):
        raise SystemExit(f"expected {len(SEEDS)} runs for {label}, got {len(vals)}")
    return statistics.mean(vals)


def _score(values):
    random_mean = statistics.mean(values["Random"])
    tpe_mean = statistics.mean(values["TPE"])
    context_vals = values[CONTEXT_LABEL]
    no_context_vals = values.get(NO_CONTEXT_LABEL)
    context_mean = statistics.mean(context_vals)
    no_context_mean = statistics.mean(no_context_vals) if no_context_vals else 0.0
    best_baseline = min(random_mean, tpe_mean)
    margin = best_baseline - context_mean
    context_margin = no_context_mean - context_mean if no_context_vals else -999.0
    gap_score = margin - 0.10 if not no_context_vals else min(margin - 0.10, context_margin - 0.05)
    out = {
        "random_mean": random_mean,
        "tpe_mean": tpe_mean,
        "context_mean": context_mean,
        "no_context_mean": no_context_mean,
        "best_random_tpe": best_baseline,
        "margin_vs_best_random_tpe": margin,
        "context_margin_vs_no_context": context_margin,
        "gap_score": gap_score,
        "success": float(bool(no_context_vals) and margin >= 0.10 and context_margin >= 0.05),
    }
    for i, value in enumerate(context_vals):
        out[f"context_s{i}"] = value
    for i, value in enumerate(no_context_vals or []):
        out[f"no_context_s{i}"] = value
    return out


def _target1_failed(metrics):
    return metrics["margin_vs_best_random_tpe"] < 0.10


def _seed_is_close_enough(current_vals, reference_vals):
    return all(cur <= ref + SEED_REGRESSION_LIMIT for cur, ref in zip(current_vals, reference_vals))


def _reference_context_vals_from_metrics(retained_metrics, previous_metrics):
    vals = []
    for i in range(len(SEEDS)):
        key = f"context_s{i}"
        value = previous_metrics.get(key)
        if value is None:
            value = retained_metrics.get(key)
        if value is None:
            raise SystemExit("retained GPT-5.5-medium seed results are required for pruning")
        vals.append(float(value))
    return vals


def _previous_context_metrics(state, asset_vals):
    if state.get("last_status") in {"keep", "discard", "drift"}:
        return state.get("last_trial_metrics", {})
    return {f"context_s{i}": value for i, value in enumerate(asset_vals)}


def _reference_context_vals():
    if STATE.exists():
        state = json.loads(STATE.read_text()).get("state", {})
        metrics = state.get("current_metrics", {})
        previous = _previous_context_metrics(state, _vals(CONTEXT_LABEL))
        return _reference_context_vals_from_metrics(metrics, previous)
    vals = _vals(CONTEXT_LABEL)
    if len(vals) == len(SEEDS):
        return vals
    raise SystemExit("retained GPT-5.5-medium seed results are required for pruning")


def _emit(metrics, *, pruned=False, reason=""):
    metrics = dict(metrics)
    metrics["pruned"] = float(pruned)
    if reason:
        print(f"pruned: {reason}", file=sys.stderr)
    print(json.dumps(metrics, sort_keys=True))


def main():
    reference_vals = _reference_context_vals()
    _clean(CONTEXT_LABEL)
    _clean(NO_CONTEXT_LABEL)

    random_vals = _vals("Random")
    tpe_vals = _vals("TPE")
    context_vals = []
    for seed in SEEDS:
        _run("codex", seed)
        context_vals = _vals(CONTEXT_LABEL)
        metrics = _score({"Random": random_vals, "TPE": tpe_vals, CONTEXT_LABEL: context_vals})
        if not _seed_is_close_enough(context_vals, reference_vals):
            _emit(metrics, pruned=True, reason=f"context seed {len(context_vals) - 1} regressed >0.03")
            return

    metrics = _score({"Random": random_vals, "TPE": tpe_vals, CONTEXT_LABEL: context_vals})
    if _target1_failed(metrics):
        _emit(metrics, pruned=True, reason="target1 failed")
        return

    no_context_vals = []
    for seed in SEEDS:
        _run("codex-no-context", seed)
        no_context_vals = _vals(NO_CONTEXT_LABEL)
    _emit(_score({
        "Random": random_vals,
        "TPE": tpe_vals,
        CONTEXT_LABEL: context_vals,
        NO_CONTEXT_LABEL: no_context_vals,
    }))


if __name__ == "__main__":
    main()
