#!/usr/bin/env python3
"""Measure anchor-free GPT reward ratios on MNIST and CIFAR-10."""

import argparse
import importlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "autoresearch-results" / "classification-n10-s5"
BASELINES = RUN_ROOT / "baselines"
GPT_CURRENT = RUN_ROOT / "gpt-current"
SEEDS = (0, 1, 2, 3, 4)
TRIALS = 10
EPOCHS = 3
WORKERS = 4
EFFORT = "medium"
MODEL = "gpt-5.5"
TIMEOUT = 600
GPU_SPLITS = {"mnist": tuple(range(8)), "cifar10": tuple(range(8))}
LABELS = {"Random": "Random", "TPE": "TPE", "codex": "GPT-5.5-medium"}


def _dataset_module(dataset):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(f"examples.{dataset}")


def _reward_curve(values):
    best = math.inf
    out = []
    for value in values:
        best = min(best, float(value))
        out.append(best)
    return out


def _curve_path(root, dataset, label, seed):
    return root / dataset / f"{dataset}_curves_{label}_s{seed}.json"


def _reward(root, dataset, label):
    rewards = []
    for seed in SEEDS:
        path = _curve_path(root, dataset, label, seed)
        data = json.loads(path.read_text())
        if data["label"] != label or data["seed"] != seed or data["trials"] != TRIALS:
            raise ValueError(f"incompatible curve metadata in {path}")
        records = data["records"]
        if len(records) != TRIALS:
            raise ValueError(f"expected {TRIALS} records in {path}, got {len(records)}")
        values = []
        for record in records:
            if record.get("state", "complete") != "complete" or record.get("test_error") is None:
                raise ValueError(f"incomplete trial in {path}")
            values.append(float(record["test_error"]))
        rewards.append(sum(_reward_curve(values)))
    return statistics.mean(rewards), rewards


def _complete(root, dataset, label):
    try:
        _reward(root, dataset, label)
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _worker_command(dataset, method, root, seed):
    gpus = GPU_SPLITS[dataset]
    offset = seed * (WORKERS - 1) % len(gpus)
    gpus = gpus[offset:] + gpus[:offset]
    return [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--dataset", dataset,
        "--method", method,
        "--seed", str(seed),
        "--assets", str(root / dataset),
        "--storage", str(root / "storage" / dataset / method),
        "--gpus", *map(str, gpus),
    ]


def _run_command(command):
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, file=sys.stderr, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)


def _run_pair(method, root):
    commands = [_worker_command(dataset, method, root, seed)
                for dataset in GPU_SPLITS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = [pool.submit(_run_command, command) for command in commands]
        for future in futures:
            future.result()


def _prepare_baselines():
    ready = all(_complete(BASELINES, dataset, label)
                for dataset in GPU_SPLITS for label in ("Random", "TPE"))
    if ready:
        return
    shutil.rmtree(BASELINES, ignore_errors=True)
    for method in ("Random", "TPE"):
        _run_pair(method, BASELINES)


def _run_gpt():
    shutil.rmtree(GPT_CURRENT, ignore_errors=True)
    _run_pair("codex", GPT_CURRENT)


def _metrics():
    metrics = {}
    ratios = []
    for dataset in GPU_SPLITS:
        random_reward, random_seeds = _reward(BASELINES, dataset, "Random")
        tpe_reward, tpe_seeds = _reward(BASELINES, dataset, "TPE")
        gpt_reward, gpt_seeds = _reward(GPT_CURRENT, dataset, "GPT-5.5-medium")
        baseline = min(random_reward, tpe_reward)
        ratio = gpt_reward / baseline
        ratios.append(ratio)
        prefix = "cifar10" if dataset == "cifar10" else dataset
        metrics.update({
            f"{prefix}_ratio": ratio,
            f"{prefix}_random_reward": random_reward,
            f"{prefix}_tpe_reward": tpe_reward,
            f"{prefix}_gpt_reward": gpt_reward,
        })
        for label, values in (("random", random_seeds), ("tpe", tpe_seeds),
                              ("gpt", gpt_seeds)):
            metrics.update({f"{prefix}_{label}_s{i}": value
                            for i, value in enumerate(values)})
    metrics["max_ratio"] = max(ratios)
    return metrics


def _worker(args):
    module = _dataset_module(args.dataset)
    sampler = module._sampler("codex", args.seed, EFFORT, TIMEOUT, MODEL)
    if sampler.anchor_proposals:
        raise SystemExit(f"{args.dataset} benchmark injects anchor proposals")
    module.ASSETS = Path(args.assets)
    module.STORAGE = Path(args.storage)
    module.run(args.method, [args.seed], TRIALS, EPOCHS, WORKERS,
               args.gpus, EFFORT, TIMEOUT, MODEL)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    worker = sub.add_parser("worker")
    worker.add_argument("--dataset", required=True, choices=tuple(GPU_SPLITS))
    worker.add_argument("--method", required=True, choices=tuple(LABELS))
    worker.add_argument("--seed", type=int, required=True, choices=SEEDS)
    worker.add_argument("--assets", required=True)
    worker.add_argument("--storage", required=True)
    worker.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        _worker(args)
        return
    _prepare_baselines()
    _run_gpt()
    print(json.dumps(_metrics(), sort_keys=True))


if __name__ == "__main__":
    main()
