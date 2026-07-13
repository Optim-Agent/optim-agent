"""Tune a transparent quality, latency, and cost model for AI inference.

The evaluator is deterministic and dependency-free so the workflow runs on a
laptop. Replace `evaluate_configuration` with measurements from your own eval
set and serving stack; keep the quality floor explicit.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa


QUALITY_FLOOR = 0.82


def suggest_params(trial):
    return {
        "quantization": trial.suggest_categorical(
            "quantization", ("fp16", "int8", "int4"),
            context="lower precision is cheaper and faster but can reduce eval quality",
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", (1, 2, 4, 8, 16),
            context="larger batches improve throughput but can hurt tail latency",
        ),
        "max_tokens": trial.suggest_categorical(
            "max_tokens", (64, 128, 256, 512),
            context="generation cap; longer outputs cost more and may improve task completion",
        ),
        "speculative_decoding": trial.suggest_categorical(
            "speculative_decoding", (False, True),
            context="draft-model decoding lowers latency with a small extra compute cost",
        ),
    }


def evaluate_configuration(params):
    precision_quality = {"fp16": 0.91, "int8": 0.895, "int4": 0.84}
    precision_speed = {"fp16": 1.0, "int8": 0.72, "int4": 0.55}
    precision_cost = {"fp16": 1.0, "int8": 0.68, "int4": 0.48}
    quantization = params["quantization"]
    batch_size = params["batch_size"]
    max_tokens = params["max_tokens"]
    speculative = params["speculative_decoding"]

    quality = precision_quality[quantization] + 0.008 * math.log2(max_tokens / 64)
    if speculative:
        quality -= 0.002
    latency = (
        (85 + 1.4 * max_tokens)
        * precision_speed[quantization]
        * (1 + 0.045 * (batch_size - 1))
        * (0.72 if speculative else 1.0)
    )
    cost = (
        1.8
        * precision_cost[quantization]
        * (max_tokens / 128)
        / math.sqrt(batch_size)
        * (1.08 if speculative else 1.0)
    )
    return {
        "quality": min(1.0, quality),
        "p95_latency_ms": latency,
        "cost_per_1k_requests_usd": cost,
    }


def utility(metrics):
    if metrics["quality"] < QUALITY_FLOOR:
        return -100.0 - 100 * (QUALITY_FLOOR - metrics["quality"])
    return (
        100 * metrics["quality"]
        - metrics["p95_latency_ms"] / 20
        - 2 * metrics["cost_per_1k_requests_usd"]
    )


def run(trials=12, seed=0, backend="mock", model=None, history=5,
        explicit_reasoning=True, qualitative_notes=True):
    sampler = oa.AgentSampler(
        backend=backend,
        model=model,
        effort="medium",
        n_init=3,
        seed=seed,
        context=(
            "maximize inference utility while preserving quality >= 0.82; utility "
            "values evaluation quality and penalizes p95 latency and cost per 1k requests"
        ),
        history=history,
        explicit_reasoning=explicit_reasoning,
        qualitative_notes=qualitative_notes,
    )
    study = oa.create_study(direction="maximize", sampler=sampler, seed=seed)
    study.optimize(
        lambda trial: utility(evaluate_configuration(suggest_params(trial))),
        n_trials=trials,
    )
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--backend", choices=("mock", "claude", "codex", "opencode"),
        default="mock",
    )
    parser.add_argument("--model")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--explicit-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qualitative-notes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    study = run(
        args.trials,
        backend=args.backend,
        model=args.model,
        history=args.history,
        explicit_reasoning=args.explicit_reasoning,
        qualitative_notes=args.qualitative_notes,
    )
    metrics = evaluate_configuration(study.best_params)
    print(f"best utility: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")
    print(f"quality: {metrics['quality']:.3f}")
    print(f"p95 latency: {metrics['p95_latency_ms']:.1f} ms")
    print(f"cost / 1k requests: ${metrics['cost_per_1k_requests_usd']:.3f}")


if __name__ == "__main__":
    main()
