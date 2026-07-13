"""Five-minute, dependency-free optim-agent quickstart."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa


def objective(trial):
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-1,
        log=True,
        context="step size; values around 0.01 work well for this synthetic task",
    )
    depth = trial.suggest_int(
        "depth",
        2,
        12,
        context="model depth; the synthetic optimum is near 7",
    )
    return (learning_rate - 0.01) ** 2 * 1000 + (depth - 7) ** 2


def run(trials=12, seed=7, backend="mock", history=5,
        explicit_reasoning=True, qualitative_notes=True):
    sampler = oa.AgentSampler(
        backend=backend,
        effort="medium",
        n_init=2,
        seed=seed,
        context="small-budget tuning of learning rate and model depth",
        history=history,
        explicit_reasoning=explicit_reasoning,
        qualitative_notes=qualitative_notes,
    )
    study = oa.create_study(direction="minimize", sampler=sampler, seed=seed)
    study.optimize(objective, n_trials=trials, verbose=False)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--backend",
        choices=("mock", "claude", "codex", "opencode"),
        default="mock",
    )
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--explicit-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qualitative-notes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    study = run(
        trials=args.trials,
        backend=args.backend,
        history=args.history,
        explicit_reasoning=args.explicit_reasoning,
        qualitative_notes=args.qualitative_notes,
    )
    print(f"best value: {study.best_value:.6g}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
