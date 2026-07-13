"""Tune a scikit-learn random forest on a small, built-in dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 50, 300,
            context="more trees reduce variance but increase training and inference cost",
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 2, 20,
            context="tree depth controls capacity and overfitting",
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 20,
            context="larger values regularize small leaves",
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ("sqrt", "log2", None),
            context="features considered at each split; lower values decorrelate trees",
        ),
    }


def make_objective(seed=0):
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error

    features, labels = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=seed,
    )

    def objective(trial):
        model = RandomForestClassifier(
            **suggest_params(trial), random_state=seed, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    return objective


def run(trials=10, seed=0, backend="mock", model=None, history=5,
        explicit_reasoning=True, qualitative_notes=True):
    sampler = oa.AgentSampler(
        backend=backend,
        model=model,
        effort="medium",
        n_init=2,
        seed=seed,
        context=(
            "maximize validation accuracy for a random forest on the breast-cancer "
            "dataset while keeping configurations practical on a laptop"
        ),
        history=history,
        explicit_reasoning=explicit_reasoning,
        qualitative_notes=qualitative_notes,
    )
    study = oa.create_study(direction="maximize", sampler=sampler, seed=seed)
    study.optimize(make_objective(seed), n_trials=trials)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
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
    print(f"best accuracy: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
