"""Leakage-aware walk-forward tuning example for a toy trading strategy.

This educational example uses synthetic returns and is not investment advice.
Replace the data and objective only after defining a genuinely held-out final
test period.
"""

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa


def synthetic_returns(length=480, seed=0):
    rng = random.Random(seed)
    returns = []
    for day in range(length):
        regime = 0.0008 if (day // 80) % 2 == 0 else -0.0004
        cycle = 0.0007 * math.sin(day / 13)
        returns.append(regime + cycle + rng.gauss(0, 0.009))
    return returns


def walk_forward_slices(length, train_size=160, test_size=40):
    folds = []
    test_start = train_size
    while test_start + test_size <= length:
        folds.append((
            test_start - train_size,
            test_start,
            test_start,
            test_start + test_size,
        ))
        test_start += test_size
    return folds


def suggest_params(trial):
    return {
        "lookback": trial.suggest_int(
            "lookback", 5, 60,
            context="past-only momentum window in trading days",
        ),
        "entry_threshold": trial.suggest_float(
            "entry_threshold", 0.0002, 0.006, log=True,
            context="minimum average past return required before taking a position",
        ),
        "rebalance_every": trial.suggest_int(
            "rebalance_every", 1, 10,
            context="days between position changes; slower trading reduces turnover",
        ),
    }


def _fold_returns(returns, fold, params, cost=0.0005):
    train_start, _, test_start, test_end = fold
    lookback = params["lookback"]
    position = 0
    net = []
    for index in range(test_start, test_end):
        previous = position
        if (index - test_start) % params["rebalance_every"] == 0:
            history_start = max(train_start, index - lookback)
            history = returns[history_start:index]
            momentum = statistics.fmean(history) if history else 0.0
            threshold = params["entry_threshold"]
            position = 1 if momentum > threshold else -1 if momentum < -threshold else 0
        turnover = abs(position - previous)
        net.append(position * returns[index] - turnover * cost)
    return net


def _risk_adjusted_score(returns):
    if len(returns) < 2:
        return -100.0
    volatility = statistics.stdev(returns)
    sharpe = 0.0 if volatility == 0 else statistics.fmean(returns) / volatility * math.sqrt(252)
    equity = peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1 + value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return sharpe - 0.5 * max_drawdown


def walk_forward_score(returns, params, train_size=120, test_size=40):
    scores = [
        _risk_adjusted_score(_fold_returns(returns, fold, params))
        for fold in walk_forward_slices(len(returns), train_size, test_size)
    ]
    if not scores:
        raise ValueError("not enough observations for one walk-forward fold")
    return float(statistics.fmean(scores))


def run(trials=12, seed=0, backend="mock", model=None):
    returns = synthetic_returns(seed=seed)
    sampler = oa.AgentSampler(
        backend=backend,
        model=model,
        effort="medium",
        n_init=3,
        seed=seed,
        context=(
            "maximize walk-forward risk-adjusted return for a momentum strategy; "
            "signals use past data only and every position change pays transaction costs"
        ),
    )
    study = oa.create_study(direction="maximize", sampler=sampler, seed=seed)
    study.optimize(
        lambda trial: walk_forward_score(returns, suggest_params(trial)),
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
    args = parser.parse_args()
    study = run(args.trials, backend=args.backend, model=args.model)
    print(f"best walk-forward score: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
