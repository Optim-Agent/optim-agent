"""Hard benchmark: standard Branin (2D) and Ackley (5D) with a strong TPE baseline.

The agents are given only the input bounds and the trial history — never the
function name — so they cannot recall a known optimum and must genuinely
optimize. Two baselines: uniform Random (weak) and Optuna's TPE (strong,
classical Bayesian). With a tight budget of 10 trials this is a sample-
efficiency race, which is where agent reasoning is meant to pay off.

Candidates: Random, TPE, GPT-5.5 medium, and GPT-5.5 medium without context.

    python examples/hard_functions.py distributed --trials 10 --seeds 0 1 2 3 4
    python examples/hard_functions.py run --agent TPE --trials 10
    python examples/hard_functions.py plot
    python examples/hard_functions.py selfcheck   # verify the function values
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optim_agent as oa

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

# label -> backend/model + plotting group/color/style.
# backend: real CLI name, None (uniform Random), or "tpe" (Optuna TPE baseline).
POOL = {
    "Random": dict(backend=None, model=None, group="both", color="#9ca3af", style="solid"),
    "TPE": dict(backend="tpe", model=None, group="both", color="#111827", style=(0, (2, 2))),
    "GPT-5.5-medium": dict(backend="codex", model="gpt-5.5", group="tier",
                           color="#10a37f", style=(0, (4, 2))),
    "GPT-5.5-medium-no-context": dict(backend="codex", model="gpt-5.5",
                                      no_context=True, group="tier",
                                      color="#10a37f", style=(0, (1, 1.5))),
}


def branin(x):  # bayeso-benchmarks Branin, global min 0.397887 at 3 points
    x1, x2 = x
    a, b, c, r, s, t = 1, 5.1 / (4 * math.pi**2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * math.cos(x1) + s


def ackley(x):  # bayeso-benchmarks Ackley, global min 0 at the origin
    d = len(x)
    return (-20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in x) / d))
            - math.exp(sum(math.cos(2 * math.pi * v) for v in x) / d) + 20 + math.e)


FUNCTIONS = {
    "branin":  dict(fn=branin, bounds=[(-5.0, 10.0), (0.0, 15.0)], opt=0.397887),
    "ackley5": dict(fn=ackley, bounds=[(-32.768, 32.768)] * 5,      opt=0.0),
}
TITLES = {"branin": "Branin 2D  (optimum 0.3979)", "ackley5": "Ackley 5D  (optimum 0)"}


def make_objective(spec):
    """One closure that runs under either an optim-agent trial or an Optuna trial —
    both expose suggest_float(name, low, high)."""
    def objective(trial):
        x = [trial.suggest_float(f"x{i + 1}", lo, hi) for i, (lo, hi) in enumerate(spec["bounds"])]
        return spec["fn"](x)
    return objective


def _context(spec):
    bounds = "; ".join(f"x{i + 1} in [{lo:g}, {hi:g}]" for i, (lo, hi) in enumerate(spec["bounds"]))
    return (f"a black-box objective with {len(spec['bounds'])} continuous inputs ({bounds}). "
            "Its functional form, optimum location and optimum value are all unknown; "
            "only the bounds and the trial history are available.")


def _tpe_curve(spec, trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3))
    study.optimize(make_objective(spec), n_trials=trials)
    return [t.value for t in study.trials], [t.params for t in study.trials]


def _agent_curve(preset, spec, trials, seed, timeout):
    if preset["backend"] is None:
        sampler = oa.RandomSampler()
    else:
        sampler = oa.AgentSampler(
            backend=preset["backend"], model=preset["model"], effort="medium",
            context=None if preset.get("no_context") else _context(spec),
            n_init=3, timeout=timeout, seed=seed,
        )
    study = oa.create_study(sampler=sampler, seed=seed)
    study.optimize(make_objective(spec), n_trials=trials)
    return [t.value for t in study.trials], [t.params for t in study.trials]


def run(label, trials, seed, timeout):
    preset = POOL[label]
    out = {
        "label": label,
        "seed": seed,
        "trials": trials,
        "model": preset["model"],
        "effort": "medium" if preset["backend"] == "codex" else None,
        "functions": {},
    }
    for name, spec in FUNCTIONS.items():
        print(f"== {label} on {name} (seed {seed}) ==")
        if preset["backend"] == "tpe":
            values, params = _tpe_curve(spec, trials, seed)
        else:
            values, params = _agent_curve(preset, spec, trials, seed, timeout)
        print(f"   best={min(values):.4g}")
        out["functions"][name] = {"values": values, "params": params}
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"hard_curves_{label}_s{seed}.json"  # one file per (method, seed)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


def run_distributed(labels, trials, seeds, timeout):
    for label in labels:
        with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            futures = [pool.submit(run, label, trials, seed, timeout) for seed in seeds]
            for future in futures:
                future.result()


def _mean_best_curve(seed_runs, name):
    """Mean over seeds of the best-so-far curve for one function; None if absent."""
    import numpy as np
    curves = [np.minimum.accumulate(r["functions"][name]["values"])
              for r in seed_runs if name in r["functions"]]
    if not curves:
        return None, 0
    width = min(len(c) for c in curves)
    return np.mean([c[:width] for c in curves], axis=0), len(curves)


def _plot_group(group, by_label, fname):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    labels = [l for l in by_label if POOL.get(l, {}).get("group") in (group, "both")]
    if not any(POOL[l]["group"] == group for l in labels):
        return
    nseed = max(len(by_label[l]) for l in labels)
    fig, axes = plt.subplots(1, len(FUNCTIONS), figsize=(11, 4.4))
    for ax, name in zip(axes, FUNCTIONS):
        for label in labels:
            mean, k = _mean_best_curve(by_label[label], name)
            if mean is None:
                continue
            p = POOL[label]
            base = dict(color=p["color"], linestyle=p["style"], alpha=0.9)
            if label in ("TPE", "Random"):  # baselines: thick, no marker
                ax.plot(range(1, len(mean) + 1), mean, lw=2.6, **base, label=label)
            else:
                ax.plot(range(1, len(mean) + 1), mean, marker="o", ms=4, lw=1.6, **base, label=label)
        ax.set_xlabel("trial"), ax.set_ylabel("best value so far (mean over seeds)")
        ax.set_title(TITLES[name], fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # trials are integers
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"GPT-5.5 medium context ablation vs. TPE & Random — lower is better, "
                 f"mean of {nseed} seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / fname, dpi=130)
    print(f"wrote {ASSETS / fname}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    by_label = {}
    for path in sorted(ASSETS.glob("hard_curves_*_s*.json")):
        r = json.loads(path.read_text())
        by_label.setdefault(r["label"], []).append(r)
    if not by_label:
        sys.exit("no hard_curves_*_s*.json in docs/assets — run some agents first")
    _plot_group("tier", by_label, "hard_benchmarks_tier.png")


def selfcheck():
    assert abs(branin([-math.pi, 12.275]) - 0.397887) < 1e-3, branin([-math.pi, 12.275])
    assert abs(branin([math.pi, 2.275]) - 0.397887) < 1e-3
    assert abs(ackley([0.0] * 5)) < 1e-9
    assert ackley([10.0] * 5) > 15  # far from origin is bad — a real gradient to climb
    print("selfcheck ok: Branin min ~0.3979 at known points, Ackley(0)=0, Ackley far-field large")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--agent", required=True, choices=list(POOL))
    p_run.add_argument("--trials", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--timeout", type=float, default=600,
                       help="seconds per agent call")
    p_dist = sub.add_parser("distributed")
    p_dist.add_argument("--agents", nargs="+", choices=list(POOL), default=list(POOL))
    p_dist.add_argument("--trials", type=int, default=10)
    p_dist.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p_dist.add_argument("--timeout", type=float, default=600)
    sub.add_parser("plot")
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    {"run": lambda: run(args.agent, args.trials, args.seed, args.timeout),
     "distributed": lambda: run_distributed(args.agents, args.trials, args.seeds, args.timeout),
     "plot": plot, "selfcheck": selfcheck}[args.cmd]()
