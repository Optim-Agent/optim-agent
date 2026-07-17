"""No-context Branin (2D) and Ackley (5D) agent benchmark.

The agents are given only the input bounds and the trial history — never the
function name or repository source. Standard functions can still be inferred
from their bounds and observations. Two baselines: uniform Random (weak) and
Optuna's TPE (strong, classical Bayesian). With a tight budget of 10 trials
this is a sample-efficiency race, which is where agent reasoning is meant to
pay off.

Pools:
  - tier: GPT-5.5, Opus-4.8, Sonnet-5, Kimi-K3, Minimax-M3, and GLM-5.2
  - free: rotating free OpenCode models for users without paid model APIs

Every agent uses medium effort and receives no task context.

    python examples/hard_functions.py distributed \
      --agents Random TPE GPT-5.5 Opus-4.8 Sonnet-5 GLM-5.2 Big-pickle \
      DeepSeek-V4-Flash Nemotron-3-Ultra MiMo-v2.5 \
      --trials 10 --seeds 0 1 2 3 4
    cp ~/.claude/settings-kimi.json ~/.claude/settings.json
    python examples/hard_functions.py distributed --agents Kimi-K3 --trials 10 --seeds 0 1 2 3 4
    cp ~/.claude/settings-minimax.json ~/.claude/settings.json
    python examples/hard_functions.py distributed --agents Minimax-M3 --trials 10 --seeds 0 1 2 3 4
    python examples/hard_functions.py plot
    python examples/hard_functions.py selfcheck   # verify the function values
"""

import argparse
import json
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optim_agent as oa
from optim_agent import agent as agent_api

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
AGENT_EFFORT = "medium"

# label -> backend/model + plotting group/color/style.
# backend: real CLI name, None (uniform Random), or "tpe" (Optuna TPE baseline).
POOL = {
    "Random": dict(backend=None, model=None, group="both", color="#9ca3af", style="solid"),
    "TPE": dict(backend="tpe", model=None, group="both", color="#111827", style=(0, (2, 2))),
    "GPT-5.5": dict(backend="codex", model="gpt-5.5", group="tier",
                    color="#10a37f", style=(0, (6, 2))),
    "Opus-4.8": dict(backend="claude", model="claude-opus-4-8", group="tier",
                     color="#8b5cf6", style="solid"),
    "Sonnet-5": dict(backend="claude", model="claude-sonnet-5", group="tier",
                     color="#d97706", style=(0, (3, 1))),
    "Kimi-K3": dict(backend="claude", model="kimi-k3", group="tier",
                    color="#dc2626", style=(0, (5, 1, 1, 1))),
    "Minimax-M3": dict(backend="claude", model="MiniMax-M3", group="tier",
                       color="#7c3aed", style=(0, (1, 1))),
    "GLM-5.2": dict(backend="opencode", model="glm-5.2", group="tier",
                    color="#2563eb", style=(0, (4, 2, 1, 2))),
    "Big-pickle": dict(backend="opencode", model="opencode/big-pickle", group="free",
                       color="#16a34a", style="solid"),
    "DeepSeek-V4-Flash": dict(backend="opencode",
                              model="opencode/deepseek-v4-flash-free", group="free",
                              color="#0891b2", style=(0, (6, 2))),
    "Nemotron-3-Ultra": dict(backend="opencode",
                             model="opencode/nemotron-3-ultra-free", group="free",
                             color="#db2777", style=(0, (1, 1.5))),
    "MiMo-v2.5": dict(backend="opencode", model="opencode/mimo-v2.5-free", group="free",
                      color="#ca8a04", style=(0, (4, 2, 1, 2))),
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


def _is_agent(preset):
    return preset["backend"] not in (None, "tpe")


def _make_sampler(preset, seed, timeout, agent_cwd=None, history=5,
                  explicit_reasoning=True, qualitative_notes=True):
    if preset["backend"] is None:
        return oa.RandomSampler()
    if preset["backend"] == "tpe":
        raise ValueError("TPE uses Optuna directly")
    return oa.AgentSampler(
        backend=preset["backend"], model=preset["model"], effort=AGENT_EFFORT,
        context=None, n_init=3, timeout=timeout, seed=seed,
        agent_cwd=agent_cwd,
        history=history,
        explicit_reasoning=explicit_reasoning,
        qualitative_notes=qualitative_notes,
    )


def _tpe_curve(spec, trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3))
    study.optimize(make_objective(spec), n_trials=trials)
    return [t.value for t in study.trials], [t.params for t in study.trials]


def _agent_curve(preset, spec, trials, seed, timeout, history,
                 explicit_reasoning, qualitative_notes):
    with tempfile.TemporaryDirectory(prefix="optim-agent-hard-") as agent_cwd:
        sampler = _make_sampler(
            preset, seed, timeout, agent_cwd=agent_cwd,
            history=history,
            explicit_reasoning=explicit_reasoning,
            qualitative_notes=qualitative_notes,
        )
        study = oa.create_study(sampler=sampler, seed=seed)
        study.optimize(make_objective(spec), n_trials=trials)
        return [t.value for t in study.trials], [t.params for t in study.trials]


def run(label, trials, seed, timeout, history=5,
        explicit_reasoning=True, qualitative_notes=True):
    preset = POOL[label]
    out = {
        "label": label,
        "backend": preset["backend"],
        "model": preset["model"],
        "effort": AGENT_EFFORT if _is_agent(preset) else None,
        "no_context": True if _is_agent(preset) else None,
        "seed": seed,
        "trials": trials,
        "functions": {},
    }
    for name, spec in FUNCTIONS.items():
        print(f"== {label} on {name} (seed {seed}) ==")
        if preset["backend"] == "tpe":
            values, params = _tpe_curve(spec, trials, seed)
        else:
            values, params = _agent_curve(
                preset, spec, trials, seed, timeout,
                history, explicit_reasoning, qualitative_notes,
            )
        print(f"   best={min(values):.4g}")
        out["functions"][name] = {"values": values, "params": params}
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"hard_curves_{label}_s{seed}.json"  # one file per (method, seed)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


def _seed_workers(label, seeds):
    # Concurrent OpenCode processes share a local database and can fail with
    # "database is locked". Other backends retain five-way seed parallelism.
    return 1 if POOL[label]["backend"] == "opencode" else len(seeds)


def run_distributed(labels, trials, seeds, timeout, history=5,
                    explicit_reasoning=True, qualitative_notes=True):
    for label in labels:
        with ThreadPoolExecutor(max_workers=_seed_workers(label, seeds)) as pool:
            if (history, explicit_reasoning, qualitative_notes) == (5, True, True):
                futures = [pool.submit(run, label, trials, seed, timeout) for seed in seeds]
            else:
                futures = [
                    pool.submit(
                        run, label, trials, seed, timeout,
                        history, explicit_reasoning, qualitative_notes,
                    )
                    for seed in seeds
                ]
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

    labels = [label for label, preset in POOL.items()
              if label in by_label and preset["group"] in (group, "both")]
    if not any(POOL[label]["group"] == group for label in labels):
        return
    nseed = max(len(by_label[label]) for label in labels)
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
    heading = ("Top-tier agents" if group == "tier" else "Free OpenCode agents")
    fig.suptitle(f"{heading}, medium effort, no context vs. TPE & Random — "
                 f"lower is better, mean of {nseed} seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / fname, dpi=130)
    print(f"wrote {ASSETS / fname}")


def _validate_run(label, run):
    preset = POOL[label]
    expected_effort = AGENT_EFFORT if _is_agent(preset) else None
    expected_no_context = True if _is_agent(preset) else None
    valid = (
        run.get("label") == label
        and run.get("backend") == preset["backend"]
        and run.get("model") == preset["model"]
        and run.get("effort") == expected_effort
        and run.get("no_context") is expected_no_context
        and run.get("seed") in range(5)
        and run.get("trials") == 10
        and set(run.get("functions", ())) == set(FUNCTIONS)
    )
    if valid:
        valid = all(
            len(result.get("values", ())) == 10
            and len(result.get("params", ())) == 10
            for result in run["functions"].values()
        )
    if not valid:
        raise SystemExit(f"incompatible hard-function plot data for {label}")


def _load_plot_runs():
    by_label = {}
    for path in sorted(ASSETS.glob("hard_curves_*_s*.json")):
        r = json.loads(path.read_text())
        if r.get("label") not in POOL:
            continue
        by_label.setdefault(r["label"], []).append(r)
    if set(by_label) != set(POOL):
        raise SystemExit(f"hard-function plot requires {tuple(POOL)}")
    for label, runs in by_label.items():
        if len(runs) != 5 or {run.get("seed") for run in runs} != set(range(5)):
            raise SystemExit(f"hard-function plot requires seeds 0..4 for {label}")
        for run in runs:
            _validate_run(label, run)
    return by_label


def plot():
    import matplotlib
    matplotlib.use("Agg")
    by_label = _load_plot_runs()
    _plot_group("tier", by_label, "hard_benchmarks_tier.png")
    _plot_group("free", by_label, "hard_benchmarks_free.png")


def preflight(timeout):
    prompt = 'Reply with ONLY this JSON object: {"ok": true}'
    for label, preset in POOL.items():
        if not _is_agent(preset):
            continue
        with tempfile.TemporaryDirectory(prefix="optim-agent-preflight-") as agent_cwd:
            reply = agent_api.call_agent(
                preset["backend"], preset["model"], prompt, timeout,
                effort=AGENT_EFFORT, cwd=agent_cwd,
            )
        data = agent_api.extract_json(reply)
        if not data or data.get("ok") is not True:
            raise RuntimeError(f"preflight failed for {label}: invalid JSON reply")
        print(f"preflight ok: {label} ({preset['backend']} {preset['model']})")


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
    p_run.add_argument("--history", type=int, default=5)
    p_run.add_argument("--explicit-reasoning", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--qualitative-notes", action=argparse.BooleanOptionalAction, default=True)
    p_dist = sub.add_parser("distributed")
    p_dist.add_argument("--agents", nargs="+", choices=list(POOL), default=list(POOL))
    p_dist.add_argument("--trials", type=int, default=10)
    p_dist.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p_dist.add_argument("--timeout", type=float, default=600)
    p_dist.add_argument("--history", type=int, default=5)
    p_dist.add_argument("--explicit-reasoning", action=argparse.BooleanOptionalAction, default=True)
    p_dist.add_argument("--qualitative-notes", action=argparse.BooleanOptionalAction, default=True)
    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--timeout", type=float, default=600)
    sub.add_parser("plot")
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    {"run": lambda: run(args.agent, args.trials, args.seed, args.timeout,
                        args.history, args.explicit_reasoning, args.qualitative_notes),
     "distributed": lambda: run_distributed(
         args.agents, args.trials, args.seeds, args.timeout,
         args.history, args.explicit_reasoning, args.qualitative_notes,
     ),
     "preflight": lambda: preflight(args.timeout),
     "plot": plot, "selfcheck": selfcheck}[args.cmd]()
