"""Two ablations, both on Branin (2D) and Ackley (5D), reusing the main-benchmark
GLM-5.2 / Random / TPE curves where valid:

  (1) effort  — GLM-5.2 sampler at all five efforts (low..max) vs Random & TPE.
  (2) prune   — GLM-5.2 sampler with each AgentPruner tightness (none/loose/
                medium/tight) vs Random & TPE.

Pruning needs an intermediate learning curve, but Branin/Ackley are scalar. So
each evaluation is dressed with a SYNTHETIC descending curve toward f(x) (noisy,
with occasional slow-starters), and the pruner may stop a trial early. The
metric is best-value vs compute (reported steps): pruning's payoff is spending
fewer steps on doomed trials, not a lower final value.

    python examples/ablations.py effort --variant low   --seeds 0 1 2
    python examples/ablations.py prune  --variant tight  --seeds 0 1
    python examples/ablations.py plot
    python examples/ablations.py selfcheck
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optim_agent as oa
from hard_functions import ASSETS, FUNCTIONS, _context, make_objective

GLM = dict(backend="opencode", model=None)     # GLM-5.2 = opencode default
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
PRUNES = ["none", "loose", "medium", "tight"]
S = 4                                            # synthetic curve length (steps/trial)
N_TRIALS = 10
REF = {"branin": 40.0, "ackley5": 12.0}          # rough value span, sets curve height

# reuse map: which method's main-benchmark curve stands in for an ablation series
REUSE = {"high": "GLM-5.2", "Random": "Random", "TPE": "TPE", "none": "GLM-5.2"}


# -- ablation 1: sampler effort -------------------------------------------

def run_effort(variant, seeds, timeout):
    for seed in seeds:
        out = {"label": variant, "functions": {}}
        for name, spec in FUNCTIONS.items():
            sampler = oa.AgentSampler(backend=GLM["backend"], model=GLM["model"],
                                      effort=variant, context=_context(spec),
                                      n_init=3, timeout=timeout, seed=seed)
            study = oa.create_study(sampler=sampler, seed=seed)
            print(f"== effort {variant} on {name} (seed {seed}) ==")
            study.optimize(make_objective(spec), n_trials=N_TRIALS)
            out["functions"][name] = {"values": [t.value for t in study.trials]}
        (ASSETS / f"abl_effort_{variant}_s{seed}.json").write_text(json.dumps(out, indent=1))
        print(f"wrote abl_effort_{variant}_s{seed}.json")


# -- ablation 2: pruner tightness -----------------------------------------

def _synth(f, name, rng):
    """A noisy loss curve of length S descending toward the final value f."""
    gap = rng.uniform(0.4, 1.2) * REF[name]
    if rng.random() < 0.2:
        gap *= 2.5                                # slow starter: high start, same final
    for s in range(S):
        frac = (s + 1) / S
        v = f + gap * (1 - frac) ** 1.5 * (1 + 0.08 * rng.gauss(0, 1))
        yield s, max(v, f)                        # never dip below the final value


def run_prune(variant, seeds, timeout):
    for seed in seeds:
        out = {"label": variant, "functions": {}}
        for name, spec in FUNCTIONS.items():
            pruner = None if variant == "none" else \
                oa.AgentPruner(backend=GLM["backend"], model=GLM["model"], level=variant, timeout=120)
            sampler = oa.AgentSampler(backend=GLM["backend"], model=GLM["model"],
                                      effort="high", context=_context(spec),
                                      n_init=3, timeout=timeout, seed=seed)
            study = oa.create_study(sampler=sampler, pruner=pruner, seed=seed)
            print(f"== prune {variant} on {name} (seed {seed}) ==")
            cum, curve = 0, []                    # curve: (cumulative_steps, best_so_far) per trial
            for k in range(N_TRIALS):
                trial = study.ask()
                x = [trial.suggest_float(f"x{i + 1}", lo, hi)
                     for i, (lo, hi) in enumerate(spec["bounds"])]
                f = spec["fn"](x)
                rng = random.Random(hash((seed, name, k)) & 0xFFFFFFFF)
                pruned = False
                for s, v in _synth(f, name, rng):
                    trial.report(v, s)
                    cum += 1
                    if trial.should_prune():
                        pruned = True
                        break
                study.tell(trial, state="pruned" if pruned else "complete")
                best = study.best_value
                curve.append((cum, best if best is not None else float("nan")))
            out["functions"][name] = {"curve": curve}
        (ASSETS / f"abl_prune_{variant}_s{seed}.json").write_text(json.dumps(out, indent=1))
        print(f"wrote abl_prune_{variant}_s{seed}.json")


# -- plotting --------------------------------------------------------------

def _load_best_values(label, name, seed):
    """Best-value-per-trial for a reused main-benchmark method, or an effort file."""
    src = REUSE.get(label, label)
    p = ASSETS / (f"hard_curves_{src}_s{seed}.json" if label in REUSE
                  else f"abl_effort_{label}_s{seed}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())["functions"][name]["values"]


def _effort_curves(name, seeds):
    import numpy as np
    out = {}
    for label in EFFORTS + ["Random", "TPE"]:
        per_seed = [np.minimum.accumulate(v) for s in seeds
                    if (v := _load_best_values(label, name, s)) is not None]
        if per_seed:
            w = min(map(len, per_seed))
            out[label] = np.mean([c[:w] for c in per_seed], axis=0)
    return out


def _prune_curves(name, seeds):
    """(mean cumulative steps, mean best) per trial index, per pruner variant + baselines."""
    import numpy as np
    out = {}
    for label in PRUNES + ["Random", "TPE"]:
        xs, ys = [], []                           # per-seed lists of per-trial arrays
        for seed in seeds:
            if label in ("none", "Random", "TPE"):     # reuse: steps are deterministic k*S
                vals = _load_best_values(label, name, seed)
                if vals is None:
                    continue
                best = np.minimum.accumulate(vals)[:N_TRIALS]
                xs.append(np.arange(1, len(best) + 1) * S)
                ys.append(best)
            else:
                p = ASSETS / f"abl_prune_{label}_s{seed}.json"
                if not p.exists():
                    continue
                curve = json.loads(p.read_text())["functions"][name]["curve"]
                xs.append(np.array([c[0] for c in curve]))
                ys.append(np.minimum.accumulate([c[1] for c in curve]))
        if xs:
            w = min(map(len, xs))
            out[label] = (np.mean([a[:w] for a in xs], axis=0),
                          np.mean([a[:w] for a in ys], axis=0))
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    seeds = [0, 1, 2]
    e_colors = dict(zip(EFFORTS, ["#c7d2fe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"]))
    p_colors = dict(zip(PRUNES, ["#9ca3af", "#fbbf24", "#f97316", "#dc2626"]))
    base = {"Random": ("#111827", (0, (2, 2))), "TPE": ("#6b7280", (0, (5, 2)))}

    # --- effort figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axes, FUNCTIONS):
        cur = _effort_curves(name, seeds)
        for label, y in cur.items():
            xs = range(1, len(y) + 1)
            if label in base:
                ax.plot(xs, y, color=base[label][0], linestyle=base[label][1], lw=2.4, label=label)
            else:
                ax.plot(xs, y, color=e_colors[label], marker="o", ms=4, lw=1.7, label=f"effort={label}")
        ax.set_xlabel("trial"), ax.set_ylabel("best value so far (mean of 3 seeds)")
        ax.set_title("Branin 2D" if name == "branin" else "Ackley 5D", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Ablation 1 — GLM-5.2 sampler effort (low→max) vs Random & TPE", fontsize=12)
    fig.tight_layout(); fig.savefig(ASSETS / "abl_effort.png", dpi=130)
    print("wrote abl_effort.png")

    # --- prune figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axes, FUNCTIONS):
        cur = _prune_curves(name, [0, 1])
        for label, (x, y) in cur.items():
            if label in base:
                ax.plot(x, y, color=base[label][0], linestyle=base[label][1], lw=2.4, label=label)
            else:
                ax.plot(x, y, color=p_colors[label], marker="o", ms=4, lw=1.7,
                        label=f"pruner={label}")
        ax.set_xlabel("compute (reported steps)"), ax.set_ylabel("best value so far (mean of 2 seeds)")
        ax.set_title("Branin 2D" if name == "branin" else "Ackley 5D", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Ablation 2 — AgentPruner tightness with GLM-5.2, best value vs compute", fontsize=12)
    fig.tight_layout(); fig.savefig(ASSETS / "abl_prune.png", dpi=130)
    print("wrote abl_prune.png")


def selfcheck():
    rng = random.Random(0)
    steps = list(_synth(2.5, "branin", rng))
    assert len(steps) == S and steps[-1][1] == 2.5, steps          # curve ends exactly at f
    assert all(v >= 2.5 for _, v in steps)                          # never below final
    assert steps[0][1] > 2.5                                        # starts above final
    # reuse wiring resolves to real files
    assert _load_best_values("high", "branin", 0) is not None       # -> hard_curves_GLM-5.2
    assert _load_best_values("Random", "ackley5", 0) is not None
    print("selfcheck ok: synthetic curve ends at f, descends, and reuse map resolves")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for exp, choices in [("effort", EFFORTS), ("prune", PRUNES)]:
        p = sub.add_parser(exp)
        p.add_argument("--variant", required=True, choices=choices)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--timeout", type=float, default=240)
    sub.add_parser("plot")
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    if args.cmd == "effort":
        run_effort(args.variant, args.seeds, args.timeout)
    elif args.cmd == "prune":
        run_prune(args.variant, args.seeds, args.timeout)
    elif args.cmd == "plot":
        plot()
    else:
        selfcheck()
