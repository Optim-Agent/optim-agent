"""Toy benchmark: agents vs random search on classic test functions.

Run one agent (writes docs/assets/curves_<label>.json, both functions):
    python examples/classic_functions.py run --agent GPT-5.5 --trials 12
Plot everything found in docs/assets/ into the README figures:
    python examples/classic_functions.py plot
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optim_agent as oa

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

# Labels are what the plots show; edit backend/model to match your CLI setup.
# model=None uses the CLI's configured default model.
PRESETS = {
    "GPT-5.5": dict(backend="codex", model="gpt-5.5"),
    "Fable-5": dict(backend="claude", model="claude-fable-5"),
    "Opus-4.8": dict(backend="claude", model="claude-opus-4-8"),
    "GLM-5.2": dict(backend="opencode", model=None),
    "Random": dict(backend=None, model=None),
    "Mock": dict(backend="mock", model=None),
}

COLORS = {"GPT-5.5": "#10a37f", "Fable-5": "#d97757", "Opus-4.8": "#8b5cf6",
          "GLM-5.2": "#2563eb", "Random": "#9ca3af", "Mock": "#374151"}
# distinct dashes so exactly-overlapping convergence curves stay visible
STYLES = {"GPT-5.5": (0, (6, 2)), "Fable-5": "solid", "Opus-4.8": (0, (1, 1.5)),
          "GLM-5.2": (0, (4, 2, 1, 2)), "Random": "solid", "Mock": "solid"}


def gramacy_lee(x):
    return math.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


FUNCTIONS = {
    "gramacy_lee": dict(
        objective=lambda t: gramacy_lee(t.suggest_float("x", 0.5, 2.5)),
        context="x of the univariate Gramacy & Lee test function on [0.5, 2.5]: "
                "a wiggly multimodal curve; global minimum near x=0.55."),
    "himmelblau": dict(
        objective=lambda t: himmelblau(t.suggest_float("x", -5, 5),
                                       t.suggest_float("y", -5, 5)),
        context="(x, y) of the bivariate Himmelblau function on [-5, 5]^2; "
                "four global minima with value 0."),
}


def run(label, trials, level, seed):
    preset = PRESETS[label]
    out = {"label": label, "level": level, "functions": {}}
    for name, spec in FUNCTIONS.items():
        if preset["backend"] is None:
            sampler = oa.RandomSampler()
        else:
            sampler = oa.AgentSampler(backend=preset["backend"], model=preset["model"],
                                      level=level, context=spec["context"],
                                      n_init=2, timeout=240, seed=seed)
        study = oa.create_study(sampler=sampler, seed=seed)
        print(f"== {label} on {name} ==")
        study.optimize(spec["objective"], n_trials=trials)
        out["functions"][name] = {
            "values": [t.value for t in study.trials],
            "params": [t.params for t in study.trials],
        }
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"curves_{label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [json.loads(p.read_text()) for p in sorted(ASSETS.glob("curves_*.json"))]
    if not runs:
        sys.exit("no curves_*.json in docs/assets — run some agents first")

    for name, fname in [("gramacy_lee", "univariate.png"), ("himmelblau", "bivariate.png")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        if name == "gramacy_lee":
            xs = np.linspace(0.5, 2.5, 400)
            ax1.plot(xs, [gramacy_lee(x) for x in xs], "k-", lw=1, alpha=0.6)
            ax1.set_xlabel("x"), ax1.set_ylabel("f(x)")
            ax1.set_title("Gramacy & Lee — sampled points")
        else:
            g = np.linspace(-5, 5, 200)
            X, Y = np.meshgrid(g, g)
            Z = (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2
            ax1.contourf(X, Y, Z, levels=np.logspace(0, 3, 20), cmap="Greys", alpha=0.7)
            ax1.set_xlabel("x"), ax1.set_ylabel("y")
            ax1.set_title("Himmelblau — sampled points")
        for r in runs:
            fn = r["functions"].get(name)
            if not fn:
                continue
            c = COLORS.get(r["label"], None)
            if name == "gramacy_lee":
                px = [p["x"] for p in fn["params"]]
                ax1.scatter(px, fn["values"], s=22, color=c, label=r["label"], zorder=3)
            else:
                ax1.scatter([p["x"] for p in fn["params"]], [p["y"] for p in fn["params"]],
                            s=22, color=c, label=r["label"], zorder=3)
            best = np.minimum.accumulate(fn["values"])
            ax2.plot(range(1, len(best) + 1), best, marker="o", ms=3.5, color=c,
                     linestyle=STYLES.get(r["label"], "solid"), lw=1.8, alpha=0.9,
                     label=r["label"])
        ax2.set_xlabel("trial"), ax2.set_ylabel("best value so far")
        ax2.set_title("Convergence")
        ax2.legend(fontsize=8)
        ax1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS / fname, dpi=130)
        print(f"wrote {ASSETS / fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--agent", required=True, choices=list(PRESETS))
    p_run.add_argument("--trials", type=int, default=12)
    p_run.add_argument("--level", default="high", choices=list(oa.samplers.LEVELS))
    p_run.add_argument("--seed", type=int, default=0)
    sub.add_parser("plot")
    args = ap.parse_args()
    run(args.agent, args.trials, args.level, args.seed) if args.cmd == "run" else plot()
