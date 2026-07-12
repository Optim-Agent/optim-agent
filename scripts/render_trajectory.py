"""Render the committed Branin optimization trajectories as an animated GIF."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs/assets"
AGENT_CANDIDATES = ("GPT-5.5", "GPT-5.5-medium")
BASELINE_COLOR = "#2563eb"
AGENT_COLOR = "#d94841"
OPTIMUM_COLOR = "#147d64"
TITLE = "Optim-agent (GPT-5.5-medium w/o ctx) v.s. Optuna (TPE) on Branin 2D"
AGENT_LABEL = "optim-agent"
BASELINE_LABEL = "optuna"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(assets=ASSETS, seed=0, agent_label=None):
    assets = Path(assets)
    baseline = _load(assets / f"hard_curves_TPE_s{seed}.json")
    candidates = (agent_label,) if agent_label else AGENT_CANDIDATES
    for label in candidates:
        path = assets / f"hard_curves_{label}_s{seed}.json"
        if path.exists():
            return baseline, _load(path)
    expected = ", ".join(f"hard_curves_{label}_s{seed}.json" for label in candidates)
    raise FileNotFoundError(f"no agent trajectory found; expected one of: {expected}")


def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def _incumbent(values):
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def render(output, baseline, agent_run, dpi=90):
    baseline_data = baseline["functions"]["branin"]
    agent_data = agent_run["functions"]["branin"]
    baseline_points = baseline_data["params"]
    agent_points = agent_data["params"]
    trials = min(len(baseline_points), len(agent_points))
    frames = list(range(1, trials + 1)) + [trials, trials]

    x1 = np.linspace(-5, 10, 180)
    x2 = np.linspace(0, 15, 180)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    landscape = np.log1p(branin(grid_x1, grid_x2))
    baseline_best = _incumbent(baseline_data["values"])
    agent_best = _incumbent(agent_data["values"])

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fbfcfe",
    })
    figure, (space_axis, curve_axis) = plt.subplots(
        1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": (1.08, 0.92)}
    )
    figure.subplots_adjust(top=0.85, bottom=0.14, left=0.07, right=0.97, wspace=0.24)

    def draw(step):
        space_axis.clear()
        curve_axis.clear()
        figure.suptitle(
            TITLE,
            fontsize=19,
            fontweight="bold",
            y=0.95,
        )

        space_axis.contourf(
            grid_x1, grid_x2, landscape, levels=22, cmap="YlGnBu_r", alpha=0.88,
        )
        space_axis.contour(
            grid_x1, grid_x2, landscape, levels=10, colors="#ffffff",
            linewidths=0.45, alpha=0.55,
        )
        for optimum in ((-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)):
            space_axis.scatter(
                *optimum, marker="*", s=120, color=OPTIMUM_COLOR,
                edgecolor="white", linewidth=0.8, zorder=4,
            )

        baseline_xy = np.array([
            (point["x1"], point["x2"]) for point in baseline_points[:step]
        ])
        agent_xy = np.array([
            (point["x1"], point["x2"]) for point in agent_points[:step]
        ])
        space_axis.plot(
            baseline_xy[:, 0], baseline_xy[:, 1], color=BASELINE_COLOR,
            linewidth=1.4, alpha=0.62,
        )
        space_axis.plot(
            agent_xy[:, 0], agent_xy[:, 1], color=AGENT_COLOR,
            linewidth=1.7, alpha=0.72,
        )
        space_axis.scatter(
            baseline_xy[:, 0], baseline_xy[:, 1], label=BASELINE_LABEL, s=42,
            color=BASELINE_COLOR, edgecolor="white", linewidth=0.7, zorder=5,
        )
        space_axis.scatter(
            agent_xy[:, 0], agent_xy[:, 1], label=AGENT_LABEL, s=52,
            color=AGENT_COLOR, edgecolor="white", linewidth=0.8, zorder=6,
        )
        space_axis.scatter(
            baseline_xy[-1, 0], baseline_xy[-1, 1], s=125,
            facecolor="none", edgecolor=BASELINE_COLOR, linewidth=2.0, zorder=7,
        )
        space_axis.scatter(
            agent_xy[-1, 0], agent_xy[-1, 1], s=145,
            facecolor="none", edgecolor=AGENT_COLOR, linewidth=2.2, zorder=7,
        )
        space_axis.set(
            title="Where each method samples",
            xlabel="x1",
            ylabel="x2",
            xlim=(-5, 10),
            ylim=(0, 15),
        )
        space_axis.legend(loc="upper right", frameon=True, framealpha=0.93)

        trial_axis = np.arange(1, trials + 1)
        curve_axis.plot(
            trial_axis[:step], baseline_best[:step], "o-", label=BASELINE_LABEL,
            color=BASELINE_COLOR, linewidth=2.2, markersize=5,
        )
        curve_axis.plot(
            trial_axis[:step], agent_best[:step], "o-", label=AGENT_LABEL,
            color=AGENT_COLOR, linewidth=2.5, markersize=5.5,
        )
        curve_axis.axhline(
            0.397887, color=OPTIMUM_COLOR, linestyle="--", linewidth=1.5,
            label="global optimum",
        )
        curve_axis.set_yscale("log")
        curve_axis.set(
            title="Best objective found so far",
            xlabel="Trial",
            ylabel="Branin objective (lower is better)",
            xlim=(0.7, trials + 0.3),
            ylim=(0.3, max(baseline_best[0], agent_best[0]) * 1.5),
            xticks=trial_axis,
        )
        curve_axis.grid(True, which="both", color="#d8dee8", linewidth=0.7, alpha=0.8)
        curve_axis.legend(loc="upper right", frameon=True, framealpha=0.95)
        curve_axis.text(
            0.04, 0.05,
            f"{BASELINE_LABEL} best: {baseline_best[step - 1]:.3f}\n"
            f"{AGENT_LABEL} best: {agent_best[step - 1]:.3f}",
            transform=curve_axis.transAxes,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.92,
                  "edgecolor": "#d8dee8"},
            fontsize=11,
        )
        figure.text(
            0.5, 0.045,
            "Generated from committed benchmark JSON. Stars mark the three global optima.",
            ha="center", color="#65727f", fontsize=10,
        )

    animation = FuncAnimation(figure, draw, frames=frames, interval=650, repeat=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=1.55), dpi=dpi)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS / "optimization_trajectory.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-label")
    args = parser.parse_args()
    baseline, agent_run = load_runs(ASSETS, args.seed, args.agent_label)
    render(args.output, baseline, agent_run)
    print(args.output)


if __name__ == "__main__":
    main()
