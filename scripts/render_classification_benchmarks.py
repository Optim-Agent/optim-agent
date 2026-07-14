#!/usr/bin/env python3
"""Render MNIST and CIFAR-10 benchmark curves as one two-panel figure."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import cifar10, mnist  # noqa: E402


ASSETS = ROOT / "docs/assets"
OUTPUT = ASSETS / "classification_benchmarks.png"
SOCIAL_OUTPUT = ASSETS / "social-preview.png"
DATASETS = ("MNIST", "CIFAR-10")
SPECS = (
    ("MNIST", mnist),
    ("CIFAR-10", cifar10),
)
DISPLAY_LABELS = {
    mnist.PLOT_AGENT_LABEL: "GPT-5.5 w/ context",
    mnist.PLOT_NO_CONTEXT_LABEL: "GPT-5.5 w/o context",
}
SOCIAL_LABELS = ("Random", "TPE", mnist.PLOT_AGENT_LABEL)
SOCIAL_COLORS = {"Random": "#9ca3af", "TPE": "#58a6ff", mnist.PLOT_AGENT_LABEL: "#00a67e"}


def _style(module, label):
    method = next(
        (
            name
            for name, preset in module.METHODS.items()
            if label == preset.get("label")
            or label.startswith(preset.get("label", name) + "-")
        ),
        label.split("-")[0],
    )
    return {**module.METHODS.get(method, {}), **module.PLOT_STYLES.get(label, {})}


def render(output=OUTPUT):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    seed_counts = []
    for ax, (title, module) in zip(axes, SPECS):
        by_label = module._load_plot_runs()
        for label in module.PLOT_LABELS:
            runs = by_label[label]
            curves = [module._best_error_curve(run["records"]) for run in runs]
            width = min(len(curve) for curve in curves)
            mean = np.mean([curve[:width] for curve in curves], axis=0)
            style = _style(module, label)
            ax.plot(
                range(1, width + 1),
                mean,
                marker="o",
                ms=4,
                lw=1.8,
                color=style.get("color"),
                linestyle=style.get("style", "solid"),
                label=DISPLAY_LABELS.get(label, label),
            )
            seed_counts.append(len(runs))
        ax.set_xlabel("trial")
        ax.set_ylabel("best test error % (mean over seeds)")
        ax.set_title(f"{title} ResNet (16 dimensions)", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    nseed = min(seed_counts)
    fig.suptitle(
        f"Classification hyperparameter optimization - lower is better, mean of {nseed} seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"wrote {output}")
    return output


def render_social_preview(output=SOCIAL_OUTPUT):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MplPath

    background = "#080c14"
    panel = "#101722"
    border = "#263244"
    foreground = "#f0f6fc"
    muted = "#8b949e"
    accent = "#58a6ff"
    gold = "#d29922"

    plt.rcParams.update({"font.family": "Arial", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=background)
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_axis_off()

    for x in np.linspace(0.035, 0.965, 18):
        canvas.plot((x, x), (0.06, 0.94), color="#111a27", lw=0.5, zorder=0)
    for y in np.linspace(0.06, 0.94, 9):
        canvas.plot((0.035, 0.965), (y, y), color="#111a27", lw=0.5, zorder=0)

    logo_path = MplPath(
        [(0.057, 0.892), (0.065, 0.93), (0.071, 0.84), (0.079, 0.86),
         (0.088, 0.88), (0.094, 0.94), (0.103, 0.90),
         (0.112, 0.87), (0.114, 0.79), (0.126, 0.79), (0.136, 0.82)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
         MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
         MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO],
    )
    canvas.plot((0.054, 0.138), (0.79, 0.79), color="#30363d", lw=1.5)
    canvas.add_patch(PathPatch(logo_path, fill=False, color=accent, lw=3.2, capstyle="round"))
    canvas.add_patch(Circle((0.124, 0.79), 0.007, color=gold, zorder=3))
    fig.text(0.158, 0.842, "optim-agent", color=foreground, fontsize=25, weight="bold")

    fig.text(0.055, 0.565, "REASON. PROPOSE.\nOPTIMIZE.", color=foreground, fontsize=39,
             weight="bold", linespacing=0.92)
    fig.text(0.058, 0.49, "Coding agents propose. Your objective decides.",
             color="#c9d1d9", fontsize=16)
    fig.text(0.058, 0.446, "Any parameters · Any measurable system", color=muted, fontsize=13)

    canvas.add_patch(FancyBboxPatch(
        (0.055, 0.265), 0.34, 0.09, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#0d141f", edgecolor=border, linewidth=1.2,
    ))
    fig.text(0.078, 0.296, "$", color=gold, fontsize=15, family="monospace", weight="bold")
    fig.text(0.099, 0.296, "pip install optim-agent", color=foreground, fontsize=14,
             family="monospace")
    fig.text(0.058, 0.12, "CLAUDE CODE   ·   CODEX   ·   OPENCODE", color="#6e7681",
             fontsize=10, weight="bold")

    fig.text(0.535, 0.895, "BENCHMARK SIGNAL", color=accent, fontsize=10, weight="bold")
    fig.text(0.535, 0.852, "Best test error across 10 trials", color=foreground,
             fontsize=16, weight="bold")
    fig.text(0.535, 0.817, "Lower is better · mean of 5 seeds", color=muted, fontsize=11)

    axes = [fig.add_axes((0.535, 0.49, 0.405, 0.26)), fig.add_axes((0.535, 0.135, 0.405, 0.26))]
    for ax, (title, module) in zip(axes, SPECS):
        by_label = module._load_plot_runs()
        for label in SOCIAL_LABELS:
            runs = by_label[label]
            curves = [module._best_error_curve(run["records"]) for run in runs]
            width = min(len(curve) for curve in curves)
            mean = np.mean([curve[:width] for curve in curves], axis=0)
            style = _style(module, label)
            display_label = "GPT-5.5" if label == module.PLOT_AGENT_LABEL else label
            ax.plot(
                range(1, width + 1),
                mean,
                marker="o",
                ms=4.5,
                lw=2.4,
                color=SOCIAL_COLORS[label],
                linestyle=style.get("style", "solid"),
                label=display_label,
            )
        ax.set_facecolor(panel)
        ax.set_title(f"{title} / ResNet", loc="left", color=foreground, fontsize=12, pad=10)
        ax.set_xlim(0.7, 10.3)
        ax.set_xticks((1, 5, 10), labels=("01", "05", "10"))
        ax.tick_params(colors=muted, labelsize=8, length=0, pad=6)
        ax.grid(color="#263244", alpha=0.65, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(1)

    axes[0].legend(loc="upper right", ncol=3, frameon=False, labelcolor="#c9d1d9",
                   fontsize=9, handlelength=2.2, columnspacing=1.4)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    from PIL import Image

    with Image.open(output) as preview:
        preview.convert("RGB").save(output, format="PNG", optimize=True)
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    render_social_preview() if "--social" in sys.argv[1:] else render()
