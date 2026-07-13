#!/usr/bin/env python3
"""Render MNIST and CIFAR-10 benchmark curves as one two-panel figure."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import cifar10, mnist  # noqa: E402


ASSETS = ROOT / "docs/assets"
OUTPUT = ASSETS / "classification_benchmarks.png"
DATASETS = ("MNIST", "CIFAR-10")
SPECS = (
    ("MNIST", mnist),
    ("CIFAR-10", cifar10),
)
DISPLAY_LABELS = {
    mnist.PLOT_AGENT_LABEL: "GPT-5.5 w/ context",
    mnist.PLOT_NO_CONTEXT_LABEL: "GPT-5.5 w/o context",
}


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


if __name__ == "__main__":
    render()
