"""Re-render the credit-default figure at column-friendly geometry.

Reads the same docs/assets artifacts as examples/credit_card.py plot() and
reproduces its curve semantics (mean over seeds of incumbent-best validation
log loss per trial), but sized for a single AAAI column: no in-figure title,
8pt fonts, and the legend naming the context condition explicitly.
"""
import json
import glob
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper/src/figures/credit_card.png"
N_TRIALS = 20

styles = {
    "Random": dict(color="#7A7A7A", linestyle=(0, (2, 2)), lw=1.8),
    "TPE": dict(color="#111827", linestyle=(0, (6, 2)), lw=1.8),
    "GP-BO": dict(color="#0072B2", linestyle=(0, (1, 1)), lw=1.6),
    "GPT-5.5": dict(color="#D55E00", marker="^", lw=1.4),
    "GPT-5.5-no-context": dict(color="#CC79A7", marker="D", linestyle=(0, (4, 2)), lw=1.4),
}
labels = {
    "Random": "Random",
    "TPE": "TPE",
    "GP-BO": "GP-BO",
    "GPT-5.5": "GPT-5.5 w/ context",
    "GPT-5.5-no-context": "GPT-5.5 w/o context",
}


def mean_incumbent_curve(method):
    per_seed = []
    for f in sorted(glob.glob(str(REPO / f"docs/assets/credit_default_{method}_s*.json"))):
        d = json.load(open(f))
        assert d["method"] == method, (f, d["method"])
        vals = d["values"]
        assert len(vals) == N_TRIALS
        inc, best = [], None
        for v in vals:
            best = v if best is None else min(best, v)
            inc.append(best)
        per_seed.append(inc)
    assert len(per_seed) == 5, (method, len(per_seed))
    return [sum(s[t] for s in per_seed) / len(per_seed) for t in range(N_TRIALS)]


plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, ax = plt.subplots(figsize=(3.5, 1.72))
trials = range(1, N_TRIALS + 1)
for method in ("Random", "TPE", "GP-BO", "GPT-5.5", "GPT-5.5-no-context"):
    ax.plot(trials, mean_incumbent_curve(method), ms=3.0, label=labels[method], **styles[method])
ax.set_xlabel("Trial")
ax.set_ylabel("Best validation log loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(alpha=0.18)
ax.legend(ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
