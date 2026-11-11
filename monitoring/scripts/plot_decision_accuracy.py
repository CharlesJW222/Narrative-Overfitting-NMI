"""Decision accuracy without and with GC (Fig. 4), from data/outputs.

    python scripts/plot_decision_accuracy.py   # writes gc_decision_accuracy.pdf/.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT  # noqa: E402
from analyze import MODELS, load, rates  # noqa: E402

LABELS = ["Qwen2.5\n7B", "Qwen2.5\n14B", "R1-Distill\nQwen-7B", "R1-Distill\nQwen-14B"]
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8})


def accuracy(setting, model, key):
    resp = load(f"response_{setting}_{model}.json")
    a_r, a_n = rates(resp, "attack", key), rates(resp, "control", key)
    return round((158 * (100 - a_r) + 80 * a_n) / 238, 1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    w = 0.32
    for ax, setting, title in zip(axes, ("orig", "anon"), ("(a) ORIG", "(b) ANON")):
        for off, key, colour, label in ((-w / 2 - 0.02, "baseline_accept", "#9aa5ae", "Without GC"),
                                        (w / 2 + 0.02, "gc_accept", "#315a7d", "With GC")):
            vals = [accuracy(setting, m, key) for m in MODELS]
            bars = ax.bar([i + off for i in range(4)], vals, w, color=colour, label=label, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center",
                        fontsize=7, fontweight="bold", color=colour)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e5e8ea", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Correct judgements (%)", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"gc_decision_accuracy.{ext}", dpi=200)


if __name__ == "__main__":
    main()
