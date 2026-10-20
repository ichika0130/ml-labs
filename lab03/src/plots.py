"""Task 5: accuracy vs. parameters and vs. latency (CPU, batch 1 and batch 128)."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"MLP": "#2a78d6", "CNN": "#eb6834", "Attention": "#1baf7a"}  # categorical slots 1-3
ABLATION = "Attention (no positional embedding)"


def read(name):
    with open(RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


def main() -> None:
    acc = {r["model"]: float(r["test_accuracy"]) for r in read("test_accuracy.csv")}
    params = {r["model"]: int(r["params"]) for r in read("test_accuracy.csv")}
    lat = {(r["model"], int(r["batch"])): float(r["median_ms"])
           for r in read("latency.csv") if r["setting"] == "cpu-1t"}

    panels = [
        ("Accuracy vs. parameters", "Parameters (log scale)", lambda m: params[m], "{:,.0f}"),
        ("Accuracy vs. latency, batch 1", "Median latency per batch, ms (log scale)",
         lambda m: lat[(m, 1)], "{:.3f} ms"),
        ("Accuracy vs. latency, batch 128", "Median latency per batch, ms (log scale)",
         lambda m: lat[(m, 128)], "{:.2f} ms"),
    ]
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (title, xlabel, xval, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for model, color in COLORS.items():
            x, y = xval(model), acc[model]
            ax.scatter([x], [y], s=90, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{model}\n{fmt.format(x)} · {y:.3f}", (x, y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        if title.endswith("parameters"):
            x, y = params[ABLATION], acc[ABLATION]
            ax.scatter([x], [y], s=70, facecolor=SURFACE, edgecolor=COLORS["Attention"],
                       linewidth=2, zorder=3)
            ax.annotate(f"Attention, no pos. emb.\n{y:.3f}", (x, y), xytext=(9, -2),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        lo = min(xval(m) for m in COLORS)
        hi = max(xval(m) for m in COLORS)
        ax.set_xlim(lo / 1.6, hi * 4.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel(xlabel)
        ax.grid(True, color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Test accuracy (Fashion-MNIST, 5 epochs)")
    axes[0].set_ylim(0.835, 0.90)
    fig.suptitle("Architecture trade-offs: up and to the left is better (CPU, 1 thread)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "tradeoff.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
