"""Task 6: mini-batch SGD, 50 epochs, learning rates x widths; per-epoch log and loss curves.

The graded run uses weights ~ N(0, 0.01) read as std = 0.01. A supplementary run repeats the
grid with std = 0.1 (the other reading, variance = 0.01) to show how much the init matters.
"""
import csv

import numpy as np

from common import RESULTS_DIR, SEED, load_data, set_seeds
from mlp import accuracy, backward, cross_entropy, forward, init_params, sgd_step

LEARNING_RATES = (0.001, 0.01, 0.1)
WIDTHS = (16, 64)
EPOCHS = 50
BATCH_SIZE = 32
SPEC_STD = 0.01
SUPPLEMENTARY_STD = 0.1


def train_one(width, lr, X_train, Y_train, X_test, Y_test, std=SPEC_STD, verbose=True):
    """Same initial weights and shuffle order for every learning rate of a given width."""
    params = init_params(width, std=std, rng=np.random.default_rng(SEED))
    shuffle_rng = np.random.default_rng(SEED)
    log = []
    for epoch in range(1, EPOCHS + 1):
        order = shuffle_rng.permutation(len(X_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            P, cache = forward(params, X_train[batch])
            sgd_step(params, backward(params, cache, P, Y_train[batch]), lr)
        row = {
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": cross_entropy(forward(params, X_train)[0], Y_train),
            "test_loss": cross_entropy(forward(params, X_test)[0], Y_test),
            "train_acc": accuracy(params, X_train, Y_train),
            "test_acc": accuracy(params, X_test, Y_test),
        }
        log.append(row)
        if verbose:
            print(f"W={width:<3d} lr={lr:<6g} epoch {epoch:2d}  train_loss={row['train_loss']:.4f}  "
                  f"test_loss={row['test_loss']:.4f}  test_acc={row['test_acc']:.4f}")
    return log


def spread_labels(ys, min_gap):
    """Nudge label positions (log10 units) apart so direct labels never overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        out[cur] = max(out[cur], out[prev] + min_gap)
    return out


def plot(logs, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {0.001: "#2a78d6", 0.01: "#eb6834", 0.1: "#1baf7a"}  # categorical slots 1-3
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2})
    all_losses = [r[k] for r in logs for k in ("train_loss", "test_loss")]
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.3026)
             if min(all_losses) * 0.8 <= t <= 2.6]
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(11, 4.6), sharey=True, facecolor=surface)
    for ax, width in zip(axes, WIDTHS):
        ax.set_facecolor(surface)
        ends = []
        # widest first so a thinner, later series stays visible where curves coincide
        for lr, lw in zip(LEARNING_RATES, (3.2, 2.0, 2.0)):
            rows = [r for r in logs if r["width"] == width and r["lr"] == lr]
            epochs = [r["epoch"] for r in rows]
            ax.plot(epochs, [r["train_loss"] for r in rows], color=colors[lr], lw=lw,
                    solid_capstyle="round")
            ax.plot(epochs, [r["test_loss"] for r in rows], color=colors[lr], lw=1.2,
                    ls=(0, (4, 3)))
            ends.append((lr, rows[-1]["train_loss"], rows[-1]["test_acc"]))
        label_y = spread_labels([np.log10(y) for _, y, _ in ends], min_gap=0.075)
        for (lr, y, acc), ly in zip(ends, label_y):
            ax.annotate(f"lr {lr:g} · test acc {acc:.3f}", xy=(EPOCHS, y), xytext=(EPOCHS + 1.5, 10 ** ly),
                        va="center", fontsize=8.5, color=ink2,
                        arrowprops=dict(arrowstyle="-", color=grid, lw=0.8, shrinkA=0, shrinkB=2))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["ln 10" if t > 2.3 else f"{t:g}" for t in ticks])
        ax.set_title(f"Hidden width W = {width}", loc="left", fontsize=11, color=ink)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, EPOCHS + 20)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.grid(True, axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    axes[0].set_ylabel("Cross-entropy loss (log scale)")
    handles = [Line2D([], [], color=colors[lr], lw=2, label=f"lr = {lr:g}") for lr in LEARNING_RATES]
    handles += [Line2D([], [], color=ink2, lw=2, label="train"),
                Line2D([], [], color=ink2, lw=1.2, ls=(0, (4, 3)), label="test")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def write_outputs(logs, suffix=""):
    with open(RESULTS_DIR / f"train_log{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows({k: (f"{v:.6f}" if isinstance(v, float) and k != "lr" else v)
                          for k, v in r.items()} for r in logs)
    with open(RESULTS_DIR / f"test_accuracy{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["width", "lr", "final_train_loss", "final_test_loss",
                                               "final_test_accuracy"])
        writer.writeheader()
        for r in logs:
            if r["epoch"] == EPOCHS:
                writer.writerow({"width": r["width"], "lr": r["lr"],
                                 "final_train_loss": f"{r['train_loss']:.4f}",
                                 "final_test_loss": f"{r['test_loss']:.4f}",
                                 "final_test_accuracy": f"{r['test_acc']:.4f}"})


def main() -> None:
    set_seeds()
    X_train, X_test, Y_train, Y_test = load_data()
    RESULTS_DIR.mkdir(exist_ok=True)
    data = (X_train, Y_train, X_test, Y_test)

    logs = [r for width in WIDTHS for lr in LEARNING_RATES for r in train_one(width, lr, *data)]
    write_outputs(logs)
    plot(logs, RESULTS_DIR / "loss_curves.png",
         "Loss curves, init N(0, 0.01²): SGD, batch 32, 50 epochs")

    set_seeds()
    logs = [r for width in WIDTHS for lr in LEARNING_RATES
            for r in train_one(width, lr, *data, std=SUPPLEMENTARY_STD, verbose=False)]
    write_outputs(logs, suffix="_std0.1")
    plot(logs, RESULTS_DIR / "loss_curves_std0.1.png",
         "Supplementary: init N(0, 0.1²), otherwise identical")


if __name__ == "__main__":
    main()
