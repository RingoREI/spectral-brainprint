"""Plot pooled task metrics and observed fold gains from aggregate LDA CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


TASKS = ("EC", "EO", "Imagery", "SSVEP")
FEATURES = ("power108", "logcov171", "power_logcov279")
COLORS = ("#737C87", "#336A9E", "#CE772B")
NAMES = ("Power (108)", "Log-covariance (171)", "Power + log-covariance (279)")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metrics(results_dir):
    rows = read_rows(results_dir / "lda-within-task-metrics.csv")
    pooled = {(row["task"], row["feature_set"]): row for row in rows}
    expected = {(task, feature) for task in TASKS for feature in FEATURES}
    if len(rows) != 12 or set(pooled) != expected:
        raise ValueError("Expected one pooled row for each of four tasks and three feature sets")
    fold_gains = {}
    for task in TASKS:
        counts = {int(pooled[(task, feature)]["n_validation"]) for feature in FEATURES}
        if len(counts) != 1 or min(counts) <= 0:
            raise ValueError(f"{task}: mismatched validation counts")
        if any(not 0 <= float(pooled[(task, feature)]["balanced_accuracy"]) <= 1 for feature in FEATURES):
            raise ValueError(f"{task}: invalid balanced accuracy")
        folds = read_rows(results_dir / f"lda-{task.lower()}-within-task/fold_metrics.csv")
        keyed = {(int(row["fold"]), row["feature_set"]): row for row in folds}
        if len(folds) != 9 or set(keyed) != {(fold, feature) for fold in range(3) for feature in FEATURES}:
            raise ValueError(f"{task}: expected three complete validation folds")
        fold_gains[task] = [100 * (float(keyed[(fold, "power_logcov279")]["balanced_accuracy"])
                                    - float(keyed[(fold, "logcov171")]["balanced_accuracy"]))
                            for fold in range(3)]
    return pooled, fold_gains


def plot_overview(results_dir, output_prefix):
    pooled, fold_gains = load_metrics(results_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
             "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
             "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), gridspec_kw={"width_ratios": [1.65, 1]})
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.20, top=0.74, wspace=0.26)
        fig.suptitle("LDA feature ablation across task conditions", fontsize=14, y=0.99)
        fig.legend(handles=[Patch(facecolor=color, label=name) for color, name in zip(COLORS, NAMES)],
                   ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=10)
        axis = axes[0]
        x = np.arange(len(TASKS))
        for index, feature in enumerate(FEATURES):
            scores = [100 * float(pooled[(task, feature)]["balanced_accuracy"]) for task in TASKS]
            positions = x + (index - 1) * 0.24
            axis.bar(positions, scores, width=0.22, color=COLORS[index], edgecolor="none")
            for position, value in zip(positions, scores):
                axis.annotate(f"{value:.2f}", (position, value), xytext=(0, 3),
                              textcoords="offset points", ha="center", va="bottom", fontsize=8)
        counts = [int(pooled[(task, "power108")]["n_validation"]) for task in TASKS]
        axis.set_xticks(x, [f"{task}\n(n={count:,})" for task, count in zip(TASKS, counts)])
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_ylabel("Pooled balanced accuracy (%)")
        axis.set_title("A  Matched representations within each task", loc="left", pad=20, fontsize=11)

        axis = axes[1]
        for position, task in enumerate(TASKS):
            values = fold_gains[task]
            axis.scatter(position + np.array([-0.16, 0, 0.16]), values, s=32,
                         facecolors="none", edgecolors="#697482", linewidths=1.1, zorder=3)
            gain = 100 * (float(pooled[(task, "power_logcov279")]["balanced_accuracy"])
                          - float(pooled[(task, "logcov171")]["balanced_accuracy"]))
            axis.scatter(position, gain, marker="D", s=50, color=COLORS[2],
                         edgecolors="white", linewidths=0.7, zorder=4)
            offset = (12, 0) if task == "Imagery" else (6, 7)
            axis.annotate(f"+{gain:.2f}", (position, gain), xytext=offset,
                          textcoords="offset points", va="center" if task == "Imagery" else "bottom",
                          color="#A45B1D", fontsize=9, zorder=5)
        axis.axhline(0, color="#A8AFB7", linewidth=0.8)
        axis.set_xticks(x, TASKS)
        axis.set_xlim(-0.5, len(TASKS) - 0.35)
        axis.set_ylim(-0.2, max(max(values) for values in fold_gains.values()) + 1.0)
        axis.set_ylabel("Combined minus log-covariance (pp)")
        axis.set_title("B  Increment from adding power", loc="left", pad=20, fontsize=11)
        axis.legend(handles=[Line2D([], [], marker="D", color="none", markerfacecolor=COLORS[2],
                                    markeredgecolor="white", markersize=7, label="Pooled gain"),
                             Line2D([], [], marker="o", color="none", markerfacecolor="none",
                                    markeredgecolor="#697482", markersize=6, label="Observed fold")],
                    loc="upper left", frameon=False, fontsize=8.5, ncol=2, handletextpad=0.4,
                    columnspacing=1.0)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#A8AFB7")
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E5E8EB", linewidth=0.6)
            axis.tick_params(length=3, colors="#303841")
        fig.text(0.06, 0.055, "20 registered identities | Same-session, within-task development protocol",
                 fontsize=9, color="#505B66")
        fig.text(0.06, 0.015, "Fold points describe filename-block partitions; they are not independent replications.",
                 fontsize=8.5, color="#505B66")
        for suffix in ("png", "pdf"):
            fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight",
                        metadata={"Title": "LDA feature ablation across task conditions", "Creator": "Spectral Brainprint"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    plot_overview(args.results_dir, args.output_prefix)


if __name__ == "__main__":
    main()
