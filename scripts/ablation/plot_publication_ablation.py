"""Render the README figure from the two aggregate protocol-comparison tables."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np


TASKS = ("EC", "EO", "Imagery", "SSVEP")
VARIANTS = ("M00", "M10", "M01", "M11")
SEEDS = (20261102, 20261103, 20261104)
DATA_DIR = Path(__file__).resolve().parents[2] / "docs/results/ablation"


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def figure_data(data_dir):
    """Align the figure by seed and verify each contrast against both cell scores."""
    cells = {}
    for row in read_table(data_dir / "paired_cells.csv"):
        if row["model"] != "MLP":
            continue
        key = row["task"], int(row["seed"]), row["variant"]
        values = np.array([float(row["mixed_ba"]), float(row["common_ba"])])
        if key in cells or not np.all((values >= 0) & (values <= 1)):
            raise ValueError(f"Invalid or duplicate BA cell {key}")
        cells[key] = int(row["records"]), values

    effects = {}
    for row in read_table(data_dir / "paired_effects.csv"):
        if (row["model"], row["contrast"]) != ("MLP", "A_given_P"):
            continue
        key = row["task"], int(row["seed"])
        if key in effects:
            raise ValueError(f"Duplicate primary contrast {key}")
        effects[key] = np.array([float(row["mixed_pp"]), float(row["common_pp"])])

    if set(cells) != {(t, s, v) for t in TASKS for s in SEEDS for v in VARIANTS}:
        raise ValueError("Expected four MLP arms for each task and each prespecified seed")
    if set(effects) != {(t, s) for t in TASKS for s in SEEDS}:
        raise ValueError("Expected a paired MLP primary contrast for each task and seed")

    means = np.empty((len(VARIANTS), len(TASKS)))
    paired = np.empty((len(TASKS), len(SEEDS), 2))
    counts = []
    for j, task in enumerate(TASKS):
        cohort = {cells[task, s, v][0] for s in SEEDS for v in VARIANTS}
        if len(cohort) != 1 or min(cohort) <= 0:
            raise ValueError(f"Validation support differs in {task}")
        counts.append(cohort.pop())
        for i, variant in enumerate(VARIANTS):
            means[i, j] = 100 * np.mean([cells[task, s, variant][1][1] for s in SEEDS])
        for k, seed in enumerate(SEEDS):
            expected = 100 * (cells[task, seed, "M11"][1] - cells[task, seed, "M10"][1])
            if not np.allclose(effects[task, seed], expected, rtol=0, atol=1e-10):
                raise ValueError(f"Primary contrast differs from BA cells for {task}, seed {seed}")
            paired[j, k] = effects[task, seed]
    return means, paired, counts


def plot(data_dir, output_dir):
    means, paired, counts = figure_data(data_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#172b3a", "axes.labelcolor": "#172b3a",
                         "xtick.color": "#304453", "ytick.color": "#304453"})
    fig = plt.figure(figsize=(8.5, 6.8), facecolor="white")
    left, width = .146, .737
    fig.text(.027, .955, "A", size=15, weight="bold")
    fig.text(.078, .955, "Identification accuracy with a common PSD", size=15, weight="bold")
    fig.text(.078, .915, "Mean MLP balanced accuracy (%) across three training seeds", size=10)
    matrix = fig.add_axes([left, .585, width, .247])
    cmap = LinearSegmentedColormap.from_list("ink", ["#f2f7fa", "#9bc5d5", "#255770"])
    norm = Normalize(70, 100)
    heat = matrix.imshow(means, aspect="auto", cmap=cmap, norm=norm)
    for i in range(len(VARIANTS)):
        for j in range(len(TASKS)):
            value = means[i, j]
            color = "white" if value >= 92.5 else "#102a39"
            matrix.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=11, color=color)
    matrix.set_xticks(range(len(TASKS)), TASKS, fontsize=11, weight="bold")
    matrix.set_yticks(range(len(VARIANTS)), ["S", "S + P", "S + A", "S + P + A"])
    matrix.xaxis.tick_top()
    matrix.tick_params(which="both", length=0, pad=7)
    matrix.set_xticks(np.arange(1, len(TASKS)) - .5, minor=True)
    matrix.set_yticks(np.arange(1, len(VARIANTS)) - .5, minor=True)
    matrix.grid(which="minor", color="white", linewidth=1.2)
    matrix.spines[:].set_visible(False)
    scale = fig.add_axes([.900, .585, .011, .247])
    cbar = fig.colorbar(heat, cax=scale, ticks=[70, 85, 100])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9)
    for j, count in enumerate(counts):
        fig.text(left + (j + .5) * width / len(TASKS), .557, f"n = {count:,}",
                 ha="center", fontsize=9, color="#4b6070")

    fig.text(.027, .480, "B", size=15, weight="bold")
    fig.text(.078, .480, "Increment from adding A36 to S + P", size=15, weight="bold")
    fig.text(.078, .440, "Change in MLP balanced accuracy (percentage points)", size=10)
    mean_color, seed_color = "#934326", "#788e9b"
    handles = [Line2D([], [], color=seed_color, lw=1.1, marker="o", markersize=3,
                      label="Paired seed"),
               Line2D([], [], color=mean_color, lw=2.4, marker="o", markersize=5,
                      label="Seed mean")]
    fig.legend(handles=handles, bbox_to_anchor=(.887, .462), loc="upper right",
               ncol=2, frameon=False, borderaxespad=0, handlelength=1.5, columnspacing=1,
               fontsize=9)
    lower, upper = .5 * np.floor(paired.min() / .5), .5 * np.ceil(paired.max() / .5)
    for j, task in enumerate(TASKS):
        x0 = left + j * width / len(TASKS)
        axis = fig.add_axes([x0, .126, width / len(TASKS) - .009, .233])
        axis.set_title(task, fontsize=11, weight="bold", pad=7)
        axis.set_xlim(-.26, 1.26)
        axis.set_ylim(min(0, lower) - .05, max(0, upper) + .05)
        axis.yaxis.set_major_locator(MultipleLocator(.5))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.grid(axis="y", color="#dae2e7", linewidth=.65, zorder=0)
        axis.axhline(0, color="#5b6972", linestyle=(0, (3, 2)), lw=.9)
        for pair in paired[j]:
            axis.plot([0, 1], pair, color=seed_color, lw=1.2, marker="o", markersize=3)
        axis.plot([0, 1], paired[j].mean(axis=0), color=mean_color, lw=2.3,
                  marker="o", markersize=5, zorder=4)
        axis.set_xticks([0, 1], ["Mixed", "Common"], fontsize=9)
        axis.tick_params(axis="both", length=0, pad=4)
        axis.tick_params(axis="y", labelleft=j == 0, labelsize=9)
        axis.spines[:].set_visible(False)
    fig.text(.078, .057, "S = spatial covariance     P = band power     A = aperiodic offset and exponent",
             fontsize=9.5)
    fig.text(.078, .022, "Mixed PSD uses 2 s mean Welch for P. A36 and common-PSD P both use 1 s median Welch.",
             fontsize=9, color="#4b6070")
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"ablation-overview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    plot(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
