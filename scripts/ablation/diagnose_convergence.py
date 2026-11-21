"""Audit saved training/inner-development curves without using outer outcomes.

This is a descriptive review of a completed, fixed-budget experiment. It neither
changes the training protocol nor declares convergence from an invented cutoff.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np


TASKS = ("EC", "EO", "Imagery", "SSVEP")
VARIANTS = ("M00", "M10", "M01", "M11")
INPUT_NAMES = {"M00": "S", "M10": "S+P", "M01": "S+A", "M11": "S+P+A"}
FOLD_COLORS = ("#31688E", "#CC7631", "#278275")
SEED_STYLES = ("-", "--", ":")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected a finite nonnegative value")
    return value


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def tail_slope(values, epochs):
    """OLS slope in logged NLL units per epoch; no threshold is applied."""
    if len(values) < 2:
        return None
    centered = epochs - epochs.mean()
    return float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))


def diagnose_run(value):
    """Read configuration and training fields only, never outer metrics/predictions."""
    config = value["config"]
    maximum = positive_integer(config["max_epochs"], "max_epochs")
    completed = positive_integer(value["epochs_completed"], "epochs_completed")
    selected = positive_integer(value["selected_epoch"], "selected_epoch")
    curve = value["training_curve"]
    if not isinstance(curve, list) or len(curve) != completed or completed > maximum:
        raise ValueError("Curve length and configured/completed budget disagree")
    if [row["epoch"] for row in curve] != list(range(1, completed + 1)):
        raise ValueError("Epochs must be consecutive, beginning at 1")
    train = np.array([finite_nonnegative(row["train_nll"], "train_nll") for row in curve])
    dev = np.array([finite_nonnegative(row["development_nll"], "development_nll") for row in curve])
    best_index = int(dev.argmin())
    if selected != best_index + 1:
        raise ValueError("Selected epoch is not the first actual minimum inner-development NLL")
    best_saved = finite_nonnegative(value["best_development_nll"], "best_development_nll")
    if not np.isclose(best_saved, dev[best_index], rtol=0, atol=1e-12):
        raise ValueError("Saved best NLL differs from selected curve value")
    window = min(10, completed)
    indices = np.arange(completed - window + 1, completed + 1, dtype=float)
    tail_dev, tail_train = dev[-window:], train[-window:]
    task, variant = value["task"], value["variant"]
    fold, seed = value["fold"], value["seed"]
    if task not in TASKS or variant not in VARIANTS or type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Unknown task, condition or fold")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    row = dict(task=task, variant=variant, fold=fold, seed=seed,
               n_train=positive_integer(value["n_train"], "n_train"),
               n_development=positive_integer(value["n_development"], "n_development"),
               max_epochs=maximum, epochs_completed=completed, selected_epoch=selected,
               reached_budget=int(completed == maximum),
               selected_at_budget=int(selected == maximum),
               selected_at_final_epoch=int(selected == completed),
               selected_in_final_10_epochs=int(selected >= completed - window + 1),
               training_nll_first=float(train[0]), training_nll_at_selected=float(train[best_index]),
               training_nll_final=float(train[-1]), development_nll_first=float(dev[0]),
               development_nll_best=float(dev[best_index]), development_nll_final=float(dev[-1]),
               development_final_minus_best_nll=float(dev[-1] - dev[best_index]),
               development_minus_logged_training_nll_at_selected=float(dev[best_index] - train[best_index]),
               tail_epoch_count=window, tail_first_epoch=int(indices[0]),
               tail_development_nll_slope=tail_slope(tail_dev, indices),
               tail_training_nll_slope=tail_slope(tail_train, indices),
               tail_development_nll_change=float(tail_dev[-1] - tail_dev[0]),
               tail_development_relative_improvement_percent=(None if tail_dev[0] == 0 else
                   100 * float((tail_dev[0] - tail_dev[-1]) / tail_dev[0])),
               relative_improvement_status="undefined_zero_start" if tail_dev[0] == 0 else "defined")
    epochs = [dict(task=task, variant=variant, fold=fold, seed=seed,
                   epoch=index + 1, train_nll=float(train[index]), development_nll=float(dev[index]),
                   is_selected=int(index + 1 == selected)) for index in range(completed)]
    return row, epochs


def load_experiment(root):
    """Require four complete, three-fold, matched-condition task experiments."""
    diagnostics, trajectories, sources, configurations = [], [], [], {}
    for task in TASKS:
        folder = root / task.lower() / "full/runs"
        manifest_path = folder / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("scope") != "development" or manifest.get("task") != task:
            raise ValueError(f"{task}: require a completed development run manifest")
        config = manifest["config"]
        seeds, folds = config["seeds"], config["folds"]
        if sorted(folds) != [0, 1, 2] or len(set(seeds)) != len(seeds) or len(seeds) < 3:
            raise ValueError("Require all three folds and at least three distinct seeds")
        if config["selection"] != "minimum inner-development NLL; min_delta controls patience only; no outer evaluation during fitting":
            raise ValueError("This review expects the frozen minimum-inner-NLL selection rule")
        configurations[task] = config
        expected = {(fold, seed, variant) for fold in folds for seed in seeds for variant in VARIANTS}
        actual = set()
        role_counts = {}
        for path in sorted((folder / "results").glob("result*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["task"] != task or value.get("scope") != "development" or value["config"] != config:
                raise ValueError(f"{path.name}: task, scope or frozen settings differ from manifest")
            key = (value["fold"], value["seed"], value["variant"])
            if key in actual or key not in expected:
                raise ValueError("Duplicate or unexpected run")
            actual.add(key)
            row, epochs = diagnose_run(value)
            previous = role_counts.setdefault(value["fold"], (row["n_train"], row["n_development"]))
            if previous != (row["n_train"], row["n_development"]):
                raise ValueError("Training/development counts differ across matched runs")
            diagnostics.append(row)
            trajectories.extend(epochs)
            sources.append(dict(file=path.relative_to(root).as_posix(), sha256=sha256(path)))
        if actual != expected:
            raise ValueError(f"{task}: missing formal runs: {sorted(expected - actual)}")
        sources.append(dict(file=manifest_path.relative_to(root).as_posix(), sha256=sha256(manifest_path)))
    return diagnostics, trajectories, sources, configurations


def aggregate(diagnostics, keys):
    grouped = {}
    for row in diagnostics:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    fields = ("epochs_completed", "selected_epoch", "training_nll_at_selected", "development_nll_best",
              "development_nll_final", "development_final_minus_best_nll",
              "development_minus_logged_training_nll_at_selected", "tail_development_nll_slope",
              "tail_development_relative_improvement_percent")
    results = []
    for group, rows in grouped.items():
        result = dict(zip(keys, group))
        result.update(n_runs=len(rows), n_reached_budget=sum(row["reached_budget"] for row in rows),
                      n_selected_at_budget=sum(row["selected_at_budget"] for row in rows),
                      n_selected_at_final_epoch=sum(row["selected_at_final_epoch"] for row in rows),
                      n_final_above_selected_nll=sum(row["development_final_minus_best_nll"] > 0 for row in rows),
                      n_negative_tail_development_slope=sum(row["tail_development_nll_slope"] is not None and row["tail_development_nll_slope"] < 0 for row in rows))
        for field in fields:
            values = np.array([row[field] for row in rows if row[field] is not None])
            result[field + "_n_defined"] = len(values)
            for name, fn in (("mean", np.mean), ("median", np.median), ("minimum", np.min), ("maximum", np.max)):
                result[field + "_" + name] = None if not len(values) else float(fn(values))
            result[field + "_descriptive_run_sd"] = None if len(values) < 2 else float(values.std(ddof=1))
        results.append(result)
    return results


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty table")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(diagnostics, trajectories, configurations, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    groups = {}
    for row in trajectories:
        groups.setdefault((row["task"], row["variant"], row["fold"], row["seed"]), []).append(row)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        figure, axes = plt.subplots(4, 4, figsize=(13, 10.6), sharex=True)
        figure.subplots_adjust(left=.085, right=.985, top=.86, bottom=.105, hspace=.32, wspace=.30)
        for i, task in enumerate(TASKS):
            seeds = sorted(configurations[task]["seeds"])
            for j, variant in enumerate(VARIANTS):
                axis = axes[i, j]
                for fold in range(3):
                    for k, seed in enumerate(seeds):
                        rows = groups[(task, variant, fold, seed)]
                        x, y = [r["epoch"] for r in rows], [r["development_nll"] for r in rows]
                        axis.plot(x, y, color=FOLD_COLORS[fold], linestyle=SEED_STYLES[k % 3], linewidth=.8, alpha=.7)
                        selected = next(row for row in rows if row["is_selected"])
                        axis.scatter(selected["epoch"], selected["development_nll"], color=FOLD_COLORS[fold], s=9, zorder=3)
                if all(row["development_nll"] > 0 for key, rows in groups.items() if key[:2] == (task, variant) for row in rows):
                    axis.set_yscale("log")
                    axis.set_ylabel(f"{task}\nInner-dev NLL (log axis)" if j == 0 else "")
                else:
                    axis.set_ylim(bottom=0)
                    axis.set_ylabel(f"{task}\nInner-dev NLL (linear axis)" if j == 0 else "")
                if i == 0:
                    axis.set_title(f"{variant}: {INPUT_NAMES[variant]}", fontsize=11, pad=12)
                if i == 3:
                    axis.set_xlabel("Epoch")
                axis.set_xlim(0, configurations[task]["max_epochs"] + 2)
                axis.grid(axis="y", color="#E2E6EA", linewidth=.5)
                axis.set_axisbelow(True)
        handles = [Line2D([0], [0], color=color, label=f"Fold {fold}") for fold, color in enumerate(FOLD_COLORS)]
        handles.append(Line2D([0], [0], color="#646E78", marker="o", linewidth=0, markersize=4, label="Selected checkpoint"))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .947), ncol=4, frameon=False)
        figure.suptitle("Inner-development trajectories | All fixed-budget MLP runs", y=.995, fontsize=15)
        figure.text(.085, .045, "Each panel contains every fold/seed trajectory; line style distinguishes seeds. Lines stop when the saved run stops.", fontsize=9)
        figure.text(.085, .020, "No outer-evaluation trajectory is used. A decreasing tail is descriptive and is not a convergence test.", fontsize=9)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"development_trajectories.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
        figure.subplots_adjust(left=.075, right=.98, top=.82, bottom=.14, hspace=.36, wspace=.18)
        for axis, task in zip(axes.flat, TASKS):
            all_rows = [row for row in diagnostics if row["task"] == task]
            maximum = configurations[task]["max_epochs"]
            for j, variant in enumerate(VARIANTS):
                rows = sorted((row for row in all_rows if row["variant"] == variant), key=lambda r: (r["fold"], r["seed"]))
                jitter = np.linspace(-.16, .16, len(rows))
                axis.scatter(j + jitter, [row["epochs_completed"] for row in rows], facecolors="none", edgecolors="#929AA3", s=35, linewidths=.9, zorder=2)
                axis.scatter(j + jitter, [row["selected_epoch"] for row in rows], color="#31688E", s=17, zorder=3)
            axis.axhline(maximum, color="#CC7631", linestyle="--", linewidth=.8, zorder=1)
            reached = sum(row["reached_budget"] for row in all_rows)
            selected = sum(row["selected_at_budget"] for row in all_rows)
            axis.set_title(f"{task}: {reached}/{len(all_rows)} at budget; {selected} selected at budget", loc="left", fontsize=10)
            axis.set_xticks(range(4), [INPUT_NAMES[v] for v in VARIANTS])
            axis.set_ylabel("Epoch")
            axis.set_ylim(0, maximum + 6)
            axis.grid(axis="y", color="#E2E6EA", linewidth=.5)
            axis.set_axisbelow(True)
        handles = [Line2D([0], [0], marker="o", color="#31688E", linewidth=0, markersize=4, label="Selected epoch"),
                   Line2D([0], [0], marker="o", markerfacecolor="none", color="#929AA3", linewidth=0, markersize=5, label="Completed epochs"),
                   Line2D([0], [0], color="#CC7631", linestyle="--", label="Maximum budget")]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .93), ncol=3, frameon=False)
        figure.suptitle("Checkpoint location and stopping budget | Every fitted model", y=.995, fontsize=14)
        figure.text(.075, .065, "Each condition shows all fold/seed pairs; selected and completed points share the same horizontal position.", fontsize=9)
        figure.text(.075, .025, "Epoch distributions describe this experiment; folds and seeds are not independent acquisition replicates.", fontsize=9)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"checkpoint_budget.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def write_readme(diagnostics, task_summary, output, include_plots=True):
    count = len(diagnostics)
    reached = sum(row["reached_budget"] for row in diagnostics)
    selected = sum(row["selected_at_budget"] for row in diagnostics)
    lines = ["# Descriptive convergence and budget review", "",
             f"This review covers all {count} fitted MLP models from the completed four-task feature ablation. It reads only saved configuration, training loss and inner-development loss; outer metrics and predictions are not used to select settings or diagnose the optimization trajectory.", "",
             "## What the saved trajectories show", "",
             "| Task | Fits | Reached maximum budget | Selected at maximum | Negative final-tail dev-NLL slope | Median final minus best dev NLL | Median final-tail relative dev-NLL improvement (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in task_summary:
        improvement = row["tail_development_relative_improvement_percent_median"]
        improvement_text = "NA" if improvement is None else f"{improvement:.3f}"
        lines.append(f"| {row['task']} | {row['n_runs']} | {row['n_reached_budget']} | {row['n_selected_at_budget']} | {row['n_negative_tail_development_slope']} | {row['development_final_minus_best_nll_median']:.6f} | {improvement_text} |")
    lines += ["", f"Overall, **{reached}/{count}** runs reached the configured maximum, and **{selected}/{count}** selected their best inner-development checkpoint at that maximum. A best checkpoint at the boundary limits the ability to claim the optimum was reached within this budget. It does not by itself prove that extra training would improve outer recognition.", ""]
    if include_plots:
        lines += ["![All inner-development trajectories](development_trajectories.png)", "", "![Selected epochs and stopping budgets](checkpoint_budget.png)", ""]
    lines += ["## Interpretation rules", "",
              "- Negative tail slope means that the least-squares trend of the final min(10, completed_epochs) saved inner-development NLL values is downward. There is no numerical threshold for declaring flatness, convergence, significance or undertraining. A downward slope is descriptive evidence of a continuing trend, not a formal convergence test.",
              "- Tail relative improvement is 100 × (first tail NLL − final tail NLL) / first tail NLL. Positive values mean lower loss over the tail. It is undefined if the first tail NLL is zero; no epsilon replacement or silent undefined-value exclusion is used. Defined counts are exported.",
              "- Final minus best development NLL measures deterioration after the chosen checkpoint. The model actually evaluated outside training used the best checkpoint, not the final checkpoint. A positive value describes the later development trajectory and is not, by itself, proof of overfitting.",
              "- Logged training NLL averages successive minibatches while parameters change and dropout is active. Development NLL uses the end-of-epoch model in eval mode on different records. Their difference is exported explicitly as an observed logged-loss difference; it is not a matched-checkpoint generalization-gap estimate and should not be interpreted as a direct overfitting test.",
              "- Run-level SD, medians and ranges are descriptive summaries. Fold/seed runs share data and are not independent acquisition samples. They do not support naive confidence intervals or significance tests.", "",
              "## Can this fixed-budget experiment close?", "",
              "The feature ablation can be documented as a completed **fixed-budget development comparison**, with the budget-boundary counts and all results retained. The optimization review does not establish that every run converged, that the architecture was fully tuned, or that the reported score is its best possible performance. EC and EO in particular have many checkpoints at the budget boundary. Before making an architecture ranking, use a comparator fitted on exactly the same inner-training records and preprocessing scope. Keep real source-group and cross-session limitations visible.", "",
              "No model was retrained and no learning rate, patience, seed, epoch budget or primary outcome was changed by this review. If a different budget is explored later, freeze a new protocol and use fresh outputs; do not replace the current scores or change the budget in response to outer performance.", "",
              "## Files and definitions", "",
              "| File | Meaning |", "|---|---|",
              "| run_diagnostics.csv | One row per actual fit, including selected/final NLL and final-tail descriptors |",
              "| by_task.csv | Per-task counts and descriptive run summaries |",
              "| by_task_variant.csv | Per-task, per-condition summaries |",
              "| by_task_fold.csv | Per-task, per-fold summaries; not independent replicate statistics |",
              "| epoch_trajectories.csv | Every saved training/inner-development epoch value |",
              "| provenance.json | Result and manifest hashes, settings, script hash and scope |", "",
              "NLL uses natural-log units. Tail slopes use NLL units per epoch. Relative tail improvement is a percentage, not a BA percentage-point effect. Primary feature effects and outer-error analysis live in their separate reports. The figures use log NLL axes only when all displayed values are strictly positive; a zero value triggers a linear axis rather than an arbitrary floor.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Artifact root containing <task>/full/runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output exists; choose a fresh diagnostics directory")
    diagnostics, trajectories, sources, configurations = load_experiment(args.input_root)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convergence-", dir=args.output_dir.parent) as temporary:
        output = Path(temporary) / "report"
        output.mkdir()
        write_csv(output / "run_diagnostics.csv", diagnostics)
        write_csv(output / "epoch_trajectories.csv", trajectories)
        task_summary = aggregate(diagnostics, ("task",))
        write_csv(output / "by_task.csv", task_summary)
        write_csv(output / "by_task_variant.csv", aggregate(diagnostics, ("task", "variant")))
        write_csv(output / "by_task_fold.csv", aggregate(diagnostics, ("task", "fold")))
        if not args.skip_plots:
            plot_results(diagnostics, trajectories, configurations, output)
        write_readme(diagnostics, task_summary, output, include_plots=not args.skip_plots)
        provenance = dict(
            analysis="descriptive convergence/budget review, fixed tail of at most 10 saved epochs",
            source_files=sources, generator_sha256=sha256(__file__),
            configurations=configurations, environment=dict(python=platform.python_version(), numpy=np.__version__),
            outcome_access="Only config, training_curve, selected/best development fields and train/dev counts; no outer outcomes used",
            uncertainty="No inferential CI, p-value or numerical convergence threshold", retraining=False,
            plots_generated=not args.skip_plots,
            training_loss_caveat="Dynamic minibatches with dropout; not a fixed-checkpoint train-vs-dev comparison")
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output.rename(args.output_dir)
    print(f"Reviewed {len(diagnostics)} saved runs; no retraining. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
