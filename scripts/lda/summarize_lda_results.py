"""Validate frozen LDA results and produce aggregate experiment tables."""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


FEATURE_SETS = ("power108", "logcov171", "power_logcov279")
FEATURE_DIMENSIONS = {"power108": 108, "logcov171": 171, "power_logcov279": 279}
LABELS = np.arange(1, 21)
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")


def _metrics(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS,
                                  average="macro", zero_division=0)),
    }


def _positive_integer(value, field, filename):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{filename}: {field} must be a positive integer")
    return value


def _relative_input_hashes(result, filename):
    hashes = result["input_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{filename}: input_sha256 must be a nonempty mapping")
    for name, value in hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{filename}: invalid source input name")
        canonical = PurePosixPath(name)
        if canonical.is_absolute() or ".." in canonical.parts or ":" in name or "\\" in name:
            raise ValueError(f"{filename}: source inputs must use relative POSIX paths")
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)):
            raise ValueError(f"{filename}: invalid SHA256 for {name}")
    return hashes


def _prediction_arrays(result, filename):
    rows = result["predictions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{filename}: predictions must contain records")
    arrays = {}
    for field in ("record_id", "truth", "prediction"):
        values = [row[field] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{filename}: {field} must contain integers")
        arrays[field] = np.asarray(values, dtype=np.int64)
    ids = arrays["record_id"]
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{filename}: duplicate or negative record IDs")
    for field in ("truth", "prediction"):
        if not np.isin(arrays[field], LABELS).all():
            raise ValueError(f"{filename}: {field} must use identities 1 through 20")
    if len(rows) != result["n_validation"]:
        raise ValueError(f"{filename}: prediction count differs from n_validation")
    return arrays


def analyze_results(input_dir):
    """Read one task/protocol's fold JSON files and recompute all metrics.

    Each present fold must contain all three feature sets. Validation records
    are matched in their saved order, and each record occurs once across folds.
    The returned tables and provenance contain aggregate values and relative
    source file names. Per-record identifiers are used only during validation.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("fold*.json"))
    if not files:
        raise ValueError("Input directory contains no fold*.json result files")

    runs = {}
    arrays = {}
    source_files = []
    source_hashes = None
    experiment = None
    environments = []
    for path in files:
        content = path.read_bytes()
        result = json.loads(content.decode("utf-8-sig"))
        feature_set = result["feature_set"]
        fold = result["fold"]
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"{path.name}: unsupported feature set {feature_set}")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ValueError(f"{path.name}: fold must be a nonnegative integer")
        if path.name != f"fold{fold}-{feature_set}.json":
            raise ValueError(f"{path.name}: filename and result metadata differ")
        key = (fold, feature_set)
        if key in runs:
            raise ValueError(f"Duplicate result for fold {fold}, {feature_set}")
        _positive_integer(result["n_train"], "n_train", path.name)
        _positive_integer(result["n_validation"], "n_validation", path.name)
        identity = (result["task"], result["protocol"])
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"{path.name}: task and protocol must be named")
        if experiment is None:
            experiment = identity
        elif identity != experiment:
            raise ValueError("Results mix different tasks or protocols")

        hashes = _relative_input_hashes(result, path.name)
        if source_hashes is None:
            source_hashes = hashes
        elif hashes != source_hashes:
            raise ValueError(f"{path.name}: frozen source input hashes differ")
        history_match = result["matches_history_at_1e_12"]
        if not isinstance(history_match, bool):
            raise ValueError(f"{path.name}: history match status must be boolean")
        environment = result["environment"]
        if not isinstance(environment, dict):
            raise ValueError(f"{path.name}: environment must be a mapping")
        if environment not in environments:
            environments.append(environment)

        values = _prediction_arrays(result, path.name)
        recomputed = _metrics(values["truth"], values["prediction"])
        for name, value in recomputed.items():
            stored = result["metrics"][name]
            if (isinstance(stored, bool) or not isinstance(stored, (int, float))
                    or not np.isfinite(stored) or abs(stored - value) > 1e-12):
                raise ValueError(f"{path.name}: saved {name} differs from its predictions")
        runs[key] = {**result, "metrics": recomputed}
        arrays[key] = values
        source_files.append({"filename": path.name,
                             "sha256": hashlib.sha256(content).hexdigest(),
                             "matches_history_at_1e_12": history_match})

    folds = sorted({fold for fold, _ in runs})
    seen_ids = set()
    for fold in folds:
        if {feature for candidate, feature in runs if candidate == fold} != set(FEATURE_SETS):
            raise ValueError(f"Fold {fold} must contain exactly the three feature sets")
        baseline = arrays[(fold, "logcov171")]
        for feature_set in FEATURE_SETS:
            values = arrays[(fold, feature_set)]
            if (not np.array_equal(values["record_id"], baseline["record_id"])
                    or not np.array_equal(values["truth"], baseline["truth"])):
                raise ValueError(f"Fold {fold}: validation IDs or labels differ for {feature_set}")
            if runs[(fold, feature_set)]["n_train"] != runs[(fold, "logcov171")]["n_train"]:
                raise ValueError(f"Fold {fold}: training counts differ for {feature_set}")
        ids = set(baseline["record_id"].tolist())
        if seen_ids.intersection(ids):
            raise ValueError(f"Fold {fold}: validation record IDs overlap another fold")
        seen_ids.update(ids)

    summary_rows = []
    fold_rows = []
    per_identity_rows = []
    confusions = {}
    pooled = {}
    for feature_set in FEATURE_SETS:
        truth = np.concatenate([arrays[(fold, feature_set)]["truth"] for fold in folds])
        prediction = np.concatenate([arrays[(fold, feature_set)]["prediction"] for fold in folds])
        pooled[feature_set] = prediction
        correct = int(np.count_nonzero(truth == prediction))
        summary_rows.append({"feature_set": feature_set,
                             "n_features": FEATURE_DIMENSIONS[feature_set],
                             "n_validation": len(truth), "correct": correct,
                             "errors": len(truth) - correct, **_metrics(truth, prediction)})
        confusions[feature_set] = confusion_matrix(truth, prediction, labels=LABELS)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=LABELS, zero_division=0)
        per_identity_rows.extend({"feature_set": feature_set, "identity": int(label),
                                  "support": int(support[index]),
                                  "precision": float(precision[index]),
                                  "recall": float(recall[index]), "f1": float(f1[index])}
                                 for index, label in enumerate(LABELS))
        for fold in folds:
            result = runs[(fold, feature_set)]
            baseline_score = runs[(fold, "logcov171")]["metrics"]["balanced_accuracy"]
            fold_rows.append({"fold": fold, "feature_set": feature_set,
                              "n_train": result["n_train"],
                              "n_validation": result["n_validation"], **result["metrics"],
                              "delta_vs_logcov171_pp": 100 * (
                                  result["metrics"]["balanced_accuracy"] - baseline_score)})

    baseline_correct = pooled["logcov171"] == truth
    full_correct = pooled["power_logcov279"] == truth
    transitions = {
        "both_correct": int(np.count_nonzero(baseline_correct & full_correct)),
        "corrected": int(np.count_nonzero(~baseline_correct & full_correct)),
        "regressed": int(np.count_nonzero(baseline_correct & ~full_correct)),
        "both_wrong": int(np.count_nonzero(~baseline_correct & ~full_correct)),
    }
    task, protocol = experiment
    provenance = {
        "task": task, "protocol": protocol, "folds": folds,
        "source_files": source_files, "input_sha256": source_hashes,
        "environments": environments,
        "aggregation": "Metrics recomputed from pooled nonduplicated validation predictions",
        "validation_records": len(seen_ids), "identity_labels": LABELS.tolist(),
        "all_history_metrics_match": all(item["matches_history_at_1e_12"] for item in source_files),
    }
    return {"summary_rows": summary_rows, "fold_rows": fold_rows,
            "per_identity_rows": per_identity_rows, "confusions": confusions,
            "transitions": transitions, "task": task, "protocol": protocol,
            "folds": folds, "provenance": provenance}


def _write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(analysis, output_dir):
    """Write aggregate CSVs and a source manifest into an output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in [("metrics_summary.csv", "summary_rows"),
                          ("fold_metrics.csv", "fold_rows"),
                          ("per_identity.csv", "per_identity_rows")]:
        _write_csv(output_dir / filename, analysis[key])
    transitions = [{"baseline": "logcov171", "comparison": "power_logcov279",
                    "transition": key, "count": value}
                   for key, value in analysis["transitions"].items()]
    _write_csv(output_dir / "paired_errors.csv", transitions)
    for feature_set, matrix in analysis["confusions"].items():
        rows = [{"true_identity": int(label),
                 **{f"predicted_{int(predicted)}": int(matrix[index, column])
                    for column, predicted in enumerate(LABELS)}}
                for index, label in enumerate(LABELS)]
        _write_csv(output_dir / f"confusion_{feature_set}.csv", rows)
    with (output_dir / "provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(analysis["provenance"], handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def plot_results(analysis, output_dir):
    """Export three figures as 300 dpi PNG and vector PDF using aggregate data."""
    from pathlib import Path

    import numpy as np

    try:
        import matplotlib
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise RuntimeError("Install the project's [plots] extra to generate figures") from error

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    feature_sets = ("power108", "logcov171", "power_logcov279")
    labels = {
        "power108": "Power (108)",
        "logcov171": "Log-covariance (171)",
        "power_logcov279": "Power + log-covariance (279)",
    }
    colors = {"power108": "#737C87", "logcov171": "#336A9E", "power_logcov279": "#CE772B"}
    markers = {"power108": "^", "logcov171": "s", "power_logcov279": "o"}
    summary = {row["feature_set"]: row for row in analysis["summary_rows"]}
    folds = sorted(analysis["folds"])
    fold_metrics = {(row["fold"], row["feature_set"]): row for row in analysis["fold_rows"]}
    identities = sorted(row["identity"] for row in analysis["per_identity_rows"]
                        if row["feature_set"] == "power_logcov279")
    identity_metrics = {(row["identity"], row["feature_set"]): row
                        for row in analysis["per_identity_rows"]}
    n_records = summary["power_logcov279"]["n_validation"]
    context = (f"{len(identities)} identities  |  {len(folds)} filename-block validation folds"
               f"  |  {n_records:,} unique held-out predictions")
    saved = []

    def save_figure(figure, name):
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white",
                           metadata={"Title": name, "Creator": "Spectral Brainprint"})
            saved.append(path.name)
        plt.close(figure)

    def style_axis(axis, horizontal_grid=False):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#A8AFB7")
        axis.tick_params(colors="#303841", length=3)
        axis.set_axisbelow(True)
        axis.grid(axis="x" if horizontal_grid else "y", color="#E5E8EB", linewidth=0.6)

    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.titlesize": 13, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.titleweight": "normal", "text.color": "#20272E",
        "axes.labelcolor": "#20272E", "axes.facecolor": "white",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})
        figure.subplots_adjust(left=0.065, right=0.98, top=0.76, bottom=0.19, wspace=0.28)
        figure.suptitle(f"{analysis['task']} | LDA spectral and spatial feature ablation", y=0.98)
        figure.legend(handles=[Patch(facecolor=colors[name], label=labels[name]) for name in feature_sets],
                      loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
        axis = axes[0]
        x = np.arange(3)
        metrics = ("accuracy", "balanced_accuracy", "macro_f1")
        for index, name in enumerate(feature_sets):
            values = np.array([summary[name][key] for key in metrics]) * 100
            positions = x + (index - 1) * 0.24
            axis.bar(positions, values, width=0.22, color=colors[name], edgecolor="none")
            for position, value in zip(positions, values):
                axis.text(position, min(value + 1.0, 98.5), f"{value:.2f}",
                          ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x, ["Accuracy", "Balanced accuracy", "Macro-F1"])
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_ylabel("Pooled score (%)")
        axis.set_title("A  Pooled validation predictions", loc="left", pad=12)
        style_axis(axis)

        axis = axes[1]
        positions = np.arange(len(folds))
        for name in feature_sets:
            values = np.array([fold_metrics[(fold, name)]["balanced_accuracy"] for fold in folds]) * 100
            axis.plot(positions, values, color=colors[name], marker=markers[name],
                      linewidth=1.6, markersize=5.5, markeredgecolor="white", markeredgewidth=0.6)
        axis.set_xticks(positions, [f"Fold {fold}" for fold in folds])
        axis.set_xlim(-0.25, len(folds) - 0.75)
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_ylabel("Balanced accuracy (%)")
        axis.set_title("B  Observed validation folds", loc="left", pad=12)
        style_axis(axis)
        figure.text(0.065, 0.065, context, fontsize=9, color="#505B66")
        save_figure(figure, "lda_performance")

        figure, axis = plt.subplots(figsize=(8.3, 8.2))
        figure.subplots_adjust(left=0.18, right=0.95, top=0.84, bottom=0.12)
        figure.suptitle(f"{analysis['task']} | Identity-level recall", y=0.98)
        legend = [Line2D([0], [0], color=colors[name], marker=markers[name],
                         linestyle="none", markersize=6, label=labels[name])
                  for name in ("logcov171", "power_logcov279")]
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.52, 0.93),
                      ncol=2, frameon=False)
        baseline = np.array([identity_metrics[(identity, "logcov171")]["recall"] for identity in identities]) * 100
        combined = np.array([identity_metrics[(identity, "power_logcov279")]["recall"] for identity in identities]) * 100
        positions = np.arange(len(identities))
        axis.hlines(positions, np.minimum(baseline, combined), np.maximum(baseline, combined),
                    color="#A8AFB7", linewidth=1.6, zorder=2)
        for name, values in [("logcov171", baseline), ("power_logcov279", combined)]:
            axis.scatter(values, positions, color=colors[name], marker=markers[name],
                         s=37, edgecolor="white", linewidth=0.5, zorder=3)
        ylabels = [f"{identity:02d}  (n={identity_metrics[(identity, 'power_logcov279')]['support']})"
                   for identity in identities]
        axis.set_yticks(positions, ylabels)
        axis.set_ylim(len(identities) - 0.5, -0.5)
        axis.set_xlim(-1, 101)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Recall (%)")
        axis.set_ylabel("Identity and validation records")
        style_axis(axis, horizontal_grid=True)
        improved = int(np.sum(combined > baseline))
        equal = int(np.sum(combined == baseline))
        decreased = int(np.sum(combined < baseline))
        figure.text(0.18, 0.055, f"Recall improved: {improved}  |  Equal: {equal}  |  Decreased: {decreased}",
                    fontsize=9, color="#505B66")
        figure.text(0.18, 0.025, f"Pooled validation predictions, n={n_records:,}",
                    fontsize=9, color="#505B66")
        save_figure(figure, "lda_identity_recall")

        matrix = np.asarray(analysis["confusions"]["power_logcov279"])
        support = matrix.sum(axis=1, keepdims=True)
        percent = np.divide(100.0 * matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0)
        figure, axis = plt.subplots(figsize=(8.6, 8.1))
        figure.subplots_adjust(left=0.09, right=0.9, top=0.87, bottom=0.12)
        figure.suptitle(f"{analysis['task']} | Power + log-covariance LDA", y=0.98)
        axis.set_title("Color: within-identity prediction share (%)  |  Text: record count", pad=13,
                       fontsize=10)
        image = axis.imshow(percent, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        ticks = np.arange(len(identities))
        tick_labels = [f"{identity:02d}" for identity in identities]
        axis.set_xticks(ticks, tick_labels)
        axis.set_yticks(ticks, tick_labels)
        axis.set_xlabel("Predicted identity")
        axis.set_ylabel("True identity")
        axis.tick_params(length=0)
        for (row, column), value in np.ndenumerate(matrix):
            if value:
                axis.text(column, row, str(int(value)), ha="center", va="center",
                          color="white" if percent[row, column] >= 55 else "#20272E", fontsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.025)
        colorbar.set_ticks(np.arange(0, 101, 20))
        colorbar.ax.tick_params(length=2)
        colorbar.outline.set_visible(False)
        figure.text(0.09, 0.04, context, fontsize=9, color="#505B66")
        save_figure(figure, "lda_confusion")
    return saved


def main(argv=None):
    import argparse
    import importlib.metadata
    import tempfile

    parser = argparse.ArgumentParser(description="Summarize matched LDA folds and export research figures.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing fold*-FEATURE.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for aggregate tables and figures")
    parser.add_argument("--skip-plots", action="store_true", help="Export tables without Matplotlib")
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("--output-dir already exists; choose a new report directory")
    if not args.skip_plots:
        try:
            import matplotlib
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            parser.error('Install plotting dependencies with: python -m pip install -e ".[plots]"')

    analysis = analyze_results(args.input_dir)
    analysis["provenance"]["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    names = ["numpy", "scikit-learn"] + ([] if args.skip_plots else ["matplotlib"])
    analysis["provenance"]["summary_environment"] = {
        name: importlib.metadata.version(name) for name in names
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".lda-report-") as staging:
        prepared = Path(staging) / "report"
        write_tables(analysis, prepared)
        if not args.skip_plots:
            plot_results(analysis, prepared)
        prepared.rename(destination)

    for row in analysis["summary_rows"]:
        print(f"{row['feature_set']}: balanced accuracy {row['balanced_accuracy']:.4%}, "
              f"macro-F1 {row['macro_f1']:.4%}, {row['correct']}/{row['n_validation']} correct")
    print(f"Aggregate report: {destination}")
    if not analysis["provenance"]["all_history_metrics_match"]:
        print("Some source runs differ from their historical metrics; see provenance.json.")


if __name__ == "__main__":
    main()
