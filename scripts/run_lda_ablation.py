"""Reproduce the three EC LDA feature sets on the frozen development folds."""

import argparse
import csv
import json
from pathlib import Path

from spectral_brainprint.evaluation.reproduce import reproduce


FEATURE_SETS = ("power108", "logcov171", "power_logcov279")
PROTOCOL = "within_task_blocked_provisional"
TASK = "EC"
FOLDS = (0, 1, 2)
COLUMNS = ("fold", "feature_set", "n_train", "n_validation", "balanced_accuracy",
           "delta_vs_logcov171_pp", "matches_history_at_1e_12")


def validation_pairs(result):
    return [(row["record_id"], row["truth"]) for row in result["predictions"]]


def run_fold(package_root, fold):
    results = []
    for feature_set in FEATURE_SETS:
        result = reproduce(package_root, feature_set, PROTOCOL, TASK, fold)
        print(f"EC fold {fold} {feature_set}: {result['metrics']['balanced_accuracy']:.4%}",
              flush=True)
        results.append(result)

    reference = results[0]
    expected_pairs = validation_pairs(reference)
    for result in results[1:]:
        if validation_pairs(result) != expected_pairs:
            raise ValueError(f"EC fold {fold}: validation IDs or labels differ for {result['feature_set']}")
        if (result["n_train"] != reference["n_train"]
                or result["input_sha256"] != reference["input_sha256"]):
            raise ValueError(f"EC fold {fold}: frozen inputs differ for {result['feature_set']}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True, help="Periodic handoff directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for JSON and CSV")
    parser.add_argument("--fold", type=int, choices=FOLDS, help="Run just this fold (default: all three)")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")

    folds = FOLDS if args.fold is None else (args.fold,)
    results_by_fold = [run_fold(args.package_root, fold) for fold in folds]

    # Write outputs only after every requested fold has passed the alignment checks.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = args.output_dir / "summary.csv"
    with summary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for results in results_by_fold:
            baseline = next(r for r in results if r["feature_set"] == "logcov171")
            baseline_score = baseline["metrics"]["balanced_accuracy"]
            for result in results:
                filename = f"fold{result['fold']}-{result['feature_set']}.json"
                with (args.output_dir / filename).open("x", encoding="utf-8") as target:
                    json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
                    target.write("\n")
                score = result["metrics"]["balanced_accuracy"]
                row = {key: result[key] for key in ("fold", "feature_set", "n_train", "n_validation",
                                                    "matches_history_at_1e_12")}
                row.update(balanced_accuracy=score, delta_vs_logcov171_pp=100 * (score - baseline_score))
                writer.writerow(row)

    print(f"Summary: {summary}")
    return 0 if all(r["matches_history_at_1e_12"] for fold in results_by_fold for r in fold) else 2


if __name__ == "__main__":
    raise SystemExit(main())
