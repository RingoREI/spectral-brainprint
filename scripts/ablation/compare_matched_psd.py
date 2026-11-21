"""Compare mixed- and common-PSD runs on paired outer predictions."""

import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from run import SEEDS, load_features, load_partitions, sha256
from summarize import VARIANTS, _arrays, compute_effects


TASKS = ("EC", "EO", "Imagery", "SSVEP")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_folder(root, task, model):
    if model == "MLP":
        return root / task.lower() / "full/runs"
    return root / "matched_lda/runs"


def prepared_inputs(root, task):
    directory = root / ("ec/prepared" if task == "EC" else "shared/prepared_other_tasks")
    hashes = {key: sha256(directory / name) for key, name in
              (("features", "features.npz"), ("splits", "splits.csv"))}
    data, _ = load_features(directory / "features.npz")
    roles = load_partitions(directory / "splits.csv", data, task, range(3))
    return hashes, data, roles


def check_result(result, filename, expected, input_hashes):
    task, fold, seed, variant = expected
    if (result["task"], result["fold"], result.get("seed"), result["variant"]) != expected:
        raise ValueError(f"Wrong experimental cell in {filename}")
    if (result["scope"] != "development" or result["input_sha256"] != input_hashes
            or (result["p"], result["a"]) != VARIANTS[variant]):
        raise ValueError(f"Wrong inputs or factor flags in {filename}")
    arrays = _arrays(result, filename)
    if "probabilities" not in arrays:
        raise ValueError(f"Missing probabilities in {filename}")
    return arrays


def plot_primary(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = ("#385d7a", "#a84716")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.8), sharey=True)
    for axis, task in zip(axes, TASKS):
        subset = [r for r in rows if r["task"] == task and r["contrast"] == "A_given_P"]
        for model, color in zip(("MLP", "LDA"), colors):
            model_rows = [r for r in subset if r["model"] == model]
            values = np.array([[r["mixed_pp"], r["common_pp"]] for r in model_rows])
            for pair in values:
                axis.plot([0, 1], pair, color=color, lw=0.8, alpha=0.45)
            axis.plot([0, 1], values.mean(axis=0), "o-", color=color, label=model, lw=2)
        axis.axhline(0, color="#666666", ls=":", lw=1)
        axis.spines[["right", "top"]].set_visible(False)
        axis.set(xlim=(-0.2, 1.2), xticks=[0, 1], xticklabels=["Mixed", "Common"], title=task)
    axes[0].set_ylabel("Increment from A36 (BA, pp)")
    axes[-1].legend(frameon=False)
    fig.suptitle("A36 gain above spatial features and band power")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"primary_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare(mixed_root, common_root, output):
    cells, effects, controls, sources, summary = [], [], [], {}, []
    for task in TASKS:
        old, new = [prepared_inputs(root, task) for root in (mixed_root, common_root)]
        hashes = [old[0], new[0]]
        if hashes[0]["splits"] != hashes[1]["splits"] or hashes[0]["features"] == hashes[1]["features"]:
            raise ValueError(f"{task}: require identical split files and distinct feature versions")
        for key in ("record_ids", "paths", "tasks", "y"):
            if not np.array_equal(old[1][key], new[1][key]):
                raise ValueError(f"{task}: prepared {key} changed")
        for cols in (slice(0, 171), slice(279, 315)):
            if not np.array_equal(old[1]["X"][:, cols], new[1]["X"][:, cols], equal_nan=True):
                raise ValueError(f"{task}: prepared S or A values changed")
        task_ids = old[1]["record_ids"][old[1]["tasks"] == task]
        split_ids = np.concatenate([old[1]["record_ids"][roles["outer_eval"]] for roles in old[2].values()])
        if not np.array_equal(np.sort(task_ids), np.sort(split_ids)):
            raise ValueError(f"{task}: outer folds must cover the full task once")
        for model in ("MLP", "LDA"):
            directories = [run_folder(root, task, model) for root in (mixed_root, common_root)]
            manifests = [read(d / "run_manifest.json") for d in directories]
            if any(m["status"] != "complete" for m in manifests):
                raise ValueError(f"Incomplete {model} experiment")
            for field in ("trainer_sha256", "environment", "config"):
                if manifests[0][field] != manifests[1][field]:
                    raise ValueError(f"{task}/{model}: {field} changed across versions")
            for seed in (SEEDS if model == "MLP" else (None,)):
                seed_name = f"-seed{seed}" if seed is not None else ""
                ba = [{}, {}]
                cohort = None
                for variant in VARIANTS:
                    records, truth = [], []
                    preds = [[], []]
                    for fold in range(3):
                        name = f"result-{task}-fold{fold}{seed_name}-{variant}.json"
                        files = [d / "results" / name for d in directories]
                        results = [read(path) for path in files]
                        vals = [check_result(result, name, (task, fold, seed, variant), h)
                                for result, h in zip(results, hashes)]
                        sources[f"{model}/{name}"] = [sha256(path) for path in files]
                        fields = ["protocol", "config", "n_train", "n_development", "n_validation"]
                        if model == "MLP":
                            fields += ["initial_state_sha256", "parameters"]
                        if any(results[0][k] != results[1][k] for k in fields):
                            raise ValueError(f"Model or roles changed in {name}")
                        expected_roles = old[2][fold]
                        expected_outer = expected_roles["outer_eval"]
                        if (not np.array_equal(old[1]["record_ids"][expected_outer], vals[0]["record_id"])
                                or not np.array_equal(old[1]["y"][expected_outer], vals[0]["truth"])
                                or results[0]["n_train"] != len(expected_roles["train"])
                                or results[0]["n_development"] != len(expected_roles["inner_dev"])):
                            raise ValueError(f"Prediction records or role counts differ from frozen split in {name}")
                        for field in ("truth", "record_id"):
                            if not np.array_equal(vals[0][field], vals[1][field]):
                                raise ValueError(f"Unpaired validation {field} in {name}")
                        records.extend(vals[0]["record_id"].tolist())
                        truth.extend(vals[0]["truth"].tolist())
                        for i, values in enumerate(vals):
                            preds[i].extend(values["prediction"].tolist())
                        if variant in ("M00", "M01"):
                            delta = float(np.max(np.abs(vals[0]["probabilities"] - vals[1]["probabilities"])))
                            changed = int(np.count_nonzero(vals[0]["prediction"] != vals[1]["prediction"]))
                            if changed or delta > (0 if model == "MLP" else 1e-12):
                                raise ValueError(f"Negative control changed in {name}: {changed} labels, max |dp|={delta}")
                            controls.append(dict(task=task, model=model, seed=seed, fold=fold, variant=variant,
                                                 records=len(vals[0]["truth"]), changed_labels=changed,
                                                 max_probability_difference=delta))
                    if len(set(records)) != len(records) or set(truth) != set(range(1, 21)):
                        raise ValueError(f"{task}: duplicate outer record or missing identity")
                    if cohort is None:
                        cohort = (records, truth)
                    elif (records, truth) != cohort:
                        raise ValueError(f"{task}: variant changed pooled cohort")
                    for i in range(2):
                        ba[i][variant] = float(balanced_accuracy_score(truth, preds[i]))
                    cells.append(dict(task=task, model=model, seed=seed, variant=variant, records=len(truth),
                                      mixed_ba=ba[0][variant], common_ba=ba[1][variant],
                                      change_pp=100 * (ba[1][variant] - ba[0][variant])))
                before, after = map(compute_effects, ba)
                for name in before:
                    effects.append(dict(task=task, model=model, seed=seed, contrast=name,
                                        mixed_pp=before[name], common_pp=after[name],
                                        change_pp=after[name] - before[name]))
            primary = [r for r in effects if r["task"] == task and r["model"] == model
                       and r["contrast"] == "A_given_P"]
            common = [r["common_pp"] for r in primary]
            summary.append(dict(task=task, model=model, repetitions=len(primary),
                                mixed_pp=statistics.mean(r["mixed_pp"] for r in primary),
                                common_pp=statistics.mean(common),
                                common_seed_sd_pp=statistics.stdev(common) if len(common) > 1 else None,
                                change_pp=statistics.mean(r["change_pp"] for r in primary),
                                min_common_pp=min(common), max_common_pp=max(common),
                                positive=sum(value > 0 for value in common)))
    write_csv(output / "paired_cells.csv", cells)
    write_csv(output / "paired_effects.csv", effects)
    write_csv(output / "primary_by_task.csv", summary)
    write_csv(output / "negative_controls.csv", controls)
    plot_primary(effects, output)
    provenance = {"fits_paired": len(sources), "negative_control_pairs": len(controls),
                  "script_sha256": sha256(__file__), "hash_order": ["mixed", "common"],
                  "paired_result_sha256": sources}
    (output / "validation.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-root", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a fresh comparison directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        output = Path(temporary) / "comparison"
        output.mkdir()
        compare(args.mixed_root, args.common_root, output)
        output.rename(args.output_dir)


if __name__ == "__main__":
    main()
