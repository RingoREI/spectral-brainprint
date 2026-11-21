"""Describe paired M11-versus-M10 errors without treating seeds as new records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import balanced_accuracy_score

try:
    from .summarize import analyze_results, LABELS
except ImportError:
    from summarize import analyze_results, LABELS


TASKS = ("EC", "EO", "Imagery", "SSVEP")
REFERENCE, CANDIDATE = "M10", "M11"


def paired_counts(truth, reference, candidate):
    """Audit the four mutually exclusive outcomes on one paired record set."""
    truth, reference, candidate = (np.asarray(v) for v in (truth, reference, candidate))
    if truth.ndim != 1 or not len(truth) or reference.shape != truth.shape or candidate.shape != truth.shape:
        raise ValueError("Paired arrays must be nonempty aligned vectors")
    old, new = reference == truth, candidate == truth
    corrected, regressed = int(np.sum(~old & new)), int(np.sum(old & ~new))
    return dict(support=len(truth), both_correct=int(np.sum(old & new)),
                corrected=corrected, regressed=regressed, both_wrong=int(np.sum(~old & ~new)),
                reference_recall=float(old.mean()), candidate_recall=float(new.mean()),
                recall_delta_pp=100 * (corrected - regressed) / len(truth))


def record_stability(ids, truth, references, candidates):
    """One row per unique record, with repeated-model error counts as attributes."""
    ids, truth = np.asarray(ids), np.asarray(truth)
    old, new = np.asarray(references), np.asarray(candidates)
    if (truth.ndim != 1 or ids.shape != truth.shape or len(np.unique(ids)) != len(ids)
            or old.ndim != 2 or old.shape != new.shape or old.shape[1] != len(ids) or not len(old)):
        raise ValueError("Stability requires unique ordered IDs and matched seed prediction matrices")
    old_wrong, new_wrong = old != truth, new != truth
    return [dict(record_id=int(ids[i]), identity=int(truth[i]), n_seeds=len(old),
                 reference_wrong_seeds=int(old_wrong[:, i].sum()), candidate_wrong_seeds=int(new_wrong[:, i].sum()),
                 both_wrong_seeds=int((old_wrong[:, i] & new_wrong[:, i]).sum()),
                 corrected_seeds=int((old_wrong[:, i] & ~new_wrong[:, i]).sum()),
                 regressed_seeds=int((~old_wrong[:, i] & new_wrong[:, i]).sum())) for i in range(len(ids))]


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(artifact_root):
    identity, identity_summary, seed_totals, fold_rows = [], [], [], []
    confusion, confusion_summary, stability, task_rows, sources = [], [], [], [], {}
    protocols = set()
    for task in TASKS:
        run_dir = artifact_root / task.lower() / "full/runs"
        accepted = analyze_results(run_dir)
        if accepted["task"] != task:
            raise ValueError("Task directory differs from result metadata")
        protocols.add(accepted["protocol"])
        seeds, folds = accepted["seeds"], accepted["folds"]
        provenance = accepted["provenance"]
        sources[task] = provenance
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["scope"] != "development" or manifest["task"] != task:
            raise ValueError("Error diagnostics require a completed development manifest")
        sources[task]["run_manifest_sha256"] = hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest()
        runs = {}
        for path in sorted(run_dir.rglob("result*.json")):
            result = json.loads(path.read_text(encoding="utf-8-sig"))
            runs[(result["seed"], result["fold"], result["variant"])] = result
        ids = np.concatenate([[r["record_id"] for r in runs[(seeds[0], fold, REFERENCE)]["predictions"]] for fold in folds])
        truth = np.concatenate([[r["truth"] for r in runs[(seeds[0], fold, REFERENCE)]["predictions"]] for fold in folds])
        predictions = {}
        for seed in seeds:
            for variant in (REFERENCE, CANDIDATE):
                predictions[(seed, variant)] = np.concatenate([[r["prediction"] for r in runs[(seed, fold, variant)]["predictions"]] for fold in folds])
            old, new = predictions[(seed, REFERENCE)], predictions[(seed, CANDIDATE)]
            overall = paired_counts(truth, old, new)
            overall["accuracy_delta_pp"] = overall.pop("recall_delta_pp")
            overall["reference_accuracy"] = overall.pop("reference_recall")
            overall["candidate_accuracy"] = overall.pop("candidate_recall")
            overall["reference_ba"] = float(balanced_accuracy_score(truth, old))
            overall["candidate_ba"] = float(balanced_accuracy_score(truth, new))
            overall["ba_delta_pp"] = 100 * (overall["candidate_ba"] - overall["reference_ba"])
            seed_totals.append(dict(task=task, seed=seed, reference=REFERENCE, comparison=CANDIDATE, **overall))
            per_identity = []
            for label in LABELS:
                mask = truth == label
                values = paired_counts(truth[mask], old[mask], new[mask])
                entry = dict(task=task, seed=seed, identity=int(label), **values)
                identity.append(entry)
                per_identity.append(entry)
                for wrong_label in LABELS:
                    if wrong_label == label:
                        continue
                    old_count = int(np.sum(old[mask] == wrong_label))
                    new_count = int(np.sum(new[mask] == wrong_label))
                    confusion.append(dict(task=task, seed=seed, identity=int(label), predicted_identity=int(wrong_label),
                                          support=int(mask.sum()), reference_count=old_count, candidate_count=new_count,
                                          count_delta=new_count - old_count,
                                          reference_rate=float(old_count / mask.sum()), candidate_rate=float(new_count / mask.sum())))
            if not np.isclose(np.mean([r["recall_delta_pp"] for r in per_identity]), overall["ba_delta_pp"], rtol=0, atol=1e-10):
                raise ValueError("Identity-weighted corrections do not recover BA")
            for fold in folds:
                old_result, new_result = runs[(seed, fold, REFERENCE)], runs[(seed, fold, CANDIDATE)]
                fold_truth = np.array([r["truth"] for r in old_result["predictions"]])
                if set(fold_truth.tolist()) != set(LABELS.tolist()):
                    raise ValueError("Every fold must cover all 20 registered identities")
                old_prediction = np.array([r["prediction"] for r in old_result["predictions"]])
                new_prediction = np.array([r["prediction"] for r in new_result["predictions"]])
                values = paired_counts(fold_truth, old_prediction, new_prediction)
                values["accuracy_delta_pp"] = values.pop("recall_delta_pp")
                values["reference_accuracy"] = values.pop("reference_recall")
                values["candidate_accuracy"] = values.pop("candidate_recall")
                values["reference_ba"] = float(old_result["metrics"]["balanced_accuracy"])
                values["candidate_ba"] = float(new_result["metrics"]["balanced_accuracy"])
                values["ba_delta_pp"] = 100 * (values["candidate_ba"] - values["reference_ba"])
                fold_rows.append(dict(task=task, seed=seed, fold=fold, **values))
        task_identity_summary = []
        for label in LABELS:
            selected = [r for r in identity if r["task"] == task and r["identity"] == label]
            supports = {r["support"] for r in selected}
            if len(supports) != 1:
                raise ValueError("Identity support differs across seeds")
            deltas = [r["recall_delta_pp"] for r in selected]
            entry = dict(task=task, identity=int(label), support=supports.pop(), n_seeds=len(seeds),
                         **{f"recall_delta_{name}_pp": value for name, value in stats(deltas).items()},
                         improved_seeds=sum(v > 0 for v in deltas), regressed_seeds=sum(v < 0 for v in deltas), unchanged_seeds=sum(v == 0 for v in deltas))
            for field in ("reference_recall", "candidate_recall", "corrected", "regressed", "both_wrong"):
                entry[f"{field}_mean"] = float(np.mean([r[field] for r in selected]))
            identity_summary.append(entry)
            task_identity_summary.append(entry)
            for wrong_label in LABELS:
                if wrong_label == label:
                    continue
                pair = [r for r in confusion if r["task"] == task and r["identity"] == label and r["predicted_identity"] == wrong_label]
                entry = dict(task=task, identity=int(label), predicted_identity=int(wrong_label), support=selected[0]["support"], n_seeds=len(seeds))
                for field in ("reference_count", "candidate_count", "count_delta", "reference_rate", "candidate_rate"):
                    entry.update({f"{field}_{name}": value for name, value in stats([r[field] for r in pair]).items()})
                confusion_summary.append(entry)
        task_stability = record_stability(ids, truth, [predictions[(s, REFERENCE)] for s in seeds], [predictions[(s, CANDIDATE)] for s in seeds])
        stability.extend(dict(task=task, **r) for r in task_stability)
        totals = [r for r in seed_totals if r["task"] == task]
        task_rows.append(dict(task=task, unique_records=len(ids), n_seeds=len(seeds),
                              ba_delta_mean_pp=float(np.mean([r["ba_delta_pp"] for r in totals])),
                              ba_delta_sd_pp=float(np.std([r["ba_delta_pp"] for r in totals], ddof=1)),
                              identity_mean_improved=sum(r["recall_delta_mean_pp"] > 0 for r in task_identity_summary),
                              identity_mean_regressed=sum(r["recall_delta_mean_pp"] < 0 for r in task_identity_summary),
                              identity_mean_unchanged=sum(r["recall_delta_mean_pp"] == 0 for r in task_identity_summary),
                              identity_regressed_all_seeds=sum(r["regressed_seeds"] == len(seeds) for r in task_identity_summary),
                              corrected_mean_per_seed=float(np.mean([r["corrected"] for r in totals])),
                              regressed_mean_per_seed=float(np.mean([r["regressed"] for r in totals])),
                              reference_wrong_all_seeds=sum(r["reference_wrong_seeds"] == len(seeds) for r in task_stability),
                              candidate_wrong_all_seeds=sum(r["candidate_wrong_seeds"] == len(seeds) for r in task_stability),
                              both_wrong_all_seeds=sum(r["both_wrong_seeds"] == len(seeds) for r in task_stability),
                              negative_fold_seed_primary=sum(r["ba_delta_pp"] < 0 for r in fold_rows if r["task"] == task)))
    if len(protocols) != 1:
        raise ValueError("Task reports use different protocols")
    return dict(identity_by_seed=identity, identity_summary=identity_summary, seed_totals=seed_totals,
                fold_seed_primary=fold_rows, confusion_by_seed=confusion, confusion_summary=confusion_summary,
                record_stability=stability, task_summary=task_rows,
                provenance=dict(tasks=list(TASKS), protocol=protocols.pop(), reference=REFERENCE, comparison=CANDIDATE,
                                primary_contrast="A_given_P", source_reports=sources,
                                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                validator_sha256=hashlib.sha256(Path(analyze_results.__code__.co_filename).read_bytes()).hexdigest(),
                                inference="Post-hoc descriptive diagnostics, no confidence intervals or significance tests",
                                repeated_records="Predictions pool unique folds within a seed; seed-level estimates are summarized, not stacked"))


def plot(analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        lookup = {(r["task"], r["identity"]): r["recall_delta_mean_pp"] for r in analysis["identity_summary"]}
        values = np.array([[lookup[(task, int(label))] for label in LABELS] for task in TASKS])
        limit = max(1., float(np.abs(values).max()))
        figure, axis = plt.subplots(figsize=(13, 4.7))
        figure.subplots_adjust(left=.09, right=.94, top=.79, bottom=.26)
        im = axis.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        for i in range(len(TASKS)):
            for j in range(len(LABELS)):
                axis.text(j, i, f"{values[i, j]:+.1f}" if values[i, j] else "0.0", ha="center", va="center", fontsize=7.5,
                          color="white" if abs(values[i, j]) > .6 * limit else "#252525")
        axis.set_xticks(range(len(LABELS)), [f"{label:02d}" for label in LABELS]); axis.set_yticks(range(len(TASKS)), TASKS)
        axis.set_xlabel("Registered identity ID"); axis.tick_params(length=0, pad=7)
        for spine in axis.spines.values(): spine.set_visible(False)
        bar = figure.colorbar(im, ax=axis, fraction=.025, pad=.025)
        bar.set_label("Recall increment (pp)")
        figure.suptitle("Where does A36 help or hurt? | M11 minus M10", y=.97, fontsize=14)
        figure.text(.09, .87, "Per-identity recall difference: mean of paired seed estimates after pooling folds", fontsize=10)
        figure.text(.09, .13, "Blue: increased recall; red: decreased recall. Three seeds; no confidence intervals or significance markers.", fontsize=9)
        figure.text(.09, .065, "Same-session development; each record is evaluated once per seed. Support and seed ranges remain in the CSV tables.", fontsize=8.5)
        for suffix in ("png", "pdf"): figure.savefig(output / f"identity_effects.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        figure.subplots_adjust(left=.11, right=.98, top=.78, bottom=.25)
        positions = np.arange(len(TASKS)); width=.28
        for i, task in enumerate(TASKS):
            rows = [r for r in analysis["seed_totals"] if r["task"] == task]
            corrected = np.array([r["corrected"] for r in rows], dtype=float)
            regressed = np.array([r["regressed"] for r in rows], dtype=float)
            axis.bar(i - .17, corrected.mean(), width=width, color="#4C8EA9", label="Corrected" if i == 0 else None)
            axis.bar(i + .17, -regressed.mean(), width=width, color="#BD6C59", label="Regressed (below zero)" if i == 0 else None)
            axis.scatter(i - .17 + np.linspace(-.065,.065,len(rows)), corrected, facecolors="none", edgecolors="#333333", s=22, zorder=4)
            axis.scatter(i + .17 + np.linspace(-.065,.065,len(rows)), -regressed, facecolors="none", edgecolors="#333333", s=22, zorder=4)
            axis.scatter(i, (corrected-regressed).mean(), marker="D", color="#20272E", s=42, label="Net corrected mean" if i == 0 else None, zorder=5)
        axis.axhline(0, color="#636A70", linewidth=.8)
        axis.set_xticks(positions, TASKS); axis.set_ylabel("Paired record counts per seed")
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E5E8EB", linewidth=.6); axis.set_axisbelow(True)
        figure.suptitle("A36 changes both correct and incorrect predictions", y=.98, fontsize=14)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53,.91), ncol=3, frameon=False, fontsize=9)
        figure.text(.11, .14, "Bars: mean per seed; hollow circles: each seed. A record can recur across models, without becoming a new sample.", fontsize=9)
        figure.text(.11, .075, "Counts weight records equally. BA instead weights identity recalls equally, so net counts do not recover the BA gain.", fontsize=8.5)
        figure.text(.11, .025, "Primary contrast: S+P+A versus S+P; same-session provisional development data; no inferential error bars.", fontsize=8.5)
        for suffix in ("png", "pdf"): figure.savefig(output / f"paired_turnover.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def write_report(analysis, output, *, with_plots=True):
    for name in ("identity_by_seed", "identity_summary", "seed_totals", "fold_seed_primary", "confusion_by_seed", "confusion_summary", "record_stability", "task_summary"):
        write_csv(output / f"{name}.csv", analysis[name])
    (output / "provenance.json").write_text(json.dumps(analysis["provenance"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Paired Error Diagnostics: A36 Added to S+P", "", "**Post-hoc descriptive diagnostics of the frozen development experiment.** The primary comparison remains M11 (S+P+A) minus M10 (S+P). Model settings, checkpoints and evaluation records are unchanged by this analysis.", "",
             "All four conditions and all three folds are revalidated with the existing summarizer before diagnosis: ordered record IDs and truths, seed/fold completeness, input/configuration hashes, class probabilities and prediction-derived metrics. Predictions pool nonduplicated folds within each seed. No seed is chosen as best, and seeds are not stacked into additional independent records.", "",
             "## Heterogeneity and persistent errors", "",
             "| Task | Unique records per seed | Mean BA gain ± seed SD (pp) | Identities with positive/negative/zero mean recall gain | Identities regressing in every seed | M11 wrong in every seed (unique records) |", "|---|---:|---:|---:|---:|---:|"]
    for r in analysis["task_summary"]:
        lines.append(f"| {r['task']} | {r['unique_records']:,} | {r['ba_delta_mean_pp']:+.3f} ± {r['ba_delta_sd_pp']:.3f} | {r['identity_mean_improved']}/{r['identity_mean_regressed']}/{r['identity_mean_unchanged']} | {r['identity_regressed_all_seeds']} | {r['candidate_wrong_all_seeds']} |")
    lines += ["", "![Identity recall increments](identity_effects.png)", "", "Positive aggregate BA does not mean that every identity benefits. Identities with zero change can include ceiling cases and offsetting corrections/regressions. Seed SD and ranges describe training randomness conditional on this cohort and its partitions, not confidence intervals, independent acquisition repetition or statistical significance.", "",
              "The unique-record stability table has one row per original evaluation record and reports how often that record is wrong across models. It does not multiply the cohort size by the number of seeds. An error in every seed is a repeated failure under these fitted models; it does not identify a physiological cause or prove an intrinsically difficult recording.", "",
              "## Largest identity-level regressions and gains", "", "Ranked by seed-mean recall change for description only. All 20 identities and all seed values remain in the CSVs. Identity IDs are enrolled class labels, not names.", "",
              "| Task | Direction | Identity | Support per seed | Mean gain (pp) | Seed range (pp) | Positive/negative seeds |", "|---|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        selected = [r for r in analysis["identity_summary"] if r["task"] == task]
        for direction, entries in (("Regression", sorted((r for r in selected if r["recall_delta_mean_pp"] < 0), key=lambda r:r["recall_delta_mean_pp"])[:3]),
                                   ("Gain", sorted((r for r in selected if r["recall_delta_mean_pp"] > 0), key=lambda r:r["recall_delta_mean_pp"], reverse=True)[:3])):
            for r in entries:
                lines.append(f"| {task} | {direction} | {r['identity']:02d} | {r['support']} | {r['recall_delta_mean_pp']:+.3f} | [{r['recall_delta_minimum_pp']:+.3f}, {r['recall_delta_maximum_pp']:+.3f}] | {r['improved_seeds']}/{r['regressed_seeds']} |")
    lines += ["", "## Corrections and regressions", "", "Corrected means reference-wrong/candidate-correct; regressed means reference-correct/candidate-wrong. Both-wrong records may have the same or different incorrect labels. Four mutually exclusive states sum to the record support in every paired comparison.", "",
              "![Paired correction and regression counts](paired_turnover.png)", "", "The plot shows mean counts per seed, individual seed points and net corrections. Counts weight records equally. BA change equals the mean across identities of 100 × (corrected − regressed) / identity support; raw net corrected counts instead recover unweighted accuracy change. These can differ and can even have opposite signs.", "",
              "## Negative fold/seed comparisons", "", "Fold scores are descriptive partitions; do not average these rows to replace pooled-seed BA or analyze nine fold/seed estimates as independent samples.", "",
              "| Task | Fold | Seed | BA change (pp) | Corrected | Regressed |", "|---|---:|---:|---:|---:|---:|"]
    negatives = [r for r in analysis["fold_seed_primary"] if r["ba_delta_pp"] < 0]
    for r in negatives:
        lines.append(f"| {r['task']} | {r['fold']} | {r['seed']} | {r['ba_delta_pp']:+.4f} | {r['corrected']} | {r['regressed']} |")
    lines += ["", "## Frequent remaining confusions", "", "Top five directed class confusions per task, ranked by M11 mean count per seed. Counts/rates are repeated-model summaries over the same cohort, not new unique events. Rates are relative to the true identity's support. All 380 off-diagonal class pairs are retained for each task and seed, including zero counts.", "",
              "| Task | Truth → predicted identity | True support | M10 mean count | M11 mean count | M11 minus M10 mean count |", "|---|---|---:|---:|---:|---:|"]
    for task in TASKS:
        selected = sorted((r for r in analysis["confusion_summary"] if r["task"] == task), key=lambda r:(-r["candidate_count_mean"],r["identity"],r["predicted_identity"]))[:5]
        for r in selected:
            lines.append(f"| {task} | {r['identity']:02d} → {r['predicted_identity']:02d} | {r['support']} | {r['reference_count_mean']:.2f} | {r['candidate_count_mean']:.2f} | {r['count_delta_mean']:+.2f} |")
    lines += ["", "## Interpretation and next controls", "", "Inspect consistently regressing identities and repeated wrong records against the separately saved spectra/fit-quality diagnostics, recording blocks and raw data. Such follow-up may generate hypotheses; it must not justify deleting difficult records from the completed result, changing its primary endpoint or tuning against outer-evaluation outcomes. A separate matched-training-role LDA control can show whether the current error pattern is classifier-specific. Verified source groups and independent labeled sessions are still required for stronger generalization claims.", "",
              "| File | Contents |", "|---|---|", "| identity_by_seed.csv / identity_summary.csv | Recall, support, four paired states, paired pp changes and seed variability |",
              "| seed_totals.csv | Unique pooled evaluation support, BA/accuracy changes and paired counts per seed |",
              "| fold_seed_primary.csv | Every fold/seed primary comparison, including regressions |",
              "| confusion_by_seed.csv / confusion_summary.csv | Complete directed off-diagonal class pairs, counts, rates and seed summaries |",
              "| record_stability.csv | One row per unique record; error/correction/regression frequency across fixed seeds |",
              "| task_summary.csv | Task-level heterogeneity and persistent-error summary |", "| provenance.json | Input result hashes, validator/generator hashes and scope |", "",
              "All outputs are local experiment artifacts. Individual-record stability rows are not intended as a public aggregate table. No confidence intervals, p-values, significance markers or post-hoc training changes are introduced.", ""]
    if not with_plots:
        lines = [line for line in lines if not line.startswith("![")]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output exists; choose a fresh diagnostic directory")
    analysis = analyze(args.artifact_root)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".error-diagnostics-") as temporary:
        output = Path(temporary) / "report"; output.mkdir()
        write_report(analysis, output, with_plots=not args.skip_plots)
        if not args.skip_plots:
            plot(analysis, output)
        output.rename(args.output_dir)
    for r in analysis["task_summary"]:
        print(f"{r['task']}: identity means improved/regressed/unchanged={r['identity_mean_improved']}/{r['identity_mean_regressed']}/{r['identity_mean_unchanged']}; {r['negative_fold_seed_primary']} negative fold/seed comparisons")
    print(f"Error diagnostics: {args.output_dir}")


if __name__ == "__main__":
    main()
