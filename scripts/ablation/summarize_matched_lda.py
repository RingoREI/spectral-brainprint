"""Validate matched-role LDA controls and compare them with the fixed-budget MLP."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import log_loss

import summarize as mlp_summary

TASKS = ("EC", "EO", "Imagery", "SSVEP")
VARIANTS = mlp_summary.VARIANTS
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll")
DIMENSIONS = {"M00": 171, "M10": 279, "M01": 207, "M11": 315}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_arrays(arrays):
    if "probabilities" not in arrays:
        raise ValueError("Class probabilities are required for the matched control")
    return {**mlp_summary._metrics(arrays["truth"], arrays["prediction"]),
            "nll": float(log_loss(arrays["truth"], arrays["probabilities"], labels=mlp_summary.LABELS))}


def stat(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()))


def analyze(lda_dir, artifact_root):
    lda_dir, artifact_root = Path(lda_dir), Path(artifact_root)
    manifest = read_json(lda_dir / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("LDA run manifest must be complete")
    runs, arrays, sources, configurations = {}, {}, [], []
    for path in sorted(lda_dir.rglob("result*.json")):
        result = read_json(path)
        if "seed" in result:
            raise ValueError("Deterministic LDA controls must not be represented as seed replications")
        task, variant, fold = result["task"], result["variant"], result["fold"]
        if task not in TASKS or variant not in VARIANTS or isinstance(fold, bool) or fold not in (0, 1, 2):
            raise ValueError(f"Unexpected LDA condition in {path.name}")
        if result.get("model") != "LDA" or result.get("scope") != "development":
            raise ValueError("Matched controls must be development-scope LDA results")
        if (result["p"], result["a"]) != VARIANTS[variant]:
            raise ValueError("LDA factor flags differ from the variant")
        for field in ("n_train", "n_development", "n_validation"):
            mlp_summary._integer(result[field], field, minimum=1)
        mlp_summary._input_hashes(result, path.name)
        config = result["config"]
        if config.get("solver") != "lsqr" or config.get("shrinkage") != "auto":
            raise ValueError("Expected the frozen lsqr/auto LDA specification")
        values = mlp_summary._arrays(result, path.name)
        metrics = score_arrays(values)
        for metric in METRICS:
            saved = result["metrics"][metric]
            if isinstance(saved, bool) or not isinstance(saved, (int, float)) or not math.isfinite(saved):
                raise ValueError("Invalid saved LDA metric")
            tolerance = 1e-10 if metric == "nll" else 1e-12
            if not math.isclose(saved, metrics[metric], abs_tol=tolerance, rel_tol=0):
                raise ValueError(f"{path.name}: {metric} differs from the saved predictions")
        key = task, fold, variant
        if key in runs:
            raise ValueError("Duplicate LDA task/fold/variant")
        runs[key], arrays[key] = result, values
        sources.append(dict(filename=path.relative_to(lda_dir).as_posix(), sha256=digest(path)))
        if config not in configurations:
            configurations.append(config)
    expected = {(task, fold, variant) for task in TASKS for fold in (0, 1, 2) for variant in VARIANTS}
    if set(runs) != expected:
        raise ValueError("Require exactly 48 deterministic LDA fits: four tasks, three folds, four variants")

    lda_cells, lda_effects, fold_rows, comparisons, effect_differences = [], [], [], [], []
    identity_rows, error_rows, mlp_provenance = [], [], {}
    all_ids = set()
    for task in TASKS:
        mlp_dir = artifact_root / task.lower() / "full/runs"
        analysis = mlp_summary.analyze_results(mlp_dir)
        if analysis["task"] != task or analysis["folds"] != [0, 1, 2] or analysis["pilot"]:
            raise ValueError("MLP reports must be complete and task-specific")
        if analysis["provenance"]["scope"] != "development":
            raise ValueError("MLP pilot data cannot be a comparator")
        mlp_runs = {}
        mlp_arrays = {}
        for path in sorted(mlp_dir.rglob("result*.json")):
            result = read_json(path)
            key = result["seed"], result["fold"], result["variant"]
            mlp_runs[key] = result
            mlp_arrays[key] = mlp_summary._arrays(result, path.name)
        mlp_provenance[task] = analysis["provenance"]
        seen = set()
        for fold in (0, 1, 2):
            base = arrays[(task, fold, "M00")]
            fold_ids = set(base["record_id"].tolist())
            if seen & fold_ids:
                raise ValueError("LDA evaluation records overlap across folds")
            seen.update(fold_ids)
            for variant in VARIANTS:
                result = runs[(task, fold, variant)]
                values = arrays[(task, fold, variant)]
                for field in ("record_id", "truth"):
                    if not np.array_equal(values[field], base[field]):
                        raise ValueError("LDA conditions use different ordered IDs or labels")
                for seed in analysis["seeds"]:
                    control = mlp_runs[(seed, fold, variant)]
                    mlp_values = mlp_arrays[(seed, fold, variant)]
                    if result["input_sha256"] != control["input_sha256"] or result["protocol"] != control["protocol"]:
                        raise ValueError("LDA and MLP use different frozen inputs or protocols")
                    for field in ("n_train", "n_development", "n_validation"):
                        if result[field] != control[field]:
                            raise ValueError("LDA and MLP role counts differ")
                    for field in ("record_id", "truth"):
                        if not np.array_equal(values[field], mlp_values[field]):
                            raise ValueError("LDA and MLP evaluation IDs or labels differ")
                fold_rows.append(dict(task=task, fold=fold, variant=variant,
                    n_train=result["n_train"], n_development=result["n_development"],
                    n_validation=result["n_validation"], active_features=DIMENSIONS[variant],
                    **score_arrays(values), training_seconds=result.get("training_seconds"),
                    inference_seconds=result.get("inference_seconds")))
        if seen & all_ids:
            raise ValueError("Evaluation record IDs overlap across tasks")
        all_ids.update(seen)
        truth = np.concatenate([arrays[(task, fold, "M00")]["truth"] for fold in range(3)])
        if set(truth) != set(mlp_summary.LABELS):
            raise ValueError("All 20 identities must have pooled evaluation support")
        pooled = {}
        lda_scores = {}
        for variant in VARIANTS:
            values = {field: np.concatenate([arrays[(task, fold, variant)][field] for fold in range(3)])
                      for field in ("record_id", "truth", "prediction", "probabilities")}
            pooled[variant] = values
            metrics = score_arrays(values)
            lda_scores[variant] = metrics
            lda_cells.append(dict(task=task, variant=variant, model="LDA", n_validation=len(truth),
                                  active_features=DIMENSIONS[variant], **metrics))
            for identity in mlp_summary.LABELS:
                mask = truth == identity
                identity_rows.append(dict(task=task, variant=variant, identity=int(identity), support=int(mask.sum()),
                    recall=float(np.mean(values["prediction"][mask] == identity))))
        effects = mlp_summary.compute_effects({v: lda_scores[v]["balanced_accuracy"] for v in VARIANTS})
        for contrast, value in effects.items():
            pair = mlp_summary.PAIRS.get(contrast)
            rber = mlp_summary.relative_balanced_error_reduction(lda_scores[pair[1]]["balanced_accuracy"],
                    lda_scores[pair[0]]["balanced_accuracy"]) if pair else None
            lda_effects.append(dict(task=task, contrast=contrast, effect_pp=value, rber_percent=rber,
                rber_status=("defined" if rber is not None else "undefined_perfect_reference") if pair else "not_applicable"))
        mlp_cells = {(r["seed"], r["variant"]): r for r in analysis["cell_rows"]}
        mlp_effects = {(r["seed"], r["contrast"]): r["effect_pp"] for r in analysis["effect_rows"]}
        for seed in analysis["seeds"]:
            for variant in VARIANTS:
                scores = mlp_cells[(seed, variant)]
                differences = {}
                for metric in METRICS:
                    unit = "delta" if metric == "nll" else "delta_pp"
                    differences[f"{metric}_{unit}"] = (scores[metric] - lda_scores[variant][metric]) * (1 if metric == "nll" else 100)
                comparisons.append(dict(task=task, seed=seed, variant=variant, n_validation=len(truth), **differences))
                mlp_pred = np.concatenate([mlp_arrays[(seed, fold, variant)]["prediction"] for fold in range(3)])
                old, new = pooled[variant]["prediction"] == truth, mlp_pred == truth
                error_rows.append(dict(task=task, seed=seed, variant=variant, reference="LDA", comparison="MLP",
                    n_validation=len(truth), both_correct=int((old & new).sum()), corrected_by_mlp=int((~old & new).sum()),
                    regressed_by_mlp=int((old & ~new).sum()), both_wrong=int((~old & ~new).sum())))
            for contrast, lda_value in effects.items():
                mlp_value = mlp_effects[(seed, contrast)]
                effect_differences.append(dict(task=task, seed=seed, contrast=contrast, lda_effect_pp=lda_value,
                    mlp_effect_pp=mlp_value, mlp_minus_lda_effect_pp=mlp_value - lda_value))

    comparison_summary, effect_summary, matched_scores = [], [], []
    for task in TASKS:
        for variant in VARIANTS:
            selected = [r for r in comparisons if r["task"] == task and r["variant"] == variant]
            output = dict(task=task, variant=variant, n_seeds=len(selected))
            for field in ("accuracy_delta_pp", "balanced_accuracy_delta_pp", "macro_f1_delta_pp", "nll_delta"):
                output.update({f"{field}_{name}": value for name, value in stat([r[field] for r in selected]).items()})
            comparison_summary.append(output)
            lda_cell = next(r for r in lda_cells if r["task"] == task and r["variant"] == variant)
            score_row = dict(task=task, variant=variant, n_validation=lda_cell["n_validation"], n_mlp_seeds=len(selected))
            for metric in METRICS:
                field = "nll_delta" if metric == "nll" else f"{metric}_delta_pp"
                scale = 1 if metric == "nll" else 100
                values = [lda_cell[metric] + r[field] / scale for r in selected]
                score_row.update({f"lda_{metric}": lda_cell[metric],
                                  f"mlp_{metric}_mean": float(np.mean(values)),
                                  f"mlp_{metric}_sd": float(np.std(values, ddof=1))})
            matched_scores.append(score_row)
        for contrast in mlp_summary.EFFECT_NAMES:
            selected = [r for r in effect_differences if r["task"] == task and r["contrast"] == contrast]
            output = dict(task=task, contrast=contrast, n_mlp_seeds=len(selected), lda_effect_pp=selected[0]["lda_effect_pp"])
            for field in ("mlp_effect_pp", "mlp_minus_lda_effect_pp"):
                output.update({f"{field}_{name}": value for name, value in stat([r[field] for r in selected]).items()})
            effect_summary.append(output)
    return dict(lda_cells=lda_cells, lda_effects=lda_effects, fold_rows=fold_rows,
                comparisons=comparisons, comparison_summary=comparison_summary,
                effect_differences=effect_differences, effect_summary=effect_summary,
                matched_scores=matched_scores,
                identity_rows=identity_rows, error_rows=error_rows, mlp_provenance=mlp_provenance,
                provenance=dict(lda_run_manifest_sha256=digest(lda_dir / "run_manifest.json"), lda_source_files=sources,
                    lda_configurations=configurations, mlp_inputs=mlp_provenance, lda_fit_count=len(runs),
                    unique_evaluation_records=len(all_ids), lda_replication="one deterministic fit per task/fold/condition; no seed SD",
                    aggregation="Pool nonduplicated evaluation folds within a task and condition; compare each MLP seed with the same single LDA score",
                    effect_units="BA, accuracy and macro-F1 differences in pp; NLL differences in natural-log units",
                    interpretation="Fixed-budget same-session development; seed SD is descriptive; no confidence intervals or significance tests",
                    generator_sha256=digest(Path(__file__)), shared_summary_sha256=digest(Path(mlp_summary.__file__))))


def write_csv(path, values):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def plot(analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    lda_lookup = {(r["task"], r["variant"]): r for r in analysis["lda_cells"]}
    colors = ("#336A9E", "#CE772B")
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    handles = [Line2D([], [], marker="s", color=colors[0], linestyle="none", label="LDA (deterministic)"),
               Line2D([], [], marker="o", markerfacecolor="none", color=colors[1], linestyle="none", label="MLP seed"),
               Line2D([], [], marker="D", color=colors[1], linestyle="none", label="MLP seed mean")]
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        figure.subplots_adjust(left=.08, right=.97, top=.82, bottom=.16, hspace=.42, wspace=.22)
        for task, axis in zip(TASKS, axes.flat):
            all_scores = []
            for position, variant in enumerate(VARIANTS):
                lda = 100 * lda_lookup[(task, variant)]["balanced_accuracy"]
                selected = [r for r in analysis["comparisons"] if r["task"] == task and r["variant"] == variant]
                mlp = np.array([lda + r["balanced_accuracy_delta_pp"] for r in selected])
                all_scores.extend([lda, *mlp.tolist()])
                axis.scatter(position - .16, lda, marker="s", color=colors[0], s=45, zorder=3)
                axis.scatter(position + .16 + np.linspace(-.04, .04, len(mlp)), mlp, marker="o",
                             facecolors="none", edgecolors=colors[1], s=33, zorder=3)
                axis.scatter(position + .16, mlp.mean(), marker="D", color=colors[1], s=45, zorder=4)
            axis.set_title(f"{task} | n={lda_lookup[(task, 'M00')]['n_validation']:,} per model/seed", loc="left")
            axis.set_xticks(range(4), ["S", "S+P", "S+A", "S+P+A"])
            axis.set_ylabel("Pooled balanced accuracy (%)")
            axis.set_xlim(-.6, 3.6)
            axis.set_ylim(max(0, min(all_scores)-3), min(100, max(all_scores)+2))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E8EB"); axis.set_axisbelow(True)
        figure.suptitle("Matched training roles | LDA and fixed-budget MLP", y=.98, fontsize=14)
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52,.93), ncol=3, frameon=False)
        figure.text(.08,.075,"Three folds pooled per task/condition; identical inner-training records and train-only feature transforms.",fontsize=9)
        figure.text(.08,.025,"MLP points are initialization repeats, not independent data samples. No CIs. Per-panel y-axis ranges differ.",fontsize=9)
        for suffix in ("png", "pdf"): figure.savefig(output/f"matched_model_scores.{suffix}",dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(9,4.8))
        figure.subplots_adjust(left=.15,right=.97,top=.79,bottom=.24)
        all_values = [0.]
        for position, task in enumerate(TASKS):
            selected = [r for r in analysis["effect_differences"] if r["task"] == task and r["contrast"] == "A_given_P"]
            lda = selected[0]["lda_effect_pp"]; mlp = np.array([r["mlp_effect_pp"] for r in selected])
            all_values.extend([lda, *mlp.tolist()])
            axis.scatter(lda, position-.14, marker="s", color=colors[0], s=45, zorder=3)
            axis.scatter(mlp, position+.14+np.linspace(-.04,.04,len(mlp)), facecolors="none",edgecolors=colors[1],s=33,zorder=3)
            axis.scatter(mlp.mean(), position+.14, marker="D", color=colors[1],s=45,zorder=4)
        margin=max(.15,.1*(max(all_values)-min(all_values)))
        axis.set_xlim(min(all_values)-margin,max(all_values)+margin)
        axis.axvline(0,color="#737C87",linestyle="--",linewidth=1)
        axis.set_yticks(range(4),TASKS); axis.invert_yaxis()
        axis.set_xlabel("A given P: balanced-accuracy increment (pp)")
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="x",color="#E5E8EB");axis.set_axisbelow(True)
        figure.suptitle("Incremental A36 benefit under two matched-role classifiers",y=.99,fontsize=13)
        figure.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.93),ncol=3,frameon=False)
        figure.text(.15,.105,"LDA: one pooled deterministic effect. MLP: three paired-seed effects and their mean.",fontsize=9)
        figure.text(.15,.045,"Same-session development; fixed MLP budget; no significance test or independent-session inference.",fontsize=9)
        for suffix in ("png","pdf"):figure.savefig(output/f"matched_primary_effects.{suffix}",dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(figure)


def write_readme(analysis, path):
    lines = ["# Matched-role LDA control report", "", "Development comparison, 19 November 2026. This control uses the same inner-training records, training-fitted transforms and outer evaluation records as the 315-slot MLP ablation. It does not refit LDA on the larger historical outer-training pool.", "",
        "LDA is lsqr with automatic Ledoit–Wolf shrinkage and uniform priors over 20 identities. Each arm uses its active columns (171/279/207/315), after the same train-only median imputation, scaling and float32 input quantization as MLP; LDA arithmetic is performed in float64. Disabled constant-zero columns are removed because their inclusion would change the dimensionality-dependent shrinkage target. Thus the models share available information and preprocessing, not nominal capacity or optimization rules.", "",
        "There are **48 deterministic LDA fits** (four tasks × three folds × four conditions). No fictitious LDA seed repeats or SDs are added. Inner development is retained as a distinct non-fitting role; its diagnostic scores are saved, but it is not merged into fitting data or used to search LDA settings. Scores pool each outer evaluation record once per condition. The MLP comparator retains its predeclared three seeds and 80-epoch budget.", "",
        "## Primary feature increments", "", "A_given_P = 100 × (BA_M11 − BA_M10). Differences between classifier effects are computed within each MLP seed against the same deterministic LDA effect, then summarized.", "",
        "| Task | LDA A given P (pp) | MLP A given P: mean ± seed SD (pp) | MLP minus LDA effect: mean ± seed SD (pp) |", "|---|---:|---:|---:|"]
    for task in TASKS:
        r = next(r for r in analysis["effect_summary"] if r["task"] == task and r["contrast"] == "A_given_P")
        lines.append(f"| {task} | {r['lda_effect_pp']:+.3f} | {r['mlp_effect_pp_mean']:+.3f} ± {r['mlp_effect_pp_sd']:.3f} | {r['mlp_minus_lda_effect_pp_mean']:+.3f} ± {r['mlp_minus_lda_effect_pp_sd']:.3f} |")
    lines += ["", "![Matched model scores](matched_model_scores.png)", "", "![Primary effects](matched_primary_effects.png)", "",
        "## Interpretation", "", "The control separates training-set-size differences from the current classifier comparison. A positive MLP-minus-LDA score difference favors this fixed-budget MLP on the stated records; a negative difference favors this LDA configuration. The effect difference asks whether the incremental A36 benefit differs by classifier. Neither quantity isolates nonlinearity alone: covariance assumptions, regularization, class-prior treatment and optimization differ. MLP optimization may be budget-limited, particularly in EC and EO, so this is not a universal architecture ranking or an equal hyperparameter-search comparison.", "",
        "Seed SD and range describe MLP initialization variability conditional on these data. They are not uncertainty intervals for LDA, acquisition variability or statistical superiority. No p-values, CIs or significance stars are computed. The four tasks share identities; provisional filename groups have not established independent recording Runs or cross-session validity. Per-record paired corrections and regressions remain local.", "",
        "## Files", "", "| File | Contents |", "|---|---|",
        "| lda_scores.csv, lda_fold_scores.csv | Deterministic LDA pooled and fold metrics |",
        "| matched_scores.csv | Four tasks × four conditions: deterministic LDA and MLP seed mean/SD, metrics on their original scales |",
        "| lda_effects.csv | All eight prespecified BA contrasts, plus applicable RBER |",
        "| model_comparisons_by_seed.csv, model_comparisons_summary.csv | MLP minus LDA in each condition: BA/accuracy/macro-F1 pp and NLL units |",
        "| effect_differences_by_seed.csv, effect_differences_summary.csv | All eight MLP feature effects minus deterministic LDA effects |",
        "| lda_identity_recall.csv | Identity-level support and recall for each LDA condition |",
        "| paired_model_errors.csv | MLP corrections/regressions relative to the same-condition LDA, separately per seed |",
        "| provenance.json | All source hashes, frozen inputs and analysis rules |", "",
        "Candidate minus reference is used throughout. Positive score differences are better; negative NLL differences are better. Repeated seeds never multiply the number of unique records. This local control extends the model comparison; it does not change the original MLP estimands, seeds, budget or results.", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lda-dir",type=Path,required=True)
    parser.add_argument("--artifact-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--skip-plots",action="store_true")
    args=parser.parse_args(argv)
    if args.output_dir.exists():parser.error("Output exists; choose a fresh report directory")
    analysis=analyze(args.lda_dir,args.artifact_root)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent,prefix=".matched-lda-report-") as temporary:
        output=Path(temporary)/"report";output.mkdir()
        for filename,key in (("lda_scores.csv","lda_cells"),("lda_fold_scores.csv","fold_rows"),("lda_effects.csv","lda_effects"),("matched_scores.csv","matched_scores"),
            ("model_comparisons_by_seed.csv","comparisons"),("model_comparisons_summary.csv","comparison_summary"),
            ("effect_differences_by_seed.csv","effect_differences"),("effect_differences_summary.csv","effect_summary"),
            ("lda_identity_recall.csv","identity_rows"),("paired_model_errors.csv","error_rows")):
            write_csv(output/filename,analysis[key])
        if not args.skip_plots:plot(analysis,output)
        write_readme(analysis,output/"README.md")
        (output/"provenance.json").write_text(json.dumps(analysis["provenance"],indent=2,allow_nan=False)+"\n",encoding="utf-8")
        output.rename(args.output_dir)
    print(f"Validated 48 LDA fits and all paired MLP predictions; report: {args.output_dir}")


if __name__=="__main__":main()
