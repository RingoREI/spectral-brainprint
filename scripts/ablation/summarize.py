"""Validate paired 2x2 MLP runs and report descriptive feature effects."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.metrics import precision_recall_fscore_support


VARIANTS = {"M00": (0, 0), "M10": (1, 0), "M01": (0, 1), "M11": (1, 1)}
DIMENSIONS = {"M00": 171, "M10": 279, "M01": 207, "M11": 315}
LABELS = np.arange(1, 21)
PAIRS = {
    "A_given_P": ("M10", "M11"),
    "P_without_A": ("M00", "M10"),
    "A_without_P": ("M00", "M01"),
    "P_given_A": ("M01", "M11"),
    "joint_vs_S": ("M00", "M11"),
}
EFFECT_NAMES = (*PAIRS, "M_P", "M_A", "I_PA")


def _score(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a numeric score")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name}: score must lie in [0, 1]")
    return float(value)


def compute_effects(scores):
    """Return all eight predeclared BA contrasts, in percentage points."""
    if set(scores) != set(VARIANTS):
        raise ValueError("Effects require exactly M00, M10, M01 and M11")
    y00, y10, y01, y11 = (_score(scores[name], name) for name in VARIANTS)
    effects = {name: 100 * (scores[new] - scores[reference])
               for name, (reference, new) in PAIRS.items()}
    effects["M_P"] = (effects["P_without_A"] + effects["P_given_A"]) / 2
    effects["M_A"] = (effects["A_without_P"] + effects["A_given_P"]) / 2
    effects["I_PA"] = 100 * (y11 - y10 - y01 + y00)
    return effects


def relative_balanced_error_reduction(candidate, reference):
    """Percent reduction of balanced error; undefined for a perfect reference."""
    candidate = _score(candidate, "candidate")
    reference = _score(reference, "reference")
    return None if reference == 1 else 100 * (candidate - reference) / (1 - reference)


def _metrics(truth, prediction):
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=LABELS,
                                      average="macro", zero_division=0))}


def _integer(value, field, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field}: expected an integer >= {minimum}")
    return value


def _arrays(result, filename):
    rows = result.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{filename}: predictions must be a nonempty list")
    arrays = {}
    for name in ("record_id", "truth", "prediction"):
        values = [row[name] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{filename}: {name} must contain integers")
        arrays[name] = np.asarray(values, dtype=np.int64)
    if np.any(arrays["record_id"] < 0) or len(np.unique(arrays["record_id"])) != len(rows):
        raise ValueError(f"{filename}: duplicate or negative record IDs")
    if any(not np.isin(arrays[name], LABELS).all() for name in ("truth", "prediction")):
        raise ValueError(f"{filename}: labels must be registered identities 1 through 20")
    if len(rows) != result["n_validation"]:
        raise ValueError(f"{filename}: prediction count differs from n_validation")
    if any("probabilities" in row for row in rows):
        if not all("probabilities" in row for row in rows):
            raise ValueError(f"{filename}: probabilities missing on some records")
        probability = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
        if (probability.shape != (len(rows), 20) or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise ValueError(f"{filename}: invalid class probabilities")
        if not np.array_equal(LABELS[np.argmax(probability, axis=1)], arrays["prediction"]):
            raise ValueError(f"{filename}: probabilities disagree with predicted labels")
        arrays["probabilities"] = probability
    return arrays


def _input_hashes(result, filename):
    hashes = result.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{filename}: missing input_sha256")
    for name, value in hashes.items():
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
            raise ValueError(f"{filename}: input hash names must be relative POSIX names")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"{filename}: invalid SHA256")
    return hashes


def _config_signature(config):
    if not isinstance(config, dict):
        raise ValueError("Result config must be an object")
    if config.get("input_dim") != 315:
        raise ValueError("Result config must declare the fixed 315 input slots")
    # These identify repetitions, not different optimization/model settings.
    ignored = {"task", "fold", "folds", "seed", "seeds"}
    return json.dumps({key: value for key, value in config.items() if key not in ignored},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed_statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "minimum": float(values.min()), "maximum": float(values.max())}


def analyze_results(input_dir, *, pilot=False):
    """Recompute scores from matched predictions, pooling folds within each seed."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("result*.json"))
    if not files:
        raise ValueError("No result*.json files found")
    runs, arrays, sources = {}, {}, []
    experiment = hashes = config_signature = scope = None
    environments = {}
    for path in files:
        name = path.relative_to(input_dir).as_posix()
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        variant = result.get("variant")
        if variant not in VARIANTS:
            raise ValueError(f"{name}: invalid variant")
        p, a = VARIANTS[variant]
        if (result.get("p"), result.get("a")) != (p, a):
            raise ValueError(f"{name}: factor settings disagree with variant")
        seed, fold = (_integer(result[key], f"{name}: {key}") for key in ("seed", "fold"))
        if fold not in (0, 1, 2):
            raise ValueError(f"{name}: invalid fold")
        for field in ("n_train", "n_development", "n_validation"):
            _integer(result[field], f"{name}: {field}", minimum=1)
        key = seed, fold, variant
        if key in runs:
            raise ValueError(f"{name}: duplicate seed/fold/variant")
        identity = result["task"], result["protocol"]
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError(f"{name}: task and protocol must be named")
        if experiment is not None and identity != experiment:
            raise ValueError("Results mix different tasks or protocols")
        experiment = identity
        actual_scope = result.get("scope", "development")
        if actual_scope not in ("pilot", "development") or (scope is not None and scope != actual_scope):
            raise ValueError("Results mix incompatible experiment scopes")
        scope = actual_scope
        current_hashes = _input_hashes(result, name)
        if hashes is not None and current_hashes != hashes:
            raise ValueError("Results use different frozen inputs")
        hashes = current_hashes
        current_signature = _config_signature(result["config"])
        if config_signature is not None and current_signature != config_signature:
            raise ValueError("Results use different model or training configurations")
        config_signature = current_signature
        values = _arrays(result, name)
        recomputed = _metrics(values["truth"], values["prediction"])
        for metric, value in recomputed.items():
            saved = _score(result["metrics"][metric], f"{name}: {metric}")
            if not math.isclose(value, saved, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{name}: saved {metric} differs from predictions")
        if "nll" in result["metrics"]:
            nll = result["metrics"]["nll"]
            if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
                raise ValueError(f"{name}: invalid NLL")
            if "probabilities" in values:
                recomputed_nll = float(log_loss(values["truth"], values["probabilities"], labels=LABELS))
                if not math.isclose(recomputed_nll, nll, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"{name}: saved NLL differs from probabilities")
        runs[key], arrays[key] = result, values
        sources.append({"filename": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        environment = result.get("environment", {})
        environments[json.dumps(environment, sort_keys=True)] = environment

    seeds = sorted({key[0] for key in runs})
    folds = sorted({key[1] for key in runs})
    expected = {(seed, fold, variant) for seed in seeds for fold in folds for variant in VARIANTS}
    if set(runs) != expected:
        raise ValueError("Every seed/fold must contain all four variants with the same folds")
    if not pilot and (folds != [0, 1, 2] or len(seeds) < 3 or scope == "pilot"):
        raise ValueError("A full report requires folds 0/1/2, at least three seeds and development scope; use --pilot for a pilot")

    task, protocol = experiment
    reference_seed = seeds[0]
    for seed in seeds:
        seen = set()
        for fold in folds:
            base = arrays[(reference_seed, fold, "M00")]
            reference_counts = tuple(runs[(reference_seed, fold, "M00")][name]
                                     for name in ("n_train", "n_development", "n_validation"))
            for variant in VARIANTS:
                values = arrays[(seed, fold, variant)]
                if any(not np.array_equal(values[name], base[name]) for name in ("record_id", "truth")):
                    raise ValueError("Validation IDs or labels differ across variants or seeds")
                counts = tuple(runs[(seed, fold, variant)][name]
                               for name in ("n_train", "n_development", "n_validation"))
                if counts != reference_counts:
                    raise ValueError("Train/development/validation counts differ across paired runs")
            ids = set(arrays[(seed, fold, "M00")]["record_id"].tolist())
            if seen.intersection(ids):
                raise ValueError("Validation records overlap across folds")
            seen.update(ids)

    cell_rows, effect_rows, identity_rows, paired_rows = [], [], [], []
    execution_rows, history_rows = [], []
    parameter_counts = set()
    for (seed, fold, variant), result in sorted(runs.items()):
        if "parameters" in result:
            parameter_counts.add(_integer(result["parameters"], "parameters", minimum=1))
        if any(field in result for field in ("parameters", "training_seconds", "selected_epoch")):
            execution_rows.append({"task": task, "seed": seed, "fold": fold, "variant": variant,
                                   "n_train": result["n_train"], "n_development": result["n_development"],
                                   **{field: result.get(field) for field in ("parameters", "selected_epoch",
                                      "epochs_completed", "training_seconds", "inference_seconds")}})
        for entry in result.get("training_curve", []):
            history_rows.append({"task": task, "seed": seed, "fold": fold, "variant": variant,
                                 **{field: entry[field] for field in ("epoch", "train_nll", "development_nll",
                                                                    "development_balanced_accuracy")}})
    if len(parameter_counts) > 1:
        raise ValueError("Nominal parameter counts differ across paired runs")
    for seed in seeds:
        truth = np.concatenate([arrays[(seed, fold, "M00")]["truth"] for fold in folds])
        if set(truth.tolist()) != set(LABELS.tolist()):
            raise ValueError("Pooled evaluation must support every registered identity")
        pooled = {}
        scores = {}
        for variant, (p, a) in VARIANTS.items():
            pred = np.concatenate([arrays[(seed, fold, variant)]["prediction"] for fold in folds])
            pooled[variant] = pred
            metrics = _metrics(truth, pred)
            scores[variant] = metrics["balanced_accuracy"]
            nll_values = [runs[(seed, fold, variant)]["metrics"].get("nll") for fold in folds]
            nll = (float(log_loss(truth, np.concatenate([arrays[(seed, fold, variant)]["probabilities"]
                                                        for fold in folds]), labels=LABELS))
                   if all("probabilities" in arrays[(seed, fold, variant)] for fold in folds) else
                   (float(np.average(nll_values, weights=[runs[(seed, fold, variant)]["n_validation"] for fold in folds]))
                    if all(value is not None for value in nll_values) else None))
            correct = int(np.count_nonzero(pred == truth))
            cell_rows.append({"task": task, "seed": seed, "variant": variant, "p": p, "a": a,
                              "effective_features": DIMENSIONS[variant], "input_slots": 315,
                              "n_validation": len(truth), "correct": correct, "errors": len(truth) - correct,
                              **metrics, "nll": nll})
            precision, recall, f1, support = precision_recall_fscore_support(
                truth, pred, labels=LABELS, zero_division=0)
            identity_rows.extend({"task": task, "seed": seed, "variant": variant,
                                  "identity": int(label), "support": int(support[index]),
                                  "precision": float(precision[index]), "recall": float(recall[index]),
                                  "f1": float(f1[index])} for index, label in enumerate(LABELS))
        effects = compute_effects(scores)
        for effect in EFFECT_NAMES:
            pair = PAIRS.get(effect)
            rber = relative_balanced_error_reduction(scores[pair[1]], scores[pair[0]]) if pair else None
            effect_rows.append({"task": task, "seed": seed, "contrast": effect,
                                "reference": pair[0] if pair else "", "comparison": pair[1] if pair else "",
                                "effect_pp": effects[effect], "rber_percent": rber,
                                "rber_status": ("defined" if rber is not None else "undefined_perfect_reference")
                                if pair else "not_applicable"})
        for effect, (reference, candidate) in PAIRS.items():
            ref_correct, new_correct = pooled[reference] == truth, pooled[candidate] == truth
            for label in (None, *LABELS.tolist()):
                mask = np.ones(len(truth), dtype=bool) if label is None else truth == label
                old, new = ref_correct[mask], new_correct[mask]
                accuracy_delta = 100 * float(new.mean() - old.mean())
                paired_rows.append({"task": task, "seed": seed, "contrast": effect,
                                    "scope": "overall" if label is None else "identity",
                                    "identity": "" if label is None else label, "support": int(mask.sum()),
                                    "both_correct": int(np.count_nonzero(old & new)),
                                    "corrected": int(np.count_nonzero(~old & new)),
                                    "regressed": int(np.count_nonzero(old & ~new)),
                                    "both_wrong": int(np.count_nonzero(~old & ~new)),
                                    "accuracy_delta_pp": accuracy_delta,
                                    "recall_or_ba_delta_pp": effects[effect] if label is None else accuracy_delta})

    effect_summary = []
    for effect in EFFECT_NAMES:
        rows = [row for row in effect_rows if row["contrast"] == effect]
        stats = _seed_statistics([row["effect_pp"] for row in rows])
        valid = [row["rber_percent"] for row in rows if row["rber_percent"] is not None]
        rber = _seed_statistics(valid) if len(valid) == len(rows) else None
        effect_summary.append({"task": task, "contrast": effect, "n_seeds": len(seeds),
                               **{f"effect_{name}_pp": value for name, value in stats.items()},
                               "n_rber_defined": len(valid),
                               "rber_mean_percent": rber["mean"] if rber else None,
                               "rber_sd_percent": rber["sd"] if rber else None,
                               "rber_status": "not_applicable" if effect not in PAIRS else
                               ("defined_all_seeds" if rber else "unavailable_undefined_seed"),
                               "ci_lower_pp": None, "ci_upper_pp": None,
                               "ci_status": "not_estimated_provisional_development_data"})
    return {"task": task, "protocol": protocol, "pilot": bool(pilot), "seeds": seeds, "folds": folds,
            "cell_rows": cell_rows, "effect_rows": effect_rows, "effect_summary": effect_summary,
            "identity_rows": identity_rows, "paired_rows": paired_rows,
            "execution_rows": execution_rows, "history_rows": history_rows,
            "provenance": {"task": task, "protocol": protocol, "scope": "pilot" if pilot else "development",
                           "seeds": seeds, "folds": folds, "source_files": sources, "input_sha256": hashes,
                           "model_training_config_sha256": hashlib.sha256(config_signature.encode()).hexdigest(),
                           "environments": list(environments.values()),
                           "aggregation": "Unique evaluation predictions pooled over folds separately for every seed; contrasts then summarized over paired seeds",
                           "n_validation_per_seed": cell_rows[0]["n_validation"],
                           "identity_labels": LABELS.tolist(), "primary_contrast": "A_given_P",
                           "effect_unit": "percentage_points", "confidence_intervals": None,
                           "nominal_parameters": next(iter(parameter_counts)) if parameter_counts else None,
                           "uncertainty_scope": "Descriptive initialization variability conditional on the fixed development data and partitions",
                           "nll_aggregation": "Recomputed from float64 class probabilities when available; otherwise validation-count-weighted saved fold NLL"}}


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def write_tables(analysis, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, key in (("cell_scores_by_seed.csv", "cell_rows"),
                          ("effects_by_seed.csv", "effect_rows"),
                          ("effects_summary.csv", "effect_summary"),
                          ("per_identity.csv", "identity_rows"), ("paired_errors.csv", "paired_rows")):
        _write_csv(output_dir / filename, analysis[key])
    if analysis["execution_rows"]:
        _write_csv(output_dir / "execution_summary.csv", analysis["execution_rows"])
    if analysis["history_rows"]:
        _write_csv(output_dir / "training_curves.csv", analysis["history_rows"])
    (output_dir / "provenance.json").write_text(json.dumps(analysis["provenance"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stage = "Pilot" if analysis["pilot"] else "Development experiment"
    n_records = analysis["cell_rows"][0]["n_validation"]
    lines = [f"# {analysis['task']} 2 × 2 MLP ablation report", "",
             f"**{stage}.** {len(analysis['folds'])} folds, {len(analysis['seeds'])} seeds, 20 registered identities and {n_records:,} unique evaluation records per seed.", "",
             "The fixed model uses 315 input slots [S171, P108, A36]. Conditions change feature availability after training-only preprocessing. This is a same-session, within-task development comparison.", ""]
    if analysis["pilot"]:
        lines += ["**This pilot checks the pipeline. It is not the complete three-fold, multi-seed comparison and does not establish stability or statistical significance.**", ""]
    lines += ["## Pooled scores", "", "| Condition | Input | Mean BA (%) | Seed SD (pp) |", "|---|---|---:|---:|"]
    for variant, label in (("M00", "S"), ("M10", "S+P"), ("M01", "S+A"), ("M11", "S+P+A")):
        stats = _seed_statistics([100 * row["balanced_accuracy"] for row in analysis["cell_rows"] if row["variant"] == variant])
        lines.append(f"| {variant} | {label} | {_number(stats['mean'])} | {_number(stats['sd'])} |")
    lines += ["", "Scores are recomputed from each seed's pooled, nonduplicated predictions. Fold scores are not averaged and seeds are not stacked into additional records.", "",
              "## Paired effect estimates", "", "| Contrast | Mean (pp) | Seed SD (pp) | Minimum | Maximum |", "|---|---:|---:|---:|---:|"]
    for row in analysis["effect_summary"]:
        lines.append(f"| {row['contrast']} | {_number(row['effect_mean_pp'])} | {_number(row['effect_sd_pp'])} | {_number(row['effect_minimum_pp'])} | {_number(row['effect_maximum_pp'])} |")
    lines += ["", "The primary contrast is A_given_P = 100 × (BA_M11 − BA_M10). All other effects, including negative values, are retained. Average main effects and the interaction contrast describe performance on the BA scale, not physiological causation.", "",
              "## Interpretation and files", "", "Seed SD and ranges describe training randomness conditional on these data. No inferential confidence intervals, p-values or significance stars are computed. Filename blocks have not established independent acquisition/source groups; ordinary record bootstrap or treating folds × seeds as independent samples would not resolve that limitation.", "",
              "P108 is band power and includes aperiodic background. Raw EEG also contains both components. The present comparison tests explicit feature representation; it does not isolate pure neural generators or establish cross-session/cross-task transfer.", "",
              "`cell_scores_by_seed.csv` contains cell metrics; `effects_by_seed.csv` and `effects_summary.csv` contain all eight effects. RBER is relative balanced-error reduction, not relative raw error-count reduction. A perfect reference produces an undefined ratio; any undefined seed makes its all-seed RBER summary unavailable. Per-identity precision/recall/F1 and paired correction/regression counts are retained in their own tables.", "",
              "`provenance.json` records source-result hashes, frozen input hashes and environments. Individual records, predictions, checkpoints and fitted preprocessors stay in the local run directory, separate from these aggregate reports. The analysis follows `docs/09_ablation_methods.md`.", ""]
    if analysis["execution_rows"]:
        lines += ["`execution_summary.csv` retains the parameter count, selected epoch and timing per fitted case. When available, `training_curves.csv` contains epoch-level training and inner-development values; outer evaluation is not a model-selection curve.", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(analysis, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
             "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white"}
    output_dir = Path(output_dir)
    context = (f"{analysis['task']} | {'Pilot' if analysis['pilot'] else 'Development'} | "
               f"{len(analysis['folds'])} folds | {len(analysis['seeds'])} seeds | "
               f"n={analysis['cell_rows'][0]['n_validation']:,} per seed")

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white",
                        metadata={"Title": name, "Creator": "Spectral Brainprint"})
        plt.close(fig)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.8, 6.7))
        fig.subplots_adjust(left=0.19, right=0.85, top=0.80, bottom=0.18)
        order = (("M00", "M01"), ("M10", "M11"))
        means = np.array([[np.mean([100 * row["balanced_accuracy"] for row in analysis["cell_rows"] if row["variant"] == variant])
                           for variant in variants] for variants in order])
        picture = ax.imshow(means, vmin=0, vmax=100, cmap="Blues")
        for i, variants in enumerate(order):
            for j, variant in enumerate(variants):
                stats = _seed_statistics([100 * row["balanced_accuracy"] for row in analysis["cell_rows"] if row["variant"] == variant])
                spread = f"Seed SD {_number(stats['sd'], 2)} pp" if stats["sd"] is not None else "Single seed; SD unavailable"
                ax.text(j, i, f"{variant}\n{stats['mean']:.2f}%\n{spread}", ha="center", va="center",
                        color="white" if means[i, j] > 60 else "#20272E", fontsize=11, linespacing=1.8)
        ax.set_xticks([0, 1], ["A absent", "A present"])
        ax.set_yticks([0, 1], ["P absent", "P present"])
        ax.tick_params(length=0, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(picture, ax=ax, fraction=0.047, pad=0.045)
        colorbar.set_label("Pooled balanced accuracy (%)")
        colorbar.outline.set_visible(False)
        fig.suptitle("Feature availability under a fixed MLP", y=0.98, fontsize=14)
        fig.text(0.19, 0.875, context, fontsize=9)
        fig.text(0.19, 0.09, "S171 in every condition; P108 is band power, A36 is aperiodic.", fontsize=9)
        fig.text(0.19, 0.045, "Descriptive seed variation; no confidence intervals.", fontsize=9, color="#505B66")
        save(fig, "ablation_matrix")

        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        fig.subplots_adjust(left=0.25, right=0.96, top=0.82, bottom=0.18)
        labels = ["A | P (primary)", "P without A", "A without P", "P | A", "Joint vs S", "P main effect", "A main effect", "Interaction contrast"]
        all_values = []
        for position, effect in enumerate(EFFECT_NAMES):
            values = [row["effect_pp"] for row in analysis["effect_rows"] if row["contrast"] == effect]
            all_values.extend(values)
            offset = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(values, position + offset, s=30, facecolors="none", edgecolors="#737C87", linewidths=1.2, zorder=3)
            ax.scatter(np.mean(values), position, marker="D", color="#CE772B", edgecolor="white", s=65, linewidths=0.6, zorder=4)
        low, high = min(0, min(all_values)), max(0, max(all_values))
        margin = max(0.25, 0.12 * (high - low))
        ax.set_xlim(low - margin, high + margin)
        ax.axvline(0, color="#336A9E", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Balanced accuracy difference (percentage points)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E5E8EB", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.suptitle("Paired feature increments and interaction", y=0.98, fontsize=14)
        fig.text(0.25, 0.89, context, fontsize=9)
        fig.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#737C87", label="Seed estimate"),
                            Line2D([], [], marker="D", color="none", markerfacecolor="#CE772B", markeredgecolor="white", label="Mean")],
                   ncol=2, loc="upper center", bbox_to_anchor=(0.61, 0.88), frameon=False, fontsize=9)
        fig.text(0.25, 0.065, "Same-session, within-task development; seed points are not independent data samples.", fontsize=8.5, color="#505B66")
        save(fig, "ablation_effects")


def main(argv=None):
    import importlib.metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing reports are preserved")
    parser.add_argument("--pilot", action="store_true", help="Label a partial fold/seed comparison as a pilot")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("--output-dir already exists; choose a fresh directory")
    analysis = analyze_results(args.input_dir, pilot=args.pilot)
    analysis["provenance"]["summary_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    names = ["numpy", "scikit-learn"] + ([] if args.skip_plots else ["matplotlib"])
    analysis["provenance"]["summary_environment"] = {name: importlib.metadata.version(name) for name in names}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".ablation-report-") as temporary:
        prepared = Path(temporary) / "report"
        write_tables(analysis, prepared)
        if not args.skip_plots:
            plot_results(analysis, prepared)
        prepared.rename(destination)
    primary = next(row for row in analysis["effect_summary"] if row["contrast"] == "A_given_P")
    print(f"{analysis['task']}: A_given_P {_number(primary['effect_mean_pp'])} pp; "
          f"seed SD {_number(primary['effect_sd_pp'])} pp (descriptive)")
    print(f"Aggregate report: {destination}")


if __name__ == "__main__":
    main()
