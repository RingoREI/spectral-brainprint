"""Combine the four independently validated task reports without pooling records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

TASKS = ("EC", "EO", "Imagery", "SSVEP")
VARIANTS = ("M00", "M10", "M01", "M11")
PAIRS = {"A_given_P": ("M10", "M11"), "P_without_A": ("M00", "M10"),
         "A_without_P": ("M00", "M01"), "P_given_A": ("M01", "M11"),
         "joint_vs_S": ("M00", "M11")}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output exists; choose a fresh overview directory")
    cells, effects, primary, sources, secondary = [], [], {}, {}, []
    for task in TASKS:
        folder = args.artifact_root / task.lower() / "full/report"
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        if provenance["task"] != task or provenance["scope"] != "development" or provenance["folds"] != [0, 1, 2]:
            raise ValueError(f"{task}: not a complete development report")
        rows = read_csv(folder / "cell_scores_by_seed.csv")
        seeds = sorted({int(row["seed"]) for row in rows})
        if len(seeds) < 3 or len(rows) != 4 * len(seeds):
            raise ValueError("Expected four conditions per seed")
        for variant in VARIANTS:
            selected = [row for row in rows if row["variant"] == variant]
            if sorted(int(row["seed"]) for row in selected) != seeds:
                raise ValueError("Cell seeds do not align")
            supports = {int(row["n_validation"]) for row in selected}
            if len(supports) != 1:
                raise ValueError("Validation support changed across seeds")
            scores = np.array([float(row["balanced_accuracy"]) for row in selected])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Invalid BA")
            cells.append(dict(task=task, variant=variant, n_seeds=len(seeds), n_validation=supports.pop(),
                              balanced_accuracy_mean=float(scores.mean()), balanced_accuracy_sd=float(scores.std(ddof=1))))
        summary = read_csv(folder / "effects_summary.csv")
        if len(summary) != 8 or any(row["task"] != task for row in summary):
            raise ValueError("Effect table is incomplete")
        effects.extend(summary)
        values = [row for row in read_csv(folder / "effects_by_seed.csv") if row["contrast"] == "A_given_P"]
        if sorted(int(row["seed"]) for row in values) != seeds:
            raise ValueError("Primary contrast seeds do not align")
        primary[task] = np.array([float(row["effect_pp"]) for row in values])
        row = next(row for row in summary if row["contrast"] == "A_given_P")
        if not np.isclose(primary[task].mean(), float(row["effect_mean_pp"]), atol=1e-12, rtol=0):
            raise ValueError("Primary mean differs from per-seed effects")
        score_lookup = {(int(item["seed"]), item["variant"]): item for item in rows}
        for seed in seeds:
            for contrast, (reference, candidate) in PAIRS.items():
                old, new = score_lookup[(seed, reference)], score_lookup[(seed, candidate)]
                changes = {}
                for metric in ("balanced_accuracy", "accuracy", "macro_f1", "nll"):
                    before, after = float(old[metric]), float(new[metric])
                    if not np.isfinite([before, after]).all() or min(before, after) < 0:
                        raise ValueError("Invalid secondary metric")
                    if metric != "nll" and max(before, after) > 1:
                        raise ValueError("Bounded metric exceeds 1")
                    field = "nll_delta" if metric == "nll" else f"{metric}_delta_pp"
                    changes[field] = (after - before) * (1 if metric == "nll" else 100)
                secondary.append(dict(task=task, seed=seed, contrast=contrast,
                                      reference=reference, comparison=candidate, **changes))
        for path in folder.glob("*.csv"):
            sources[str(path.relative_to(args.artifact_root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[f"{task.lower()}/full/report/provenance.json"] = hashlib.sha256((folder / "provenance.json").read_bytes()).hexdigest()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".ablation-overview-") as temporary:
        output = Path(temporary) / "report"; output.mkdir()
        write_csv(output / "metrics_by_task.csv", cells)
        write_csv(output / "effects_by_task.csv", effects)
        write_csv(output / "secondary_effects_by_seed.csv", secondary)
        secondary_summary = []
        for task in TASKS:
            for contrast in PAIRS:
                selected = [row for row in secondary if row["task"] == task and row["contrast"] == contrast]
                summary = dict(task=task, contrast=contrast, n_seeds=len(selected))
                for metric in ("balanced_accuracy_delta_pp", "accuracy_delta_pp", "macro_f1_delta_pp", "nll_delta"):
                    values = np.array([row[metric] for row in selected])
                    summary[f"{metric}_mean"] = float(values.mean())
                    summary[f"{metric}_sd"] = float(values.std(ddof=1))
                secondary_summary.append(summary)
        write_csv(output / "secondary_effects_summary.csv", secondary_summary)
        lookup = {(row["task"], row["variant"]): row for row in cells}
        style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
        with plt.rc_context(style):
            figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
            figure.subplots_adjust(left=.13, right=.97, top=.79, bottom=.20, wspace=.43)
            means = np.array([[100*lookup[(task, variant)]["balanced_accuracy_mean"] for variant in VARIANTS] for task in TASKS])
            axes[0].imshow(means, cmap="Blues", vmin=0, vmax=100, aspect="auto")
            for i, task in enumerate(TASKS):
                for j, variant in enumerate(VARIANTS):
                    sd = 100*lookup[(task, variant)]["balanced_accuracy_sd"]
                    axes[0].text(j, i, f"{means[i,j]:.2f}\nSD {sd:.2f}", ha="center", va="center",
                                 fontsize=10, color="white" if means[i,j] > 65 else "#20272E")
            axes[0].set_xticks(range(4), ["S", "S+P", "S+A", "S+P+A"])
            axes[0].set_yticks(range(4), [f"{task}\n(n={lookup[(task,'M00')]['n_validation']:,})" for task in TASKS])
            axes[0].set_title("A  Mean balanced accuracy (%)", loc="left", pad=14)
            axes[0].tick_params(length=0, pad=9)
            for spine in axes[0].spines.values(): spine.set_visible(False)
            all_values = []
            for position, task in enumerate(TASKS):
                values = primary[task]; all_values.extend(values.tolist())
                axes[1].scatter(values, position + np.linspace(-.12,.12,len(values)), facecolors="none", edgecolors="#737C87", s=35)
                axes[1].scatter(values.mean(), position, marker="D", color="#CE772B", edgecolor="white", s=65, zorder=4)
            axes[1].axvline(0, color="#336A9E", linestyle="--", linewidth=1)
            margin = max(.2, .15*(max(all_values)-min(all_values)))
            axes[1].set_xlim(min(0,min(all_values))-margin, max(0,max(all_values))+margin)
            axes[1].set_yticks(range(4), TASKS); axes[1].invert_yaxis()
            axes[1].set_xlabel("A given P: BA increment (pp)")
            axes[1].set_title("B  Primary paired contrast", loc="left", pad=14)
            axes[1].spines[["top","right"]].set_visible(False)
            axes[1].grid(axis="x", color="#E5E8EB"); axes[1].set_axisbelow(True)
            figure.suptitle("Fixed-slot MLP ablation | Four task conditions", y=.98, fontsize=14)
            figure.text(.13,.10,"Same-session, within-task development; 3 folds pooled per seed; 3 seeds; no confidence intervals.",fontsize=9)
            figure.text(.13,.045,"Panel A: seed mean and SD. Panel B: hollow seed estimates and orange mean. Training counts differ by task.",fontsize=8.5)
            for suffix in ("png","pdf"): figure.savefig(output/f"ablation_overview.{suffix}",dpi=300,bbox_inches="tight",facecolor="white")
            plt.close(figure)
        provenance = dict(source_sha256=sources, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          aggregation="Separate per-task summaries; no pooling of task records", significance="not assessed")
        (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
        lines = ["# Four-task fixed-slot MLP ablation", "", "Descriptive development results; each seed pools three outer folds.", "",
                 "| Task | S BA | S+P BA | S+A BA | S+P+A BA | A given P: mean ± seed SD (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for task in TASKS:
            values = [100*lookup[(task,variant)]["balanced_accuracy_mean"] for variant in VARIANTS]
            lines.append("| "+task+" | "+" | ".join(f"{v:.2f}%" for v in values)+f" | {primary[task].mean():+.3f} ± {primary[task].std(ddof=1):.3f} |")
        lines += ["", "![Task conditions and primary increments](ablation_overview.png)", "",
                  "`secondary_effects_by_seed.csv` and `secondary_effects_summary.csv` retain the five matched simple/joint contrasts in accuracy and macro-F1 pp and NLL units. Candidate minus reference is used throughout; a negative NLL change is better. All seed SDs are descriptive, not confidence intervals.", "",
                  "These scores use inner-training records only and cannot be compared directly with historical LDA models fitted on the larger outer-training pool. Source groups remain provisional filename blocks. No test of cross-session transfer or statistical significance is made."]
        (output/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        output.rename(args.output_dir)


if __name__ == "__main__":
    main()
