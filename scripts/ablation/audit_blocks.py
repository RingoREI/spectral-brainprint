"""Describe the original MLP filename-block and persistent-identity differences."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


TASKS = ("EC", "EO", "Imagery", "SSVEP")
ROLES = ("train", "inner_dev", "outer_eval")
SEEDS = (20261102, 20261103, 20261104)
FEATURES = {"S": slice(0, 171), "P": slice(171, 279), "A": slice(279, 315)}
LABELS = set(range(1, 21))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_prepared(folder, output, tables, sources):
    with np.load(folder / "features.npz", allow_pickle=False) as saved:
        x, y, ids, tasks, paths = (saved[k] for k in ("X", "y", "record_ids", "tasks", "paths"))
        channels = json.loads(str(saved["metadata_json"]))["channels"]
    if len(set(ids)) != len(ids) or x.shape != (len(ids), 315) or not np.isfinite(x).all():
        raise ValueError("Expected the original complete, finite 315-column feature matrix")
    lookup = {rid: i for i, rid in enumerate(ids)}
    suffix = np.array([int(re.search(r"(\d+)$", Path(str(p)).stem)[1]) for p in paths])
    splits = read_rows(folder / "splits.csv")
    qc = np.full((len(ids), len(channels)), np.nan)
    with (folder / "quality.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            i, j = lookup[int(row["record_id"])], channels.index(row["channel"])
            if row["status"] != "ok" or np.isfinite(qc[i, j]) or not np.isfinite(float(row["r2"])):
                raise ValueError("Unexpected missing/duplicate/failed original channel fit")
            qc[i, j] = float(row["r2"])
    if not np.isfinite(qc).all():
        raise ValueError("Incomplete channel QC")
    for name in ("features.npz", "splits.csv", "quality.csv"):
        with (folder / name).open("rb") as handle:
            sources[str((folder / name).relative_to(output).as_posix())] = hashlib.file_digest(handle, "sha256").hexdigest()
    for task in np.unique(tasks):
        rank = np.zeros(len(ids), dtype=int)
        for label in LABELS:
            idx = np.flatnonzero((tasks == task) & (y == label))
            if len(np.unique(suffix[idx])) != len(idx):
                raise ValueError("Duplicate within-identity/task filename suffix")
            rank[idx[np.argsort(suffix[idx])]] = np.arange(len(idx))
        evaluated = []
        for fold in range(3):
            roles = {}
            for role in ROLES:
                rows = [r for r in splits if (r["task"], int(r["fold"]), r["role"]) == (task, fold, role)]
                indices = np.array([lookup[int(r["record_id"])] for r in rows])
                offset = {"train": 2, "inner_dev": 1, "outer_eval": 0}[role]
                block = (fold + offset) % 3
                if (len(np.unique(indices)) != len(indices) or set(y[indices]) != LABELS or
                        set(tasks[indices]) != {task} or any(
                            r["group"] != f"identity-{y[i]:02d}|{task}|filename-block-{block}"
                            for i, r in zip(indices, rows))):
                    raise ValueError("Inconsistent role, label or source filename block")
                roles[role] = indices
                counts = Counter(y[indices])
                tables["role_support"].append(dict(task=task, fold=fold, role=role, filename_block=block,
                    records=len(indices), n_identities=len(counts), min_per_identity=min(counts.values()),
                    max_per_identity=max(counts.values()), median_channel_fit_r2=np.median(qc[indices]),
                    negative_exponent_pct=100 * np.mean(x[indices, 297:] < 0)))
            if sum(map(len, roles.values())) != len(set(np.concatenate(list(roles.values())))):
                raise ValueError("Repeated record across roles")
            evaluated.extend(ids[roles["outer_eval"]])
            training = roles["train"]
            # Align feature scales to the observed training rows. Constant columns contribute zero.
            sd = x[training].std(axis=0, ddof=0)
            scale = np.where(sd > 0, sd, 1)
            for role in ("inner_dev", "outer_eval"):
                rows = []
                for label in sorted(LABELS):
                    a, b = training[y[training] == label], roles[role][y[roles[role]] == label]
                    delta = (x[b].mean(axis=0) - x[a].mean(axis=0)) / scale
                    delta[sd == 0] = 0
                    separation = int(np.min(np.abs(rank[a, None] - rank[b])))
                    values = {f"{name}_rms": float(np.sqrt(np.mean(delta[s] ** 2))) for name,s in FEATURES.items()}
                    rows.append(dict(task=task, fold=fold, role=role, identity=label, train_count=len(a),
                        role_count=len(b), min_filename_rank_distance=separation, **values))
                tables["identity_shifts"].extend(rows)
                keys = ("min_filename_rank_distance", "S_rms", "P_rms", "A_rms")
                tables["block_shifts"].append(dict(task=task, fold=fold, role=role, **{
                    f"median_{k}": float(np.median([r[k] for r in rows])) for k in keys}))
        if sorted(evaluated) != sorted(ids[tasks == task]):
            raise ValueError("Incomplete outer coverage")
    return {int(i): (str(t), int(l)) for i,t,l in zip(ids, tasks, y)}


def audit_predictions(root, records, tables, sources):
    expected_stability = {int(r["record_id"]): r for r in read_rows(root / "diagnostics/errors/record_stability.csv")}
    prior = read_rows(root / "diagnostics/errors/identity_summary.csv")
    targets = {(r["task"], int(r["identity"])): r for r in prior if int(r["regressed_seeds"]) == len(SEEDS)}
    if set(records) != set(expected_stability):
        raise ValueError("Stability table differs from original cohort")
    for task in TASKS:
        folder = root / task.lower() / "full/runs/results"
        task_targets = {i for t,i in targets if t == task}
        feature_split = "ec/prepared" if task == "EC" else "shared/prepared_other_tasks"
        assignments = read_rows(root / feature_split / "splits.csv")
        for fold in range(3):
            evaluation = {int(r["record_id"]) for r in assignments if r["task"] == task and
                          int(r["fold"]) == fold and r["role"] == "outer_eval"}
            dev_nll, outer_nll = [], []
            changes = defaultdict(Counter)
            for seed in SEEDS:
                arms = {}
                for arm in ("M00", "M01", "M10", "M11"):
                    path = folder / f"result-{task}-fold{fold}-seed{seed}-{arm}.json"
                    raw = path.read_bytes()
                    sources[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
                    run = json.loads(raw)
                    if (run["task"],run["fold"],run["seed"],run["variant"],run["scope"]) != (task,fold,seed,arm,"development"):
                        raise ValueError("Run header mismatch")
                    pred = {r["record_id"]: (r["truth"], r["prediction"]) for r in run["predictions"]}
                    if (len(pred) != len(run["predictions"]) or set(pred) != evaluation or
                            any(records[rid] != (task, truth) for rid,(truth,_) in pred.items())):
                        raise ValueError("Outer evaluation ID/label mismatch")
                    arms[arm] = pred
                    dev_nll.append(run["best_development_nll"])
                    outer_nll.append(run["metrics"]["nll"])
                for label in sorted(task_targets):
                    subjects = [rid for rid in evaluation if records[rid][1] == label]
                    totals = Counter()
                    wrong = Counter()
                    for rid in subjects:
                        old, new = (arms[a][rid][1] for a in ("M10","M11"))
                        if new == label and old != label:
                            category, other = "corrected", old
                        elif old == label and new != label:
                            category, other = "regressed", new
                        else:
                            category = "both_correct" if old == label else "both_wrong"
                            other = None
                        totals[category] += 1
                        changes[rid][category] += 1
                        if other is not None:
                            wrong[(category, other)] += 1
                    tables["focus_fold_seed"].append(dict(task=task, identity=label, fold=fold, seed=seed,
                        support=len(subjects), **{k: totals[k] for k in ("both_correct","regressed","corrected","both_wrong")},
                        delta_recall_pp=100 * (totals["corrected"] - totals["regressed"]) / len(subjects)))
                    for (direction, other), n in sorted(wrong.items()):
                        tables["focus_transitions"].append(dict(task=task, identity=label, fold=fold, seed=seed,
                            direction=direction, incorrect_identity=other, records=n))
            for rid, states in sorted(changes.items()):
                for k in ("regressed","corrected","both_wrong"):
                    if states[k] != int(expected_stability[rid][k+"_seeds"]):
                        raise ValueError("Disagrees with prior record stability")
                tables["focus_records"].append(dict(task=task, identity=records[rid][1], fold=fold, record_id=rid,
                    **{k+"_seeds": states[k] for k in ("both_correct","regressed","corrected","both_wrong")}))
            tables["fold_losses"].append(dict(task=task, fold=fold, n_fits=len(dev_nll),
                mean_best_dev_nll=np.mean(dev_nll), mean_outer_nll=np.mean(outer_nll)))
    for (task,label), reference in sorted(targets.items()):
        rows = [r for r in tables["focus_fold_seed"] if (r["task"], r["identity"]) == (task,label)]
        counts = [sum(r["support"] for r in rows if r["seed"] == seed) for seed in SEEDS]
        if set(counts) != {int(reference["support"])}:
            raise ValueError("Pooled support mismatch")
        deltas = [100 * sum(r["corrected"] - r["regressed"] for r in rows if r["seed"] == seed)/counts[0]
                  for seed in SEEDS]
        if max(deltas) >= 0 or not np.isclose(np.mean(deltas), float(reference["recall_delta_mean_pp"])):
            raise ValueError("Pooled recall delta mismatch")
        subset = [r for r in tables["focus_records"] if (r["task"],r["identity"]) == (task,label)]
        tables["focus_summary"].append(dict(task=task, identity=label, support=counts[0],
            mean_delta_recall_pp=np.mean(deltas), min_delta_recall_pp=min(deltas), max_delta_recall_pp=max(deltas),
            records_regressed_any=sum(r["regressed_seeds"] > 0 for r in subset),
            records_regressed_all=sum(r["regressed_seeds"] == 3 for r in subset),
            records_corrected_any=sum(r["corrected_seeds"] > 0 for r in subset),
            records_corrected_all=sum(r["corrected_seeds"] == 3 for r in subset)))


def render(tables):
    lines = ["# Filename-block and identity audit", "",
        "This post-hoc audit uses the frozen original MLP experiment and its 11,304 labeled records. Closed-set roles contain all 20 enrolled identities. Each record appears exactly once in outer evaluation across three folds per task. The procedure only reads saved features, QC, labels, roles and model outputs.", "",
        "## Where development was difficult", "",
        "Fold 0 trains on filename block 2 and develops on 1. Fold 1 trains on 0 and develops on 2. Fold 2 trains on 1 and develops on 0. In each task, fold 1 trains and develops on opposite ends of each identity’s filename sequence. The middle block supplies outer evaluation. Filename distances count ordered records and provide no elapsed-time scale.", "",
        "| Task | Mean best dev NLL, f0 / f1 / f2 | Mean outer NLL, f0 / f1 / f2 | Median closest train–dev rank distance, f0 / f1 / f2 |",
        "|---|---|---|---|"]
    def values(rows,key,digits=3):
        return " / ".join(f"{r[key]:.{digits}f}" for r in rows)
    for task in TASKS:
        losses = [r for r in tables["fold_losses"] if r["task"] == task]
        dev = [r for r in tables["block_shifts"] if r["task"] == task and r["role"] == "inner_dev"]
        lines.append(f"| {task} | {values(losses,'mean_best_dev_nll')} | {values(losses,'mean_outer_nll')} | {values(dev,'median_min_filename_rank_distance',1)} |")
    lines += ["", "Losses average the four conditions and three seeds within each fold. Best dev NLL uses the selected inner checkpoint. Outer NLL uses that checkpoint on outer-evaluation records.", "",
        "| Task | S centroid shift, f0 / f1 / f2 | P centroid shift, f0 / f1 / f2 | A centroid shift, f0 / f1 / f2 |",
        "|---|---|---|---|"]
    for task in TASKS:
        dev = [r for r in tables["block_shifts"] if r["task"] == task and r["role"] == "inner_dev"]
        lines.append(f"| {task} | {values(dev,'median_S_rms')} | {values(dev,'median_P_rms')} | {values(dev,'median_A_rms')} |")
    lines += ["", "In the original experiment, fold 1 has the largest development loss and the largest train–dev displacement in each of the 12 task/feature combinations. Fold 0 has the largest outer loss in all four tasks. This distinction matters when investigating which records were difficult.", "",
        "For each identity and feature block, subtract its inner-training centroid from its inner-development centroid, divide each coordinate by its SD across the fold’s inner-training records (constant features contribute zero), and take the root mean square across coordinates. Tables report the median across 20 identities. Different training blocks have different scales. These are descriptive measures of feature displacement. Differences in filename separation, training distributions and record characteristics can coexist. Chronology, acquisition drift and the cause of high development loss remain unverified.", "",
        "Role counts, label coverage, median channel-fit R², negative-exponent rates, and analogous outer shifts appear in the CSV files.", "",
        "## Identities with a recall loss at every seed", "",
        "The comparison adds A36 to S+P (M11 versus M10). Identity selection uses each seed’s pooled, nonoverlapping outer records. Columns headed “any / all” count unique records changing direction under at least one / all three seeds.", "",
        "| Task | Identity | Records | Mean change (pp) | Regressed, any / all | Corrected, any / all | Mean corrected − regressed counts, f0 / f1 / f2 | Most frequent new wrong label among regressions (mean count per seed) |",
        "|---|---:|---:|---:|---|---|---|---|"]
    for r in tables["focus_summary"]:
        task,identity = r["task"],r["identity"]
        rows = [i for i in tables["focus_fold_seed"] if i["task"] == task and i["identity"] == identity]
        net = [sum(i["corrected"]-i["regressed"] for i in rows if i["fold"]==f)/3 for f in range(3)]
        counts = Counter()
        for i in tables["focus_transitions"]:
            if (i["task"],i["identity"],i["direction"]) == (task,identity,"regressed"):
                counts[i["incorrect_identity"]] += i["records"]
        top = ", ".join(str(k) for k in sorted(counts) if counts[k] == max(counts.values()))
        text = " / ".join(f"{v:+.1f}" for v in net)
        lines.append(f"| {task} | {identity} | {r['support']} | {r['mean_delta_recall_pp']:+.3f} | {r['records_regressed_any']} / {r['records_regressed_all']} | {r['records_corrected_any']} / {r['records_corrected_all']} | {text} | {top} ({max(counts.values())/3:.2f}) |")
    lines += ["", "SSVEP identity 19 loses recall under each seed, with most net losses in outer fold 0. The particular records that regressed vary across seeds. Imagery identity 13’s net losses all occur in outer fold 1. Imagery identity 20 sees both corrections and regressions, with its largest net loss in outer fold 2.", "",
        "Net counts are averaged across seeds separately within each fold. A record may change in opposite directions under different seeds, so the two ‘any’ columns may overlap. The transition table keeps each fold and seed separate, including both regressions (new wrong label) and corrections (old wrong label). These comparisons identify repeatable mistakes and localize them to filename blocks. They do not isolate a biological cause or support sample removal.", "",
        "## Reproduction", "", "Run `python scripts/ablation/audit_blocks.py --ablation-root artifacts/ablation --output-dir artifacts/ablation/source_audit/blocks` from the repository root. Full per-record results and input hashes remain local to that output.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    tables, records, sources = defaultdict(list), {}, {}
    for name in ("ec/prepared", "shared/prepared_other_tasks"):
        selected = audit_prepared(root / name, root, tables, sources)
        if records.keys() & selected.keys():
            raise ValueError("Repeated prepared record ID")
        records.update(selected)
    if len(records) != 11304:
        raise ValueError("Expected the full original development cohort")
    audit_predictions(root, records, tables, sources)
    for name in ("diagnostics/errors/identity_summary.csv", "diagnostics/errors/record_stability.csv"):
        with (root / name).open("rb") as handle:
            sources[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for name,rows in tables.items():
        write_rows(args.output_dir / (name + ".csv"), rows)
    (args.output_dir / "README.md").write_text(render(tables), encoding="utf-8")
    manifest = dict(unique_records=len(records), n_identity_task_cells=80, n_model_results=144,
        inputs=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_rows={k:len(v) for k,v in tables.items()})
    (args.output_dir / "validation.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
