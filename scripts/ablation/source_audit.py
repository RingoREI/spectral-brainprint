"""Trace labeled clips to the archived acquisition-file audit and screen their split roles."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


TASKS = ("EC", "EO", "Imagery", "SSVEP")
ROLES = ("train", "inner_dev", "outer_eval")
FIELDS = {"channel_names", "data", "fs", "session_id", "task_label", "unit"}
UNKNOWN = "unknown"
PROTOCOL = "within_task_blocked_provisional"
MIN_OVERLAP = 75  # Screen all integer shifts with at least 250 ms of shared waveform.
ATOL = 1e-8  # Original device-native amplitude units, absolute, no relative tolerance.


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def archived_records(database):
    # This delivered database is a closed snapshot. Immutable mode avoids SQLite
    # creating a WAL or SHM next to the team's read-only source material.
    require(not Path(str(database) + "-wal").exists(), "Close/checkpoint the source audit database first")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        for path, raw in db.execute("SELECT path, record FROM records WHERE path GLOB 'Train/*'"):
            row = json.loads(raw)
            require(path == row["path"], f"SQLite index differs from source payload for {path}")
            yield row


def trace_records(raw_root, competition_root):
    import h5py

    index = [r for r in read_csv(raw_root / "unique_index.csv") if r["split"] == "Train"]
    by_path = {r["path"]: r for r in index}
    require(len(by_path) == len(index) == len({r["record_id"] for r in index}), "Duplicate Train path or ID")
    require(sorted(int(r["h5_row"]) for r in index) == list(range(len(index))), "H5 row mapping differs")
    status = {r["path"]: r for r in read_csv(raw_root / "record_status.csv") if r["path"].startswith("Train/")}
    require(set(status) == set(by_path), "Processing records differ from index")
    originals = {}
    local_train = competition_root / "Train"
    if local_train.exists():
        originals = {p.relative_to(competition_root).as_posix(): p for p in local_train.rglob("*.mat")}
    records = {}
    field_sets = Counter()
    h5_path = raw_root / "train_basic.h5"
    with h5py.File(h5_path, "r") as h:
        require(h["data_basic"].shape == (len(index), 1200, 18), "Unexpected data dimensions")
        require(np.array_equal(h["relative_time_seconds"][:], np.arange(1200) / 300), "Relative time mismatch")
        require(np.all(h["fs"][:] == 300) and np.all(h["done"][:] == 1), "Sampling or completion mismatch")
        hmeta = {k: h[k].asstr()[:] for k in ("path", "subject", "session", "task", "sample", "status", "source_fields_sha256")}
        channels = list(h["channel_names"].asstr()[:])
        for a in archived_records(raw_root / "audit.sqlite"):
            path = a["path"]
            r = by_path[path]
            rid = int(r["record_id"])
            i = int(r["h5_row"])
            require(rid not in records, f"Repeated source record {rid}")
            require(all(a[k] == r[k] == hmeta[k][i] for k in ("path", "session", "task", "subject", "sample")),
                    f"Source/index/H5 metadata mismatch at {rid}")
            require(a["identity"] == int(r["identity_label"]) == int(r["subject"].removeprefix("Subject_")),
                    f"Directory identity mismatch at {rid}")
            require(a["task"] == TASKS[int(a["task_dir"].removeprefix("Task")) - 1], f"Task mismatch at {rid}")
            require(a["all_fields_sha256"] == hmeta["source_fields_sha256"][i]
                    and r["data_sha256"] == a["data_sha256"], f"Archived source hash mismatch at {rid}")
            require(a["channels"] == channels and a["fs"] == 300 and a["shape"] == [1200, 18]
                    and a["schema_ok"] and a["metadata_consistent"] and a["nonfinite_count"] == 0
                    and a["read_status"] == r["status"] == "ok", f"Archived schema/status differs at {rid}")
            require(status[path]["status"] == hmeta["status"][i] == "processed"
                    and json.loads(status[path]["notches_hz"]) == [], f"Unexpected preprocessing at {rid}")
            field_sets[tuple(sorted(a["fields"]))] += 1
            require(set(a["fields"]) == FIELDS, "Inspect the new source metadata before declaring fields unknown")
            if path in originals:
                require(sha256(originals[path]) == a["file_sha256"], f"Present MAT bytes changed at {rid}")
            records[rid] = dict(record_id=rid, identity=int(r["identity_label"]), task=r["task"],
                                session=r["session"], session_evidence="archived_MAT_session_id+index+H5",
                                source_clip_mat=path, original_mat_verified_here=int(path in originals),
                                source_mat_file_sha256=a["file_sha256"], source_data_sha256=a["data_sha256"],
                                source_fields_sha256=a["all_fields_sha256"], source_continuous_record=UNKNOWN,
                                source_run=UNKNOWN, source_window_start_sample=UNKNOWN,
                                source_window_stop_sample_exclusive=UNKNOWN, source_clip_parent=UNKNOWN,
                                augmentation_lineage_before_delivery=UNKNOWN,
                                derived_view="train_basic.h5/data_basic", h5_row=i,
                                derived_view_parent_mat=path, transform="per_record_channel_DC_subtraction",
                                clip_relative_start_sample=0, clip_relative_stop_sample_exclusive=1200, fs_hz=300,
                                trial_number=int(r["sample"].removeprefix("trial_")))
    require(len(records) == len(index), "Archived Train coverage incomplete")
    return records, {",".join(k): v for k, v in field_sets.items()}, len(originals)


def verify_splits(records, prepared, assignments):
    all_ids, split_rows = set(), []
    for directory in prepared:
        with np.load(directory / "features.npz", allow_pickle=False) as features:
            for rid, identity, task, session, path in zip(*(features[k] for k in
                    ("record_ids", "y", "tasks", "sessions", "paths")), strict=True):
                rid = int(rid)
                require(rid not in all_ids, f"Duplicated feature record {rid}")
                all_ids.add(rid)
                r = records[rid]
                require((r["identity"], r["task"], r["session"], r["source_clip_mat"]) ==
                        (int(identity), str(task), str(session), str(path)), f"Feature mismatch at {rid}")
        split_rows.extend(read_csv(directory / "splits.csv"))
    require(all_ids == set(records), "Feature cohort differs from source map")
    historical = defaultdict(set)
    blocks = {}
    for r in read_csv(assignments):
        if r["protocol"] != PROTOCOL:
            continue
        rid, fold = int(r["record_id"]), int(r["fold"])
        require(records[rid]["task"] == r["task"], f"Historical task differs at {rid}")
        require(r["role"] in {"train", "validation"} and fold in range(3), "Invalid historical role")
        bucket = historical[r["task"], fold, r["role"]]
        require(rid not in bucket, "Duplicate historical role row")
        bucket.add(rid)
        if r["role"] == "validation":
            require(rid not in blocks, f"Repeated historical evaluation record {rid}")
            blocks[rid] = fold
    require(set(blocks) == all_ids, "Historical evaluation coverage mismatch")
    by_person = defaultdict(list)
    for i, r in records.items():
        by_person[r["task"], r["identity"]].append(i)
    ranks = {}
    for ids in by_person.values():
        ordered = sorted(ids, key=lambda i: records[i]["trial_number"])
        require(len({records[i]["trial_number"] for i in ids}) == len(ids), "Trial number repeats within person/task")
        require([blocks[i] for i in ordered] == sorted(blocks[i] for i in ids), "Noncontiguous filename blocks")
        ranks.update((i, k) for k, i in enumerate(ordered))
    frozen = defaultdict(dict)
    for r in split_rows:
        rid, fold = int(r["record_id"]), int(r["fold"])
        source = records[rid]
        require(r["task"] == source["task"] and fold in range(3) and r["role"] in ROLES, "Invalid frozen role")
        group = f"identity-{source['identity']:02d}|{r['task']}|filename-block-{blocks[rid]}"
        require(r["group"] == group, f"Frozen group mismatch at {rid}")
        bucket = frozen[r["task"], fold]
        require(rid not in bucket, f"Record appears twice in fold {fold} at {rid}")
        bucket[rid] = r["role"]
    summary = []
    for task in TASKS:
        task_ids = {i for i in all_ids if records[i]["task"] == task}
        for fold in range(3):
            roles = frozen[task, fold]
            old_training = historical[task, fold, "train"]
            development = {i for i in old_training if blocks[i] == (fold + 1) % 3}
            dev_ranks = defaultdict(set)
            for i in development:
                dev_ranks[records[i]["identity"]].add(ranks[i])
            expected = {}
            for i in task_ids:
                if blocks[i] == fold:
                    role = "outer_eval"
                elif i not in old_training:
                    role = "excluded_historical_guard"
                elif i in development:
                    role = "inner_dev"
                elif any(ranks[i] + d in dev_ranks[records[i]["identity"]] for d in range(-2, 3)):
                    role = "excluded_inner_guard"
                else:
                    role = "train"
                require((i in old_training) == (role in {"inner_dev", "train", "excluded_inner_guard"}),
                        f"Historical train/evaluation overlap at {i}")
                expected[i] = role
                records[i][f"fold_{fold}_role"] = role
            require(roles == {i: r for i, r in expected.items() if r in ROLES}, "Frozen split differs from reconstruction")
            counts = Counter(expected.values())
            people = {role: {records[i]["identity"] for i, used in roles.items() if used == role} for role in ROLES}
            require(all(s == set(range(1, 21)) for s in people.values()), "Closed-set coverage mismatch")
            summary.append(dict(task=task, fold=fold, n_train=counts["train"], n_inner_dev=counts["inner_dev"],
                                n_outer_eval=counts["outer_eval"], n_excluded_historical=counts["excluded_historical_guard"],
                                n_excluded_inner=counts["excluded_inner_guard"], train_identities=len(people["train"]),
                                dev_identities=len(people["inner_dev"]), eval_identities=len(people["outer_eval"]),
                                eval_identities_missing_from_train=len(people["outer_eval"] - people["train"])))
    for i, r in records.items():
        r.update(filename_block=blocks[i], within_identity_task_rank_zero_based=ranks[i])
    return summary


def scan_overlaps(data, ids, min_overlap=MIN_OVERLAP, atol=ATOL):
    """Find suffix/prefix pairs equal up to a constant per channel, at every integer shift."""
    n, n_samples, _ = data.shape
    # First-channel increments are invariant to each clip's subtracted DC.
    increments = np.empty((n, n_samples - 1), dtype=np.float64)
    for start in range(0, n, 24):
        batch = data[start:start + 24, :, 0]
        increments[start:start + len(batch)] = np.diff(batch, axis=1)
    prefix = increments[:, 0]
    order = np.argsort(prefix)
    keys = prefix[order]
    pairs = []
    for row, line in enumerate(increments):
        anchors = line[:n_samples - min_overlap + 1]
        # Adjacent differences of residuals bounded by atol can differ by 2*atol.
        lo = np.searchsorted(keys, anchors - 2 * atol, side="left")
        hi = np.searchsorted(keys, anchors + 2 * atol, side="right")
        prior = None
        for shift in np.flatnonzero(hi > lo):
            for other in order[lo[shift]:hi[shift]]:
                if row == other or (shift == 0 and row > other):
                    continue
                if np.max(np.abs(line[shift:shift+5] - increments[other, :5])) > 2 * atol:
                    continue
                if prior is None:
                    prior = np.asarray(data[row])
                length = n_samples - shift
                residual = prior[shift:] - np.asarray(data[other])[:length]
                max_dev = float(np.max(np.abs(residual - residual[0])))
                if max_dev <= atol:
                    pairs.append(dict(prefix_record_id=int(ids[row]), incoming_record_id=int(ids[other]),
                                      incoming_start_in_prefix_samples=int(shift), n_overlap_samples=int(length),
                                      max_abs_residual_after_offset=max_dev))
    return pairs


def main():
    import h5py

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True, help="preprocessing_20261029")
    p.add_argument("--competition-root", type=Path, required=True, help="directory containing competition split folders")
    p.add_argument("--package-root", type=Path, required=True, help="periodic_handoff_20261102")
    p.add_argument("--ablation-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    require(args.competition_root.is_dir(), "Competition root unavailable")
    out = args.output_dir
    require(not out.exists() or not any(out.iterdir()), "Choose a new or empty output directory")
    records, field_sets, original_count = trace_records(args.raw_root, args.competition_root)
    prepared = [args.ablation_root / r for r in ("ec/prepared", "shared/prepared_other_tasks")]
    assignments = args.package_root / "payload/ssvep_covariance_v4/reports/case_assignments.csv"
    splits = verify_splits(records, prepared, assignments)
    ids = sorted(records, key=lambda i: records[i]["h5_row"])
    with h5py.File(args.raw_root / "train_basic.h5", "r") as h:
        pairs = scan_overlaps(h["data_basic"], ids)
    for pair in pairs:
        a, b = (records[pair[k]] for k in ("prefix_record_id", "incoming_record_id"))
        pair.update(same_task=a["task"] == b["task"], same_identity=a["identity"] == b["identity"])
        for fold in range(3):
            k = f"fold_{fold}_role"
            pair[f"fold_{fold}_cross_role"] = a["task"] == b["task"] and a[k] in ROLES and b[k] in ROLES and a[k] != b[k]
    collisions = Counter(records[i]["source_data_sha256"] for i in ids)
    cross_role = {str(f): sum(r[f"fold_{f}_cross_role"] for r in pairs) for f in range(3)}
    paths = {name: args.raw_root / name for name in ("audit.sqlite", "unique_index.csv", "train_basic.h5", "record_status.csv")}
    paths["case_assignments.csv"] = assignments
    paths.update({d.name + "/" + f: d / f for d in prepared for f in ("features.npz", "splits.csv")})
    summary = dict(records=len(records), task_counts=dict(Counter(r["task"] for r in records.values())),
                   sessions=dict(Counter(r["session"] for r in records.values())), archived_mat_field_sets=field_sets,
                   local_train_mats_in_folder=original_count,
                   local_train_mat_bytes_verified=sum(r["original_mat_verified_here"] for r in records.values()),
                   train_records_in_duplicate_signal_hash_groups=sum(n for n in collisions.values() if n > 1),
                   unknown_source_fields={k: sum(r[k] == UNKNOWN for r in records.values()) for k in
                       ("source_continuous_record", "source_run", "source_window_start_sample",
                        "source_window_stop_sample_exclusive", "source_clip_parent", "augmentation_lineage_before_delivery")},
                   all_12_splits_reconstructed=True, all_eval_identities_in_train=True,
                   suffix_prefix_screen=dict(min_samples=MIN_OVERLAP, tolerance_device_native=ATOL,
                       shifts=list((0, 1200 - MIN_OVERLAP)), shift_range="inclusive_all_integer_offsets",
                       matching_pairs_and_shifts=len(pairs), cross_role_matches_by_fold=cross_role),
                   sha256={k: sha256(v) for k, v in paths.items()}, generator_sha256=sha256(Path(__file__)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "source_map.csv", [records[i] for i in ids])
    write_csv(out / "split_coverage.csv", splits)
    write_csv(out / "overlap_matches.csv", pairs, ["prefix_record_id", "incoming_record_id",
              "incoming_start_in_prefix_samples", "n_overlap_samples", "max_abs_residual_after_offset",
              "same_task", "same_identity", *[f"fold_{f}_cross_role" for f in range(3)]])
    (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
