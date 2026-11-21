"""Recover A36 from read-only EEG and freeze matched provisional nested splits."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from spectral_brainprint.constants import CHANNELS
from spectral_brainprint.io.legacy_handoff import load_periodic, read_csv

TASKS = ("EC", "EO", "Imagery", "SSVEP")
PROTOCOL = "within_task_blocked_provisional"
SETTINGS = dict(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0,
                peak_threshold=2, aperiodic_mode="fixed", verbose=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_chunk(job):
    raw_file, output, records, fingerprint = job
    output = Path(output)
    ids = np.array([int(row["record_id"]) for row in records], dtype=np.int64)
    if output.exists():
        with np.load(output, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != fingerprint or not np.array_equal(saved["record_ids"], ids):
                raise ValueError("Cached A36 chunk does not match frozen extraction")
        return output.name
    import h5py
    from fooof import FOOOF
    from scipy.signal import welch, get_window
    values = np.full((len(ids), 36), np.nan)
    quality = np.full((len(ids), 18, 3), np.nan)
    errors = np.full((len(ids), 18), "", dtype="U240")
    with h5py.File(raw_file, "r") as h5:
        for index, row in enumerate(records):
            h5_row = int(row["h5_row"])
            path = h5["path"][h5_row]
            path = path.decode() if isinstance(path, bytes) else str(path)
            if path != row["path"]:
                raise ValueError("Raw HDF5 path/index mismatch")
            eeg = np.asarray(h5["data_basic"][h5_row])
            if eeg.shape != (1200, 18) or not np.isfinite(eeg).all():
                raise ValueError("Invalid raw EEG record")
            frequencies, psd = welch(eeg, fs=300, window=get_window("hann", 300, fftbins=True),
                                     nperseg=300, noverlap=150, nfft=300, detrend="constant",
                                     scaling="density", average="median", axis=0)
            for channel in range(18):
                try:
                    model = FOOOF(**SETTINGS)
                    model.fit(frequencies, psd[:, channel], [2, 40])
                    offset, exponent = model.aperiodic_params_
                    if not model.has_model or not np.isfinite([offset, exponent]).all():
                        raise ValueError("No finite aperiodic fit")
                    values[index, channel] = offset
                    values[index, 18 + channel] = exponent
                    quality[index, channel] = model.r_squared_, model.error_, model.n_peaks_
                except Exception as error:
                    errors[index, channel] = f"{type(error).__name__}: {error}"[:240]
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, record_ids=ids, X=values, quality=quality,
                        errors=errors, fingerprint=np.array(fingerprint))
    temporary.replace(output)
    return output.name


def make_splits(assignments, record_ids, labels, tasks, paths):
    """Recover historical filename blocks; never label them true source Runs."""
    lookup = {int(value): i for i, value in enumerate(record_ids)}
    selected = [row for row in assignments if row["protocol"] == PROTOCOL
                and int(row["record_id"]) in lookup]
    blocks = {}
    for row in selected:
        if row["role"] == "validation":
            rid = int(row["record_id"])
            if rid in blocks:
                raise ValueError("Validation record occurs in multiple historical blocks")
            blocks[rid] = int(row["fold"])
    if set(blocks) != set(lookup):
        raise ValueError("Historical validation blocks do not cover selected records")
    ranks = {}
    for task in sorted(set(tasks.tolist())):
        for identity in range(1, 21):
            rows = np.flatnonzero((tasks == task) & (labels == identity))
            def filename_number(index):
                match = re.search(r"(\d+)$", Path(str(paths[index]).replace("\\", "/")).stem)
                if match is None:
                    raise ValueError("Cannot recover declared filename order")
                return int(match.group(1))
            ordered = sorted(rows, key=filename_number)
            numbers = [filename_number(index) for index in ordered]
            if len(numbers) != len(set(numbers)):
                raise ValueError("Duplicate within-identity/task filename number")
            memberships = [blocks[int(record_ids[index])] for index in ordered]
            if memberships != sorted(memberships):
                raise ValueError("Historical blocks are not contiguous in filename order")
            ranks.update({int(record_ids[index]): rank for rank, index in enumerate(ordered)})
    output, counts = [], []
    for task in sorted(set(tasks.tolist())):
        for fold in range(3):
            case = [row for row in selected if row["task"] == task and int(row["fold"]) == fold]
            outer_train = {int(row["record_id"]) for row in case if row["role"] == "train"}
            evaluation = {int(row["record_id"]) for row in case if row["role"] == "validation"}
            if outer_train & evaluation:
                raise ValueError("Outer partition overlap")
            development = {rid for rid in outer_train if blocks[rid] == (fold + 1) % 3}
            training = {rid for rid in outer_train if blocks[rid] == (fold + 2) % 3}
            before = len(training)
            for identity in range(1, 21):
                dev_ranks = [ranks[rid] for rid in development if labels[lookup[rid]] == identity]
                training.difference_update({rid for rid in training if labels[lookup[rid]] == identity
                                            and any(abs(ranks[rid] - rank) <= 2 for rank in dev_ranks)})
            roles = {"train": training, "inner_dev": development, "outer_eval": evaluation}
            groups_by_role = []
            for role, ids in roles.items():
                if {int(labels[lookup[rid]]) for rid in ids} != set(range(1, 21)):
                    raise ValueError("A partition lacks a registered identity")
                groups = set()
                for rid in sorted(ids):
                    group = f"identity-{labels[lookup[rid]]:02d}|{task}|filename-block-{blocks[rid]}"
                    groups.add(group)
                    output.append(dict(task=task, fold=fold, role=role, record_id=rid, group=group))
                groups_by_role.append(groups)
            if any(a & b for i, a in enumerate(groups_by_role) for b in groups_by_role[i + 1:]):
                raise ValueError("Filename block crosses roles")
            counts.append(dict(task=task, fold=fold, n_train=len(training), n_inner_dev=len(development),
                               n_outer_eval=len(evaluation), extra_inner_guard_exclusions=before-len(training)))
    return output, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["EC"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.tasks)) != len(args.tasks):
        parser.error("Workers must be positive and tasks unique")
    if args.output_dir.exists() and not args.resume:
        parser.error("Output exists; choose a new directory or explicitly resume matching extraction")
    if importlib.metadata.version("fooof") != "1.1.0":
        raise ValueError("Frozen recovery requires fooof 1.1.0")
    s, y, meta = load_periodic("Train", "logcov171", package_root=args.package_root)
    p, yp, mp = load_periodic("Train", "power108", package_root=args.package_root)
    if not np.array_equal(y, yp) or not np.array_equal(meta["record_ids"], mp["record_ids"]):
        raise ValueError("Power/spatial identity mismatch")
    index_path = args.raw_root / "unique_index.csv"
    raw_path = args.raw_root / "train_basic.h5"
    assignment_path = args.package_root / "payload/ssvep_covariance_v4/reports/case_assignments.csv"
    sources = [index_path, raw_path, assignment_path,
               args.package_root / "payload/periodic_power_v1/features/train_power_v1.npz",
               args.package_root / "payload/ssvep_covariance_v4/features/train_blind_covariance_v4.npz",
               args.package_root / "payload/periodic_power_v1/features/feature_dictionary.csv",
               args.package_root / "payload/ssvep_covariance_v4/features/feature_dictionary.csv"]
    selected = np.flatnonzero(np.isin(meta["tasks"], args.tasks))
    index = {int(row["record_id"]): row for row in read_csv(index_path) if row["split"] == "Train"}
    records = [index[int(meta["record_ids"][i])] for i in selected]
    for i, row in zip(selected, records):
        if (row["path"], int(row["identity_label"]), row["task"], row["session"]) != (
                str(meta["paths"][i]), int(y[i]), str(meta["tasks"][i]), str(meta["sessions"][i])):
            raise ValueError("Raw/feature index mismatch")
    config = dict(version="ablation-a36-recovery-v1", tasks=args.tasks, input_view="data_basic",
                  unit="device_native_unverified", channels=CHANNELS, fs=300, samples=1200,
                  psd=dict(window="hann_periodic", nperseg=300, noverlap=150, nfft=300,
                           average="median", detrend="constant", scaling="density"),
                  fit_range_hz=[2, 40], fooof_settings=SETTINGS, a36_order="offset18_then_exponent18",
                  source_sha256={f"{i}:{path.name}": sha256(path) for i, path in enumerate(sources)},
                  generator_sha256=sha256(__file__),
                  environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py", "fooof")})
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = args.output_dir / "extraction_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume configuration differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    chunks = args.output_dir / "chunks"
    chunks.mkdir(exist_ok=True)
    jobs = [(str(raw_path), str(chunks / f"chunk{start // 64:04d}.npz"), records[start:start+64], fingerprint)
            for start in range(0, len(records), 64)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for count, future in enumerate(as_completed([executor.submit(extract_chunk, job) for job in jobs]), 1):
            future.result()
            print(f"A36 chunks {count}/{len(jobs)}", flush=True)
    av, quality, errors, ids = [], [], [], []
    for _, path, _, _ in jobs:
        with np.load(path, allow_pickle=False) as saved:
            av.append(saved["X"]); quality.append(saved["quality"]); errors.append(saved["errors"]); ids.extend(saved["record_ids"].tolist())
    av, quality, errors = np.concatenate(av), np.concatenate(quality), np.concatenate(errors)
    if not np.array_equal(ids, meta["record_ids"][selected]) or np.any(np.all(~np.isfinite(av), axis=0)):
        raise ValueError("A36 alignment or empty column failure")
    arrays = {key: meta[key][selected] for key in ("record_ids", "tasks", "paths", "sessions")}
    splits, counts = make_splits(read_csv(assignment_path), arrays["record_ids"], y[selected], arrays["tasks"], arrays["paths"])
    metadata = dict(config, a36_accepted=True, protocol=PROTOCOL,
                    scope="same-session provisional filename blocks; not verified acquisition Runs",
                    inner_rule="dev block=(outer+1)%3, train block=(outer+2)%3 within old outer train; rank guard2",
                    n_records=len(selected), failed_channel_fits=int(np.count_nonzero(errors != "")),
                    negative_exponents=int(np.count_nonzero(av[:, 18:] < 0)), quality_exclusion=False,
                    feature_names=[*meta["feature_names"].tolist(), *mp["feature_names"].tolist(),
                                   *[f"{parameter}__{channel}" for parameter in ("offset", "exponent") for channel in CHANNELS]],
                    split_counts=counts)
    np.savez_compressed(args.output_dir / "features.npz", X=np.concatenate([s[selected], p[selected], av], axis=1),
                        y=y[selected], **arrays, metadata_json=np.array(json.dumps(metadata)))
    with (args.output_dir / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(splits[0])); writer.writeheader(); writer.writerows(splits)
    with (args.output_dir / "quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["record_id", "channel", "r2", "fit_error", "n_peaks", "status", "error"])
        for i, rid in enumerate(ids):
            for c, channel in enumerate(CHANNELS):
                writer.writerow([rid, channel, *quality[i, c], "fit_failed" if errors[i, c] else "ok", errors[i, c]])
    (args.output_dir / "acceptance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(records=len(selected), failed_channel_fits=metadata["failed_channel_fits"], splits=counts)), flush=True)


if __name__ == "__main__":
    main()
