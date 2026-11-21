"""Refit P108 and A36 together from each labeled EEG spectrum and compare every column."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import get_window
from threadpoolctl import threadpool_limits

import prepare_matched_psd as common


def write_atomic(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_chunk(raw_file, power_cfg, rows, reference, fingerprint, chunk_id):
    tasks = {}
    failures = []
    window = get_window("hann", 300, fftbins=True)
    with threadpool_limits(limits=1), h5py.File(raw_file, "r") as h5:
        for row, saved in zip(rows, reference, strict=True):
            rid, task, h5_row, path, session, subject = row
            stats = tasks.setdefault(task, dict(records=0, matched=0, max_abs_p108=0., max_abs_a36=0.))
            stats["records"] += 1
            try:
                if (int(h5["fs"][h5_row]) != 300 or int(h5["done"][h5_row]) != 1
                        or any(h5[field].asstr()[h5_row] != expected for field, expected in [
                            ("path", path), ("session", session), ("subject", subject), ("task", task)])):
                    raise ValueError("HDF5 source/index mismatch")
                freq, psd = common.extract_spectrum(h5["data_basic"][h5_row], window)
                power, _ = common.features_from_psd(psd, freq, power_cfg)
                ap = common.refit_a36(freq, psd)
                pdiff = float(np.max(np.abs(power - saved[:108])))
                adiff = float(np.max(np.abs(ap - saved[108:])))
                if not np.isfinite([pdiff, adiff]).all():
                    raise ValueError("Non-finite error")
                stats["max_abs_p108"] = max(stats["max_abs_p108"], pdiff)
                stats["max_abs_a36"] = max(stats["max_abs_a36"], adiff)
                if pdiff != 0 or adiff != 0:
                    raise ValueError(f"Feature mismatch: max P108 {pdiff}, max A36 {adiff}")
                stats["matched"] += 1
            except Exception as err:
                failures.append(dict(record_id=rid, task=task, reason=f"{type(err).__name__}: {err}"))
    return dict(fingerprint=fingerprint, chunk=chunk_id, first_record=rows[0][0],
                last_record=rows[-1][0], tasks=tasks, failures=failures)


def verify(args):
    raw_file = args.raw_root / "train_basic.h5"
    index_file = args.raw_root / "unique_index.csv"
    power_file = args.repo_root / "configs/features/power_v1.json"
    block_files = {"ec": args.ec_prepared, "other": args.other_prepared}
    provenance = dict(raw_h5=common.digest(raw_file), index=common.digest(index_file),
                      power_config=common.digest(power_file), verify_script=common.digest(Path(__file__)),
                      extraction_script=common.digest(Path(common.__file__)),
                      environment={k: common.version(k) for k in ("numpy", "scipy", "fooof", "h5py")},
                      chunk_size=args.chunk_size,
                      features={k: common.digest(v / "features.npz") for k, v in block_files.items()})
    stamp = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    with index_file.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = [r for r in csv.DictReader(handle) if r["split"] == "Train"]
    source = {int(r["record_id"]): r for r in source_rows}
    if len(source) != len(source_rows):
        raise ValueError("Duplicate Train record IDs")
    with h5py.File(raw_file, "r") as h5:
        if h5["channel_names"].asstr()[:].tolist() != list(common.CHANNELS):
            raise ValueError("HDF5 channel order differs")
    records = []
    blocks = []
    for name, folder in block_files.items():
        acceptance = common.read_json(folder / "acceptance.json")
        if (acceptance["source_sha256"]["1:train_basic.h5"] != provenance["raw_h5"]
                or acceptance["source_sha256"]["0:unique_index.csv"] != provenance["index"]
                or acceptance["output_sha256"]["features.npz"] != provenance["features"][name]
                or acceptance["environment"] != provenance["environment"]
                or acceptance["generator_sha256"] != provenance["extraction_script"]
                or acceptance["power_config_sha256"] != provenance["power_config"]):
            raise ValueError("Prepared feature provenance differs from verification inputs")
        with np.load(folder / "features.npz", allow_pickle=False) as z:
            ids, y, tasks, paths, sessions = (z[k] for k in ("record_ids","y","tasks","paths","sessions"))
            blocks.append(z["X"][:, 171:315])
            for rid, label, task, path, session in zip(ids.tolist(), y.tolist(), tasks.tolist(),
                                                       paths.tolist(), sessions.tolist(), strict=True):
                r = source[rid]
                if (path, task, session, label) != (r["path"], r["task"], r["session"], int(r["identity_label"])):
                    raise ValueError(f"Train index/feature row mismatch for {rid}")
                records.append((rid, task, int(r["h5_row"]), path, session, r["subject"]))
    if (len(records) != len(source) or {r[0] for r in records} != set(source)
            or len({r[2] for r in records}) != len(source) or len({r[3] for r in records}) != len(source)):
        raise ValueError("Full coverage/row/path uniqueness check failed")
    reference = np.concatenate(blocks, axis=0)
    if reference.shape != (len(records), 144):
        raise ValueError("Expected 108 power and 36 aperiodic values per record")
    power_cfg = common.read_json(power_file)
    chunks = args.output_dir / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    sources_path = args.output_dir / "sources.json"
    if sources_path.exists() and common.read_json(sources_path) != provenance:
        raise ValueError("Output directory belongs to different inputs")
    write_atomic(sources_path, provenance)
    results = {}
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk_id, start in enumerate(range(0, len(records), args.chunk_size)):
            path = chunks / f"chunk{chunk_id:04d}.json"
            stop = min(start + args.chunk_size, len(records))
            if path.exists():
                old = common.read_json(path)
                if (old["fingerprint"], old["chunk"], old["first_record"], old["last_record"]) != (
                        stamp, chunk_id, records[start][0], records[stop - 1][0]):
                    raise ValueError(f"Invalid resume chunk {chunk_id}")
                results[chunk_id] = old
            else:
                job = pool.submit(run_chunk, raw_file, power_cfg, records[start:stop],
                                  reference[start:stop], stamp, chunk_id)
                futures[job] = path
        total = len(futures) + len(results)
        for job in as_completed(futures):
            result = job.result()
            write_atomic(futures[job], result)
            results[result["chunk"]] = result
            if len(results) % 10 == 0 or len(results) == total:
                print(f"Verified {len(results)}/{total} chunks", flush=True)
    tasks = {}
    failures = []
    for _, result in sorted(results.items()):
        failures.extend(result["failures"])
        for task, part in result["tasks"].items():
            row = tasks.setdefault(task, dict(records=0, matched=0, max_abs_p108=0., max_abs_a36=0.))
            for key in ("records", "matched"):
                row[key] += part[key]
            for key in ("max_abs_p108", "max_abs_a36"):
                row[key] = max(row[key], part[key])
    if {t:r["records"] for t,r in tasks.items()} != dict(Counter(r[1] for r in records)):
        raise ValueError("Verified task counts differ from index")
    all_match = sum(r["matched"] for r in tasks.values()) == len(records) and not failures
    audit = dict(status="passed" if all_match else "failed", records=len(records),
                 matched_records=sum(r["matched"] for r in tasks.values()),
                 fully_matched_channel_fits=18*sum(r["matched"] for r in tasks.values()),
                 source_role="labeled Train", tolerance="exact numerical equality",
                 method="One Welch per record; P108 integration and all 18 FOOOF fits on that same array",
                 per_task=tasks, failure_count=len(failures), failures=failures, provenance=provenance)
    write_atomic(args.output_dir / "full_psd_verification.json", audit)
    if not all_match:
        raise ValueError(f"Shared-PSD numerical verification failed for {len(failures)} records")
    print(f"Verified {len(records)} matched records, {18*len(records)} channel fits", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("repo-root", "raw-root", "ec-prepared", "other-prepared", "output-dir"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=64)
    args = p.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        p.error("workers and chunk-size must be positive")
    verify(args)
