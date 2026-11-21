"""Rebuild P108 on the frozen A36 Welch PSD from labeled Train EEG."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from fooof import FOOOF
from scipy.signal import get_window, welch

from spectral_brainprint.constants import CHANNELS
from spectral_brainprint.features.power import feature_dictionary, features_from_psd
from spectral_brainprint.spectra.welch import welch_psd


PSD = dict(window="hann_periodic", nperseg=300, noverlap=150, nfft=300,
           average="median", detrend="constant", scaling="density")
FOOOF_SETTINGS = dict(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0,
                      peak_threshold=2, aperiodic_mode="fixed", verbose=False)
TASKS = {"ec": ["EC"], "other": ["EO", "Imagery", "SSVEP"]}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_spectrum(eeg, window):
    if eeg.shape != (1200, 18) or not np.isfinite(eeg).all():
        raise ValueError("Invalid EEG row")
    return welch(eeg, fs=300, window=window, nperseg=300, noverlap=150, nfft=300,
                 detrend="constant", return_onesided=True, scaling="density",
                 axis=0, average="median")


def refit_a36(frequencies, psd):
    params = []
    for spectrum in psd.T:
        model = FOOOF(**FOOOF_SETTINGS)
        model.fit(frequencies, spectrum, [2, 40])
        if not model.has_model:
            raise ValueError("No fit on the shared PSD")
        params.append(model.aperiodic_params_)
    return np.array(params).T.reshape(-1)


def prepare(args, scratch):
    tasks = TASKS[args.group]
    original = args.repo_root / "artifacts/ablation"
    previous = original / ("ec/prepared" if args.group == "ec" else "shared/prepared_other_tasks")
    power_file = args.repo_root / "configs/features/power_v1.json"
    raw_file = args.raw_root / "train_basic.h5"
    index_file = args.raw_root / "unique_index.csv"
    source = {name: digest(previous / name) for name in ("features.npz", "splits.csv")}
    cache = read_json(previous / "extraction_config.json")
    matching = dict(tasks=tasks, psd=PSD, fooof_settings=FOOOF_SETTINGS, fit_range_hz=[2, 40],
                    fs=300, samples=1200, channels=list(CHANNELS), input_view="data_basic",
                    a36_order="offset18_then_exponent18", version="ablation-a36-recovery-v1")
    if any(cache.get(key) != value for key, value in matching.items()):
        raise ValueError("Cached A36 extraction protocol differs from the shared PSD protocol")
    current_env = {name: version(name) for name in ("numpy", "scipy", "fooof", "h5py")}
    if current_env != cache["environment"] or current_env["fooof"] != "1.1.0":
        raise ValueError("Environment differs from the cached extraction")
    if cache["generator_sha256"] != digest(args.repo_root / "scripts/ablation/prepare.py"):
        raise ValueError("Original recovery generator fingerprint differs")
    for label, path in [("0:unique_index.csv", index_file), ("1:train_basic.h5", raw_file)]:
        if digest(path) != cache["source_sha256"][label]:
            raise ValueError(f"Original A36 source fingerprint differs: {path.name}")
    previous_runs = {}
    for task in tasks:
        manifest_path = original / task.lower() / "full/runs/run_manifest.json"
        manifest = read_json(manifest_path)
        if manifest["status"] != "complete" or manifest["input_sha256"] != {
                "features": source["features.npz"], "splits": source["splits.csv"]}:
            raise ValueError(f"Original {task} run differs from cached features/splits")
        previous_runs[task] = digest(manifest_path)
    with np.load(previous / "features.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files if name != "metadata_json"}
        old_meta = json.loads(str(loaded["metadata_json"].item()))
    if any(old_meta[key] != value for key, value in cache.items()) or not old_meta["a36_accepted"]:
        raise ValueError("Original acceptance and feature metadata differ")
    cfg = read_json(power_file)
    names = [row["name"] for row in feature_dictionary(cfg)]
    if old_meta["feature_names"][171:279] != names or cfg["channels"] != list(CHANNELS):
        raise ValueError("Power dictionary/channel contract mismatch")
    x = arrays["X"]
    ids, labels = arrays["record_ids"], arrays["y"]
    n = len(ids)
    with index_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "Train" and r["task"] in tasks]
    index = {int(r["record_id"]): r for r in rows}
    if (len(rows) != n or set(index) != set(ids.tolist()) or len(index) != n
            or len({r["h5_row"] for r in rows}) != n or len(set(arrays["paths"])) != n
            or x.shape != (n, 315)):
        raise ValueError("Original Train cohort is incomplete or ambiguous")

    # One deterministic record per task and identity, selected before evaluating model outcomes.
    checks = {int(min(ids[(arrays["tasks"] == task) & (labels == identity)]))
              for task in tasks for identity in range(1, 21)}
    spotchecks = []
    window = get_window("hann", 300, fftbins=True)
    replacement = np.empty((n, 108), dtype=np.float64)
    invalid_count = 0
    with h5py.File(raw_file, "r") as raw:
        if raw["channel_names"].asstr()[:].tolist() != list(CHANNELS):
            raise ValueError("Raw channel order differs")
        raw_meta = {k: raw[k].asstr()[:] for k in ("path", "session", "task", "subject")}
        rates, done = raw["fs"][:], raw["done"][:]
        for i, rid in enumerate(ids.tolist()):
            r = index[rid]
            j = int(r["h5_row"])
            if (any(raw_meta[k][j] != r[k] for k in raw_meta)
                    or (r["path"], r["session"], r["task"], int(r["identity_label"])) !=
                    (arrays["paths"][i], arrays["sessions"][i], arrays["tasks"][i], labels[i])
                    or rates[j] != 300 or done[j] != 1):
                raise ValueError(f"Record alignment mismatch for {rid}")
            eeg = raw["data_basic"][j]
            freq, psd = extract_spectrum(eeg, window)
            if not np.array_equal(freq, np.arange(151)):
                raise ValueError("Unexpected frequency grid")
            replacement[i], invalid = features_from_psd(psd, freq, cfg)
            invalid_count += int(invalid.sum())
            if rid in checks:
                new_a = refit_a36(freq, psd)
                old_freq, old_psd = welch_psd(eeg)
                old_p, _ = features_from_psd(old_psd, old_freq, cfg)
                np.testing.assert_array_equal(new_a, x[i, 279:])
                np.testing.assert_allclose(old_p, x[i, 171:279], rtol=0, atol=1e-12)
                spotchecks.append(dict(record_id=rid, task=r["task"], truth=int(labels[i]),
                                       a36_max_abs=float(np.max(np.abs(new_a - x[i, 279:]))),
                                       legacy_p108_max_abs=float(np.max(np.abs(old_p - x[i, 171:279])))))
    unchanged_p = int(np.all(replacement == x[:, 171:279], axis=1).sum())
    if unchanged_p == n:
        raise ValueError("Power replacement did not change any records")
    updated = x.copy()
    updated[:, 171:279] = replacement
    arrays["X"] = updated
    meta = dict(old_meta, version="matched-psd-median-1s-v1", source_pipeline=old_meta["version"],
                psd=PSD, shared_psd_for=["P108", "A36"], power_config_sha256=digest(power_file),
                a36_cache_sha256=source["features.npz"], generator_sha256=digest(Path(__file__)),
                previous_generator_sha256=cache["generator_sha256"],
                previous_input_sha256=source, original_run_manifests=previous_runs)
    np.savez_compressed(scratch / "features.npz", **arrays, metadata_json=np.array(json.dumps(meta)))
    with np.load(scratch / "features.npz") as saved:
        for name, arr in arrays.items():
            if name != "X":
                np.testing.assert_array_equal(saved[name], arr)
        new_x = saved["X"]
        np.testing.assert_array_equal(new_x[:, :171], x[:, :171])
        np.testing.assert_array_equal(new_x[:, 279:], x[:, 279:])
        np.testing.assert_array_equal(new_x[:, 171:279], replacement)
    shutil.copyfile(previous / "splits.csv", scratch / "splits.csv")
    output_hashes = {k: digest(scratch / k) for k in ("features.npz", "splits.csv")}
    if output_hashes["splits.csv"] != source["splits.csv"]:
        raise ValueError("Copied split manifest differs")
    audit = dict(version=meta["version"], tasks=tasks, n_records=n, n_columns=315,
                 same_metadata=True, S171_exact=True, cached_A36_exact=True, splits_byte_identical=True,
                 common_welch=PSD, fs=300, grid_hz=1, segments_per_trial=7,
                 source_sha256=cache["source_sha256"], previous_input_sha256=source,
                 output_sha256=output_hashes, original_run_manifests=previous_runs,
                 generator_sha256=meta["generator_sha256"], power_config_sha256=meta["power_config_sha256"],
                 previous_generator_sha256=cache["generator_sha256"], environment=current_env,
                 A36_recovery="cache reused after raw HDF5, recovery configuration, frozen inputs, row alignment and sampled same-spectrum refits",
                 spotcheck_selection="minimum record_id in each task and enrolled identity",
                 refitted_channel_count=len(spotchecks)*18, numerical_checks=spotchecks,
                 n_changed_p108_records=n-unchanged_p, n_floor_flagged_channels=invalid_count)
    (scratch / "acceptance.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"{args.group}: {n} aligned records, {len(spotchecks)*18} exact A refits", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--group", choices=TASKS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Choose a new output directory to preserve previous runs")
    args.output_dir.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        ready = Path(tmp) / "prepared"
        ready.mkdir()
        prepare(args, ready)
        ready.rename(args.output_dir)


if __name__ == "__main__":
    main()
