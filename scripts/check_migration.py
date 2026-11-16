"""Local source-to-package numerical regression. Requires original materials."""
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import h5py
from spectral_brainprint.spectra.welch import welch_psd
from spectral_brainprint.features.power import features_from_psd
from spectral_brainprint.features.spatial_blind import covariance_blocks, blind_blocks


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    root = args.source_root / 'Processed Data/Periodic/periodic_handoff_20261102'
    v1 = root / 'payload/periodic_power_v1'
    v4 = root / 'payload/ssvep_covariance_v4'
    sys.path[:0] = [str(v1 / 'scripts'), str(v4 / 'scripts')]
    old_power = module('legacy_power_regression', v1 / 'scripts/common.py')
    old_spatial = module('legacy_spatial_regression', v4 / 'scripts/features.py')
    config = json.loads((v1 / 'config/power_v1.json').read_text(encoding='utf-8'))
    synthetic = np.random.default_rng(20261112).normal(size=(3, 1200, 18))
    data_dir = args.source_root / 'Processed Data/Brainprint_Processed_20261101/preprocessing_20261029'
    with h5py.File(data_dir / 'train_basic.h5', 'r') as file:
        real = file['data_basic'][:3]
    results = {}
    for label, eeg in [('synthetic', synthetic), ('real_first_three', real)]:
        f0, p0 = old_power.welch_psd(eeg)
        f1, p1 = welch_psd(eeg)
        np.testing.assert_array_equal(f0, f1)
        np.testing.assert_allclose(p0, p1, rtol=1e-12, atol=1e-12)
        a, _ = old_power.features_from_psd(p0, f0, config)
        b, _ = features_from_psd(p1, f1, config)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        errors = {'psd': float(np.max(np.abs(p0-p1))), 'power': float(np.max(np.abs(a-b)))}
        for func_old, func_new in [(old_spatial.covariance_blocks, covariance_blocks),
                                   (old_spatial.blind_blocks, blind_blocks)]:
            expected = func_old(eeg)[0]
            actual = func_new(eeg)[0]
            for key in expected:
                np.testing.assert_allclose(expected[key], actual[key], rtol=1e-12, atol=1e-12)
                errors[key] = float(np.max(np.abs(expected[key]-actual[key])))
        results[label] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump({'passed': True, 'scope': '3 synthetic + 3 real records; same current environment',
                   'maximum_absolute_errors': results}, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
