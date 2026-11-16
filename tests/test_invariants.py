import unittest
import numpy as np
from sklearn.covariance import oas
from spectral_brainprint.io.legacy_handoff import align_branch, load_periodic
from spectral_brainprint.evaluation.reproduce import validate_partition
from spectral_brainprint.models.lda import make_lda
from spectral_brainprint.features.spatial_blind import covariance_blocks
from spectral_brainprint.features.power import features_from_psd
from spectral_brainprint.spectra.welch import welch_psd


class Invariants(unittest.TestCase):
    def test_alignment_reordering_and_conflicts(self):
        meta = {"record_ids": np.array([0, 1, 2]), "paths": np.array(["a/a", "b/a", "c/a"])}
        x = np.array([[1., np.nan], [2., 4.], [3., 5.]])
        perm = [2, 0, 1]
        got, _ = align_branch(meta, x[perm], branch_record_ids=meta["record_ids"][perm],
                              branch_paths=meta["paths"][perm])
        np.testing.assert_array_equal(got, x)
        for keys in ([0, 0, 2], [0, 1, 3]):
            with self.assertRaises(ValueError):
                align_branch(meta, x, branch_record_ids=keys)
        with self.assertRaises(ValueError):
            align_branch(meta, x[:2], branch_record_ids=[0, 1])
        with self.assertRaises(ValueError):
            align_branch(meta, x, branch_record_ids=[0, 1, 2], branch_paths=meta["paths"][::-1])
        with self.assertRaises(ValueError):
            align_branch(meta, x, branch_record_ids=[0, 1, 2], base_y=[1, 2, 3], branch_y=[1, 3, 2])
        with self.assertRaises(ValueError):
            align_branch(meta, np.full((3, 2), np.inf), branch_record_ids=[0, 1, 2])

    def test_requires_explicit_source(self):
        with self.assertRaises(ValueError):
            load_periodic()

    def test_closed_set_and_record_leakage(self):
        y = np.array([1, 1, 2, 2])
        validate_partition([0, 2], [1, 3], y)
        for tr, va in [([0, 2], [0, 3]), ([0, 0, 2], [1, 3]), ([0, 1], [2, 3])]:
            with self.assertRaises(ValueError):
                validate_partition(tr, va, y)

    def test_training_only_preprocessing(self):
        x = np.array([[0., np.nan], [1., 2.], [3., 3.], [4., 4.], [5., 5.], [6., 6.]])
        y = np.array([1, 1, 1, 2, 2, 2])
        model = make_lda(2).fit(x, y)
        imputer = model.steps[0][1]
        scaler = model.steps[1][1]
        before = (imputer.statistics_.copy(), scaler.mean_.copy())
        model.predict(np.array([[1e9, np.nan], [-1e9, 1e9]]))
        np.testing.assert_array_equal(imputer.statistics_, before[0])
        np.testing.assert_array_equal(scaler.mean_, before[1])
        np.testing.assert_allclose(imputer.statistics_, np.nanmedian(x, axis=0))

    def test_power_gain_and_oas_reference(self):
        eeg = np.random.default_rng(42).normal(size=(2, 1200, 18))
        config = {"bands": [{"low_hz": a, "high_hz": b} for a, b in [(4, 8), (8, 13), (13, 30)]],
                  "relative_denominator_hz": [1, 40], "numerical_power_floor": 1e-300}
        f, psd = welch_psd(eeg)
        power, _ = features_from_psd(psd, f, config)
        scaled, _ = features_from_psd(psd * 4, f, config)
        self.assertEqual(power.shape, (2, 108))
        np.testing.assert_allclose(scaled[:, :54] - power[:, :54], np.log10(4), atol=1e-12)
        np.testing.assert_allclose(scaled[:, 54:], power[:, 54:], atol=1e-12)
        blocks, cov, _, _ = covariance_blocks(eeg)
        for i in range(2):
            np.testing.assert_allclose(cov[i], oas(eeg[i])[0], atol=1e-12)
        scaled_blocks, _, _, _ = covariance_blocks(eeg * 2)
        np.testing.assert_allclose(scaled_blocks["logcov_trace_upper"], blocks["logcov_trace_upper"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
