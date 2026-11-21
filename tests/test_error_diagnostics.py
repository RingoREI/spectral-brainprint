"""Paired accounting and unique-record units for descriptive error diagnostics."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scripts/ablation"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("ablation_error_diagnostics", DIRECTORY / "analyze_errors.py")
ERRORS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ERRORS)


class PairedErrorTests(unittest.TestCase):
    def test_class_balanced_gain_can_differ_in_sign_from_net_corrected_accuracy(self):
        truth = np.array([1, 1, 1, 2])
        old = np.array([2, 2, 1, 2])
        new = np.array([1, 1, 1, 1])
        overall = ERRORS.paired_counts(truth, old, new)
        self.assertEqual(sum(overall[key] for key in ("both_correct", "corrected", "regressed", "both_wrong")), 4)
        self.assertEqual((overall["corrected"], overall["regressed"]), (2, 1))
        self.assertEqual(overall["recall_delta_pp"], 25)
        identity_deltas = [ERRORS.paired_counts(truth[truth == c], old[truth == c], new[truth == c])["recall_delta_pp"] for c in (1, 2)]
        self.assertAlmostEqual(np.mean(identity_deltas), -100 / 6)

    def test_seed_repetitions_are_attributes_not_new_unique_records(self):
        ids = np.array([21, 32, 43])
        truth = np.array([1, 2, 3])
        old = [[2, 2, 1], [2, 2, 1], [2, 2, 3]]
        new = [[1, 1, 1], [1, 2, 1], [1, 2, 1]]
        rows = ERRORS.record_stability(ids, truth, old, new)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["corrected_seeds"], 3)
        self.assertEqual(rows[1]["regressed_seeds"], 1)
        self.assertEqual(rows[2]["candidate_wrong_seeds"], 3)
        self.assertEqual(rows[2]["both_wrong_seeds"], 2)
        with self.assertRaises(ValueError):
            ERRORS.record_stability([21, 21, 43], truth, old, new)


if __name__ == "__main__":
    unittest.main()
