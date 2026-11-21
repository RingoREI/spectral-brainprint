"""Regression tests for the leakage screen on known synthetic overlaps."""
import sys
from pathlib import Path
import unittest

import numpy as np

# Normal location is tests/. A colocated import also supports standalone checks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ablation"))
from source_audit import scan_overlaps


class OverlapTests(unittest.TestCase):
    def test_shift_with_independent_channel_offsets_and_exact_duplicate(self):
        rng = np.random.default_rng(17)
        parent = rng.normal(size=(135, 18))
        first = parent[:100]
        second = parent[35:] + np.arange(18)
        third = second.copy()
        edges = scan_overlaps(np.stack([first, second, third]), [8, 4, 7], min_overlap=65)
        matched = {(r["prefix_record_id"], r["incoming_record_id"],
                    r["incoming_start_in_prefix_samples"], r["n_overlap_samples"]) for r in edges}
        self.assertEqual(matched, {(8, 4, 35, 65), (8, 7, 35, 65), (4, 7, 0, 100)})

    def test_reject_mismatched_channel_and_short_overlap(self):
        rng = np.random.default_rng(19)
        parent = rng.normal(size=(120, 18))
        first = parent[:100]
        second = parent[20:].copy()
        self.assertEqual(scan_overlaps(np.stack([first, second]), [0, 1], min_overlap=81), [])
        second[73, -1] += 1e-5  # Single non-anchor channel invalidates the pair.
        self.assertEqual(scan_overlaps(np.stack([first, second]), [0, 1], min_overlap=75), [])


if __name__ == "__main__":
    unittest.main()
