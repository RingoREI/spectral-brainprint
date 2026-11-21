"""Check recovered filename blocks and rank guards without inventing Run labels."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_prepare", ROOT / "scripts/ablation/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(600)
        self.y = np.repeat(np.arange(1, 21), 30)
        self.tasks = np.repeat("EC", 600)
        # Deliberately nonconsecutive trial numbers: guards must use rank.
        self.paths = np.array([f"Train/Subject_{c:02d}/Task1/trial_{10*r+1:03d}.mat"
                               for c in range(1, 21) for r in range(30)])
        self.assignments = []
        for fold in range(3):
            for identity in range(20):
                for rank in range(30):
                    if rank // 10 == fold:
                        role = "validation"
                    elif min(abs(rank - index) for index in range(fold*10, fold*10+10)) > 2:
                        role = "train"
                    else:
                        continue
                    self.assignments.append(dict(protocol=prepare.PROTOCOL, task="EC", fold=fold,
                                                 role=role, record_id=identity*30+rank))

    def test_rank_guard_and_disjoint_block_roles(self):
        rows, counts = prepare.make_splits(self.assignments, self.ids, self.y, self.tasks, self.paths)
        self.assertEqual([row["extra_inner_guard_exclusions"] for row in counts], [40, 0, 40])
        self.assertEqual([row["n_train"] for row in counts], [160, 160, 120])
        for fold in range(3):
            case = [row for row in rows if row["fold"] == fold]
            sets = [{row["group"] for row in case if row["role"] == role}
                    for role in ("train", "inner_dev", "outer_eval")]
            self.assertFalse(any(a & b for i, a in enumerate(sets) for b in sets[i+1:]))
            outer = {row["record_id"] for row in case if row["role"] == "outer_eval"}
            original = {row["record_id"] for row in self.assignments
                        if row["fold"] == fold and row["role"] == "validation"}
            self.assertEqual(outer, original)

    def test_duplicate_outer_membership_rejected(self):
        self.assignments.append(dict(protocol=prepare.PROTOCOL, task="EC", fold=1,
                                     role="validation", record_id=0))
        with self.assertRaisesRegex(ValueError, "multiple historical blocks"):
            prepare.make_splits(self.assignments, self.ids, self.y, self.tasks, self.paths)

    def test_incomplete_registered_training_identity_rejected(self):
        rows = [row for row in self.assignments
                if not (row["fold"] == 0 and row["role"] == "train" and row["record_id"] < 30)]
        with self.assertRaisesRegex(ValueError, "lacks a registered identity"):
            prepare.make_splits(rows, self.ids, self.y, self.tasks, self.paths)


if __name__ == "__main__":
    unittest.main()
