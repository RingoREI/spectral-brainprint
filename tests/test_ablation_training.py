"""Regression checks for preprocessing, factor gates and evaluation isolation."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_runner", ROOT / "scripts/ablation/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AblationInputTests(unittest.TestCase):
    def test_preprocessing_learns_only_training_rows_and_gates_after_scaling(self):
        x = np.tile(np.array([10., 12., 1000.])[:, None], (1, 315))
        preprocessing = runner.fit_preprocessor(x, np.array([0, 1]))
        self.assertTrue(np.all(preprocessing[1].mean_ == 11))
        result = runner.transformed(x, preprocessing, p=0, a=0)
        np.testing.assert_allclose(result[:, :171], np.array([-1., 1., 989.])[:, None] * np.ones((1, 171)))
        self.assertTrue(np.all(result[:, 171:] == 0))
        # Zeroing raw blocks before transformation would instead yield -11.
        enabled = runner.transformed(x, preprocessing, p=1, a=1)
        np.testing.assert_allclose(enabled[2], 989.)

    def test_source_group_overlap_is_rejected_even_for_distinct_record_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.csv"
            rows = []
            for role_index, role in enumerate(runner.ROLES):
                for identity in range(1, 21):
                    rows.append({"task": "EC", "fold": 0, "role": role,
                                 "record_id": role_index * 20 + identity,
                                 "group": f"identity-{identity}"})
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            arrays = {"record_ids": np.arange(1, 61), "tasks": np.array(["EC"] * 60),
                      "y": np.tile(np.arange(1, 21), 3)}
            with self.assertRaisesRegex(ValueError, "group crosses"):
                runner.load_partitions(path, arrays, "EC", [0])

    def test_unaccepted_aperiodic_features_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(path, X=np.ones((20, 315)), y=np.arange(1, 21), record_ids=np.arange(20),
                     tasks=np.array(["EC"] * 20), paths=np.array([str(i) for i in range(20)]),
                     metadata_json=json.dumps({"a36_accepted": False}))
            with self.assertRaisesRegex(ValueError, "a36_accepted"):
                runner.load_features(path)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional deep dependency")
class AblationTrainingTests(unittest.TestCase):
    def test_outer_labels_do_not_change_checkpoint_selection_or_weights(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(34)
            x = rng.normal(size=(100, 315))
            y = np.repeat(np.arange(1, 21), 5)
            ids = np.arange(100)
            roles = np.array(["train", "train", "train", "inner_dev", "outer_eval"] * 20)
            splits = root / "splits.csv"
            with splits.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["task", "fold", "role", "record_id"])
                writer.writeheader()
                writer.writerows({"task": "EC", "fold": 0, "role": role, "record_id": int(key)}
                                 for key, role in zip(ids, roles))
            for run_index in (0, 1):
                labels = y.copy()
                if run_index:
                    labels[roles == "outer_eval"] = np.roll(labels[roles == "outer_eval"], 1)
                features = root / f"features-{run_index}.npz"
                np.savez(features, X=x, y=labels, record_ids=ids, tasks=np.array(["EC"] * 100),
                         paths=np.array([f"synthetic/{i}" for i in ids]),
                         metadata_json=json.dumps({"a36_accepted": True, "scope": "synthetic_test"}))
                with redirect_stdout(io.StringIO()):
                    runner.main(["--features", str(features), "--splits", str(splits),
                                 "--output-dir", str(root / f"run-{run_index}"),
                                 "--pilot", "--fold", "0", "--seeds", "13", "--max-epochs", "2"])
            for variant in runner.VARIANTS:
                name = f"EC-fold0-seed13-{variant}"
                results = [json.loads((root / f"run-{index}/results/result-{name}.json").read_text())
                           for index in (0, 1)]
                self.assertEqual(results[0]["selected_epoch"], results[1]["selected_epoch"])
                self.assertEqual(results[0]["training_curve"], results[1]["training_curve"])
                states = [torch.load(root / f"run-{index}/checkpoints/{name}.pt", weights_only=True)["model"]
                          for index in (0, 1)]
                for key in states[0]:
                    self.assertTrue(torch.equal(states[0][key], states[1][key]))
                self.assertNotEqual([p["truth"] for p in results[0]["predictions"]],
                                    [p["truth"] for p in results[1]["predictions"]])


if __name__ == "__main__":
    unittest.main()
