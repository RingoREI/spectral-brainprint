"""Check task routing, saved task labels, and matched-record safeguards."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lda_runner", ROOT / "scripts/run_lda_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fake_reproduce(package_root, feature_set, protocol, task, fold):
    scores = {"power108": 0.5, "logcov171": 0.75, "power_logcov279": 1.0}
    return {
        "task": task, "feature_set": feature_set, "protocol": protocol,
        "fold": fold, "n_train": 40, "n_validation": 2,
        "metrics": {"balanced_accuracy": scores[feature_set]},
        "predictions": [
            {"record_id": 2 * fold, "truth": 1, "prediction": 1},
            {"record_id": 2 * fold + 1, "truth": 2, "prediction": 2},
        ],
        "input_sha256": {"frozen.csv": "a" * 64},
        "matches_history_at_1e_12": True,
    }


class LdaRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.package = self.root / "handoff"

    def run_command(self, output, *options):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main([
                "--package-root", str(self.package),
                "--output-dir", str(output), *options,
            ])

    def test_selected_task_reaches_reproduction_and_all_saved_results(self):
        for task in ("EC", "EO", "Imagery", "SSVEP"):
            with self.subTest(task=task):
                output = self.root / task
                with patch.object(runner, "reproduce", side_effect=fake_reproduce) as reproduce:
                    self.assertEqual(self.run_command(output, "--task", task, "--fold", "1"), 0)
                self.assertEqual(reproduce.call_count, 3)
                self.assertEqual([call.args[3:] for call in reproduce.call_args_list],
                                 [(task, 1)] * 3)
                with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                self.assertEqual({row["task"] for row in rows}, {task})
                self.assertEqual({row["fold"] for row in rows}, {"1"})
                self.assertEqual({row["feature_set"] for row in rows},
                                 {"power108", "logcov171", "power_logcov279"})
                for path in output.glob("fold*.json"):
                    result = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual((result["task"], result["fold"]), (task, 1))

    def test_existing_command_still_runs_all_three_ec_folds(self):
        output = self.root / "default"
        with patch.object(runner, "reproduce", side_effect=fake_reproduce) as reproduce:
            self.assertEqual(self.run_command(output), 0)
        self.assertEqual([call.args[3:] for call in reproduce.call_args_list],
                         [("EC", fold) for fold in (0, 1, 2) for _ in range(3)])
        self.assertEqual(len(list(output.glob("fold*.json"))), 9)

    def test_same_count_but_different_validation_order_leaves_no_output(self):
        def reordered(*args):
            result = fake_reproduce(*args)
            if result["feature_set"] == "logcov171":
                result["predictions"].reverse()
            return result

        output = self.root / "rejected"
        with patch.object(runner, "reproduce", side_effect=reordered):
            with self.assertRaisesRegex(ValueError, "Imagery fold 0: validation IDs or labels differ"):
                self.run_command(output, "--task", "Imagery", "--fold", "0")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
