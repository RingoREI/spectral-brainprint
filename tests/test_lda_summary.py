"""Regression checks for pooled metrics and matched validation records."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lda_summary", ROOT / "scripts/lda/summarize_lda_results.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class LdaSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for fold, repeats in [(0, 1), (1, 2)]:
            for feature in summary.FEATURE_SETS:
                rows = []
                for index in range(20 * repeats):
                    truth = index % 20 + 1
                    prediction = truth
                    if fold == 1 and feature == "power_logcov279" and index >= 20:
                        prediction = truth % 20 + 1
                    rows.append({"record_id": 20 * fold + index, "truth": truth, "prediction": prediction})
                result = {"feature_set": feature, "task": "EC", "protocol": "test_blocked",
                          "fold": fold, "n_train": 40, "n_validation": len(rows),
                          "metrics": summary._metrics([r["truth"] for r in rows],
                                                      [r["prediction"] for r in rows]),
                          "predictions": rows, "input_sha256": {"frozen.csv": "a" * 64},
                          "matches_history_at_1e_12": True, "environment": {"python": "test"}}
                (self.root / f"fold{fold}-{feature}.json").write_text(json.dumps(result), encoding="utf-8")

    def change(self, name, update):
        path = self.root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        update(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_pooled_score_uses_predictions_instead_of_fold_mean(self):
        result = summary.analyze_results(self.root)
        combined = next(row for row in result["summary_rows"] if row["feature_set"] == "power_logcov279")
        self.assertEqual(combined["n_validation"], 60)
        self.assertAlmostEqual(combined["balanced_accuracy"], 2 / 3)
        self.assertNotAlmostEqual(combined["balanced_accuracy"], 0.75)
        self.assertEqual(result["transitions"]["regressed"], 20)

    def test_duplicate_validation_records_across_folds_rejected(self):
        for feature in summary.FEATURE_SETS:
            def overlap(value):
                value["predictions"][0]["record_id"] = 0
            self.change(f"fold1-{feature}.json", overlap)
        with self.assertRaisesRegex(ValueError, "overlap"):
            summary.analyze_results(self.root)

    def test_reordered_feature_rows_rejected(self):
        self.change("fold0-power108.json", lambda value: value["predictions"].reverse())
        with self.assertRaisesRegex(ValueError, "IDs or labels differ"):
            summary.analyze_results(self.root)

    def test_incomplete_fold_rejected(self):
        (self.root / "fold0-power108.json").unlink()
        with self.assertRaisesRegex(ValueError, "three feature sets"):
            summary.analyze_results(self.root)

    def test_saved_metric_corruption_rejected(self):
        def corrupt(value):
            value["metrics"]["accuracy"] = 0.5
        self.change("fold0-power108.json", corrupt)
        with self.assertRaisesRegex(ValueError, "saved accuracy differs"):
            summary.analyze_results(self.root)


if __name__ == "__main__":
    unittest.main()
