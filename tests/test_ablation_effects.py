"""Check factorial contrasts and safeguards against invalid paired reports."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_summary", ROOT / "scripts/ablation/summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class EffectDefinitionTests(unittest.TestCase):
    def test_all_predeclared_effects_and_relative_balanced_error(self):
        values = summary.compute_effects({"M00": .90, "M10": .93, "M01": .92, "M11": .94})
        expected = {"A_given_P": 1, "P_without_A": 3, "A_without_P": 2,
                    "P_given_A": 2, "joint_vs_S": 4, "M_P": 2.5, "M_A": 1.5, "I_PA": -1}
        self.assertEqual(set(values), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(values[name], value)
        self.assertAlmostEqual(summary.relative_balanced_error_reduction(.94, .93), 100 / 7)
        self.assertIsNone(summary.relative_balanced_error_reduction(.9, 1.0))

    def test_missing_or_nonfinite_cells_rejected(self):
        with self.assertRaises(ValueError):
            summary.compute_effects({"M00": .9})
        with self.assertRaises(ValueError):
            summary.compute_effects({"M00": .9, "M10": .93, "M01": .92, "M11": float("nan")})


class PairedReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for seed in (10, 11, 12):
            for fold in (0, 1, 2):
                repeats = fold + 1
                for variant, (p, a) in summary.VARIANTS.items():
                    rows = [{"record_id": 100 * fold + index, "truth": index % 20 + 1,
                             "prediction": index % 20 + 1} for index in range(20 * repeats)]
                    result = {"task": "EC", "protocol": "fixture_blocked_provisional", "scope": "development",
                              "seed": seed, "fold": fold, "variant": variant, "p": p, "a": a,
                              "n_train": 100, "n_development": 20, "n_validation": len(rows),
                              "metrics": summary._metrics([row["truth"] for row in rows],
                                                          [row["prediction"] for row in rows]),
                              "predictions": rows, "input_sha256": {"features": "a" * 64, "splits": "b" * 64},
                              "config": {"input_dim": 315, "learning_rate": .0005},
                              "environment": {"python": "fixture"}}
                    self.path(seed, fold, variant).write_text(json.dumps(result), encoding="utf-8")

    def path(self, seed, fold, variant):
        return self.directory / f"result-EC-fold{fold}-seed{seed}-{variant}.json"

    def change(self, seed, fold, variant, update, *, recompute=False):
        path = self.path(seed, fold, variant)
        result = json.loads(path.read_text(encoding="utf-8"))
        update(result)
        if recompute:
            result["metrics"] = summary._metrics([row["truth"] for row in result["predictions"]],
                                                 [row["prediction"] for row in result["predictions"]])
        path.write_text(json.dumps(result), encoding="utf-8")

    def test_fold_predictions_are_pooled_before_seed_contrasts(self):
        for seed in (10, 11, 12):
            for fold in (1, 2):
                def wrong(result):
                    for row in result["predictions"]:
                        row["prediction"] = row["truth"] % 20 + 1
                self.change(seed, fold, "M11", wrong, recompute=True)
        result = summary.analyze_results(self.directory)
        cell = next(row for row in result["cell_rows"] if row["seed"] == 10 and row["variant"] == "M11")
        self.assertEqual(cell["n_validation"], 120)
        self.assertAlmostEqual(cell["balanced_accuracy"], 1 / 6)
        self.assertNotAlmostEqual(cell["balanced_accuracy"], 1 / 3)
        effect = next(row for row in result["effect_summary"] if row["contrast"] == "A_given_P")
        self.assertAlmostEqual(effect["effect_mean_pp"], -100 * 5 / 6)
        self.assertAlmostEqual(effect["effect_sd_pp"], 0)
        self.assertIsNone(effect["ci_lower_pp"])

    def test_undefined_relative_error_preserved_for_all_seed_summary(self):
        result = summary.analyze_results(self.directory)
        for row in result["effect_summary"]:
            self.assertIsNone(row["rber_mean_percent"])
            self.assertEqual(row["n_rber_defined"], 0)

    def test_same_count_but_reordered_records_rejected(self):
        self.change(10, 0, "M01", lambda result: result["predictions"].reverse())
        with self.assertRaisesRegex(ValueError, "IDs or labels differ"):
            summary.analyze_results(self.directory)

    def test_duplicate_evaluation_records_across_folds_rejected(self):
        for seed in (10, 11, 12):
            for variant in summary.VARIANTS:
                self.change(seed, 1, variant, lambda result: result["predictions"][0].update(record_id=0))
        with self.assertRaisesRegex(ValueError, "overlap"):
            summary.analyze_results(self.directory)

    def test_missing_variant_rejected(self):
        self.path(10, 0, "M10").unlink()
        with self.assertRaisesRegex(ValueError, "all four variants"):
            summary.analyze_results(self.directory)

    def test_saved_score_corruption_rejected(self):
        self.change(10, 0, "M10", lambda result: result["metrics"].update(balanced_accuracy=.5))
        with self.assertRaisesRegex(ValueError, "saved balanced_accuracy differs"):
            summary.analyze_results(self.directory)

    def test_unequal_training_configuration_rejected(self):
        self.change(10, 0, "M10", lambda result: result["config"].update(learning_rate=.1))
        with self.assertRaisesRegex(ValueError, "different model or training configurations"):
            summary.analyze_results(self.directory)

    def test_probabilities_cannot_disagree_with_saved_nll(self):
        def probability_rows(result):
            for row in result["predictions"]:
                probability = [.1 / 19] * 20
                probability[row["truth"] - 1] = .9
                row["probabilities"] = probability
            # Correctness remains perfect, but this NLL does not match p(truth)=.9.
            result["metrics"]["nll"] = 1.0
        self.change(10, 0, "M00", probability_rows)
        with self.assertRaisesRegex(ValueError, "saved NLL differs"):
            summary.analyze_results(self.directory)

    def test_partial_comparison_requires_pilot_label(self):
        for path in self.directory.glob("result*.json"):
            value = json.loads(path.read_text())
            if value["seed"] != 10 or value["fold"] != 0:
                path.unlink()
        with self.assertRaisesRegex(ValueError, "full report requires"):
            summary.analyze_results(self.directory)
        result = summary.analyze_results(self.directory, pilot=True)
        self.assertTrue(result["pilot"])
        self.assertIsNone(result["effect_summary"][0]["effect_sd_pp"])


if __name__ == "__main__":
    unittest.main()
