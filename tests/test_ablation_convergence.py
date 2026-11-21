"""Guard against outcome leakage and misleading diagnostics on invalid curves."""
import copy
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "scripts/ablation/diagnose_convergence.py"
spec = importlib.util.spec_from_file_location("convergence_diagnostics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    return dict(task="EC", variant="M00", fold=0, seed=0, n_train=100, n_development=50,
                config={"max_epochs": 80}, epochs_completed=3, selected_epoch=2,
                best_development_nll=.8,
                training_curve=[dict(epoch=i + 1, train_nll=train, development_nll=dev)
                                for i, (train, dev) in enumerate(((.9, 1.), (.7, .8), (.6, .9)))])


class ConvergenceDiagnosticsTests(unittest.TestCase):
    def test_outer_outcomes_are_not_accessed(self):
        class OuterForbidden(dict):
            def __getitem__(self, key):
                if key in ("metrics", "predictions", "n_validation"):
                    raise AssertionError("Convergence diagnosis accessed outer outcomes")
                return super().__getitem__(key)
        ordinary = example()
        guarded = OuterForbidden(copy.deepcopy(ordinary))
        guarded.update(metrics=object(), predictions=object(), n_validation=object())
        self.assertEqual(module.diagnose_run(ordinary), module.diagnose_run(guarded))

    def test_selected_epoch_must_be_actual_development_minimum(self):
        value = example()
        value.update(selected_epoch=3, best_development_nll=.9)
        with self.assertRaisesRegex(ValueError, "actual minimum"):
            module.diagnose_run(value)

    def test_invalid_curve_cannot_be_summarized_as_convergence(self):
        for mutation in ("missing", "out_of_order", "nonfinite"):
            with self.subTest(mutation=mutation):
                value = example()
                if mutation == "missing":
                    value["training_curve"].pop()
                elif mutation == "out_of_order":
                    value["training_curve"][1]["epoch"] = 3
                else:
                    value["training_curve"][1]["development_nll"] = float("nan")
                with self.assertRaises(ValueError):
                    module.diagnose_run(value)

    def test_zero_tail_denominator_is_explicitly_undefined(self):
        value = example()
        value["training_curve"][0]["development_nll"] = 0
        value.update(selected_epoch=1, best_development_nll=0.)
        row, _ = module.diagnose_run(value)
        self.assertIsNone(row["tail_development_relative_improvement_percent"])
        self.assertEqual(row["relative_improvement_status"], "undefined_zero_start")


if __name__ == "__main__":
    unittest.main()
