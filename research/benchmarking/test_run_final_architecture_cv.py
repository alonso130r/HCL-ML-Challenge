import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_final_architecture_cv.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("final_cv", MODULE_PATH)
final_cv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final_cv)


class FinalArchitectureCVTests(unittest.TestCase):
    def test_group_folds_are_deterministic_complete_and_disjoint(self):
        records = []
        for dialogue in range(20):
            for turn in range(2 + dialogue % 3):
                records.append(
                    {
                        "dialogue_id": str(dialogue),
                        "label": final_cv.EMOTION_LABELS[(dialogue + turn) % 7],
                    }
                )

        first = final_cv.assign_dialogue_folds(records, folds=5, seed=17)
        second = final_cv.assign_dialogue_folds(records, folds=5, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {str(index) for index in range(20)})
        self.assertEqual(set(first.values()), set(range(5)))

    def test_temperature_calibration_returns_positive_temperature(self):
        logits = np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int64)

        temperature = final_cv.fit_temperature(logits, labels)

        self.assertGreater(temperature, 0.0)
        self.assertTrue(np.isfinite(temperature))

    def test_confusion_metrics_match_expected_values(self):
        confusion = np.array([[2, 0], [1, 1]], dtype=np.int64)

        metrics = final_cv.metrics_from_confusion(confusion)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.7333333333333334)
        self.assertAlmostEqual(metrics["macro_f1"], 0.7333333333333334)

    def test_holm_adjustment_is_monotonic_in_sorted_p_values(self):
        adjusted = final_cv.holm_adjust([0.01, 0.03, 0.02, 0.5])

        self.assertGreaterEqual(adjusted[1], adjusted[2])
        self.assertGreaterEqual(adjusted[3], adjusted[1])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))

    def test_dialogue_bootstrap_runs_on_seed_aligned_predictions(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        matched = np.array([[4, 0], [0, 4], [4, 0], [0, 4]], dtype=np.float32)
        control = np.array([[4, 0], [4, 0], [4, 0], [4, 0]], dtype=np.float32)
        run = {
            "labels": labels,
            "dialogue_ids": np.array(["0", "0", "1", "1"]),
            "logits": matched,
            "text_logits": control,
            "zero_audio_logits": control,
            "shuffled_logits": control,
            "reset_logits": control,
        }

        result = final_cv.bootstrap_margins({42: run, 43: run}, 25, 9)

        self.assertGreater(result["over_text"]["mean"], 0.0)
        self.assertEqual(len(result["matched_weighted_f1"]["ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
