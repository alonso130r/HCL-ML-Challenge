import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_svm_text_head.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("compare_svm_text_head", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class FixedClassifier:
    classes_ = np.array([2, 0, 1])

    def predict_log_proba(self, values):
        return np.tile(np.log([[0.2, 0.5, 0.3]]), (len(values), 1))


class SvmTextHeadComparisonTests(unittest.TestCase):
    def test_fit_svm_head_returns_finite_log_probabilities_for_every_class(self):
        generator = np.random.default_rng(4)
        records = []
        for label in range(3):
            for _ in range(9):
                embedding = generator.normal(size=5)
                embedding[label] += 3
                records.append(
                    {"text_embedding": embedding.astype(np.float32), "label_index": label}
                )

        classifier = comparison.fit_svm_head(records, class_count=3, seed=8, folds=3)
        scores = classifier.predict_log_proba(
            np.stack([record["text_embedding"] for record in records])
        )

        self.assertEqual(scores.shape, (27, 3))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(classifier.classes_, np.arange(3))

    def test_replace_text_logits_restores_canonical_class_order(self):
        records = [
            {"text_embedding": np.array([1.0, 2.0], dtype=np.float32)},
            {"text_embedding": np.array([3.0, 4.0], dtype=np.float32)},
        ]

        comparison.replace_text_logits(records, FixedClassifier(), class_count=3)

        expected = np.log([0.5, 0.3, 0.2])
        np.testing.assert_allclose(records[0]["text_logits"], expected)
        np.testing.assert_allclose(records[1]["text_logits"], expected)

    def test_choose_winner_requires_weighted_gain_without_macro_loss(self):
        baseline = {"weighted_f1": 0.63, "macro_f1": 0.48}

        self.assertEqual(
            comparison.choose_winner(
                baseline, {"weighted_f1": 0.64, "macro_f1": 0.48}
            ),
            "svm",
        )
        self.assertEqual(
            comparison.choose_winner(
                baseline, {"weighted_f1": 0.64, "macro_f1": 0.47}
            ),
            "neural",
        )
        self.assertEqual(
            comparison.choose_winner(
                baseline, {"weighted_f1": 0.62, "macro_f1": 0.49}
            ),
            "neural",
        )


if __name__ == "__main__":
    unittest.main()
