import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("compare_finetuned_text_encoders.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("compare_text_encoders", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class FineTunedTextEncoderComparisonTests(unittest.TestCase):
    def test_masked_mean_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])

        pooled = comparison.masked_mean_pool(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))

    def test_e5_gets_query_prefix_but_bert_does_not(self):
        text = "Context:\n[A] First\nCurrent:\n[B] Second"

        self.assertEqual(comparison.format_text(text, "bert"), text)
        self.assertEqual(comparison.format_text(text, "e5"), f"query: {text}")

    def test_winner_respects_macro_tolerance_before_weighted_f1(self):
        candidates = [
            {"name": "bert", "dev": {"weighted_f1": 0.64, "macro_f1": 0.50}},
            {"name": "e5", "dev": {"weighted_f1": 0.65, "macro_f1": 0.494}},
        ]

        winner = comparison.choose_winner(candidates, macro_tolerance=0.005)

        self.assertEqual(winner["name"], "bert")

    def test_best_learning_rate_is_selected_by_dev_macro_then_weighted(self):
        runs = [
            {"learning_rate": 1e-5, "best_dev": {"macro_f1": 0.50, "weighted_f1": 0.62}},
            {"learning_rate": 2e-5, "best_dev": {"macro_f1": 0.50, "weighted_f1": 0.63}},
        ]

        selected = comparison.select_model_run(runs)

        self.assertEqual(selected["learning_rate"], 2e-5)


if __name__ == "__main__":
    unittest.main()
