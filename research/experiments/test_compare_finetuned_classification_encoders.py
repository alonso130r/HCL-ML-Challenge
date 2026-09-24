import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name(
    "compare_finetuned_classification_encoders.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("classification_encoders", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class FineTunedClassificationEncoderTests(unittest.TestCase):
    def test_model_specs_cover_control_and_two_candidates(self):
        self.assertEqual(
            set(comparison.MODEL_SPECS), {"bert", "roberta_go_emotions", "deberta"}
        )
        self.assertEqual(
            comparison.MODEL_SPECS["roberta_go_emotions"]["model_id"],
            "SamLowe/roberta-base-go_emotions",
        )
        self.assertEqual(
            comparison.MODEL_SPECS["deberta"]["model_id"],
            "microsoft/deberta-v3-base",
        )

    def test_pretrained_load_replaces_source_head_with_single_label_head(self):
        settings = comparison.model_load_settings("roberta_go_emotions")

        self.assertEqual(settings["num_labels"], 7)
        self.assertEqual(settings["problem_type"], "single_label_classification")
        self.assertTrue(settings["ignore_mismatched_sizes"])
        self.assertEqual(settings["id2label"][0], "neutral")

    def test_model_run_selection_uses_macro_then_weighted_f1(self):
        runs = [
            {"learning_rate": 1e-5, "best_dev": {"macro_f1": 0.50, "weighted_f1": 0.62}},
            {"learning_rate": 2e-5, "best_dev": {"macro_f1": 0.50, "weighted_f1": 0.63}},
        ]

        selected = comparison.select_model_run(runs)

        self.assertEqual(selected["learning_rate"], 2e-5)

    def test_winner_uses_macro_safeguard(self):
        candidates = [
            {"name": "bert", "dev": {"weighted_f1": 0.64, "macro_f1": 0.50}},
            {"name": "deberta", "dev": {"weighted_f1": 0.65, "macro_f1": 0.494}},
        ]

        winner = comparison.choose_winner(candidates, macro_tolerance=0.005)

        self.assertEqual(winner["name"], "bert")

    def test_default_uses_one_fixed_learning_rate(self):
        args = comparison.parse_arguments([])

        self.assertEqual(args.learning_rate, 2e-5)

    def test_default_skips_bert_and_uses_larger_candidate_batches(self):
        args = comparison.parse_arguments([])

        self.assertEqual(args.models, ["roberta_go_emotions", "deberta"])
        self.assertEqual(comparison.MODEL_SPECS["roberta_go_emotions"]["batch_size"], 16)
        self.assertEqual(comparison.MODEL_SPECS["deberta"]["batch_size"], 8)

    def test_weighted_loss_accepts_half_precision_logits(self):
        logits = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float16)
        labels = torch.tensor([1, 0])
        weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

        loss = comparison.weighted_cross_entropy(logits, labels, weights)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
