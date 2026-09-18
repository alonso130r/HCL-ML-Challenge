import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("train_text_audio_phase2.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_text_audio_phase2", MODULE_PATH)
phase2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase2)


class TextAudioPhaseTwoTests(unittest.TestCase):
    def test_contrastive_loss_prefers_matching_pairs(self):
        text = torch.eye(3)
        matched = torch.eye(3)
        shuffled = matched[[1, 2, 0]]

        self.assertLess(
            phase2.symmetric_contrastive_loss(text, matched, 0.1),
            phase2.symmetric_contrastive_loss(text, shuffled, 0.1),
        )

    def test_checkpoint_score_requires_audio_to_beat_every_control(self):
        matched = {"weighted_f1": 0.640, "macro_f1": 0.490}
        controls = [
            {"weighted_f1": 0.632, "macro_f1": 0.480},
            {"weighted_f1": 0.633, "macro_f1": 0.481},
        ]

        eligible, margin, macro = phase2.checkpoint_score(
            matched, controls, minimum_margin=0.005
        )

        self.assertEqual(eligible, 1)
        self.assertAlmostEqual(margin, 0.007)
        self.assertEqual(macro, 0.490)

    def test_checkpoint_score_rejects_a_single_stronger_control(self):
        matched = {"weighted_f1": 0.640, "macro_f1": 0.490}
        controls = [
            {"weighted_f1": 0.632, "macro_f1": 0.480},
            {"weighted_f1": 0.641, "macro_f1": 0.491},
        ]

        score = phase2.checkpoint_score(matched, controls, minimum_margin=0.0)

        self.assertEqual(score[0], 0)
        self.assertAlmostEqual(score[1], -0.001)

    def test_masked_statistics_ignore_padding(self):
        sequence = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [99.0, 99.0]]])
        padding = torch.tensor([[False, False, True]])

        result = phase2.masked_mean_std(sequence, padding)

        torch.testing.assert_close(result, torch.tensor([[2.0, 4.0, 1.0, 2.0]]))

    def test_derangement_has_no_fixed_points_and_is_reproducible(self):
        first = phase2.make_derangement(12, seed=19)
        second = phase2.make_derangement(12, seed=19)

        self.assertEqual(first, second)
        self.assertTrue(all(index != value for index, value in enumerate(first)))

    def test_hard_negatives_choose_different_label_with_closest_duration(self):
        labels = torch.tensor([0, 1, 2, 0])
        lengths = torch.tensor([10, 30, 12, 40])

        negatives = phase2.hard_negative_indices(labels, lengths)

        self.assertEqual(negatives.tolist(), [2, 3, 0, 1])
        self.assertTrue(torch.all(labels[negatives] != labels))

    def test_permute_audio_keeps_text_and_labels_fixed(self):
        batch = {
            "input_ids": torch.tensor([[1], [2], [3]]),
            "labels": torch.tensor([0, 1, 2]),
            "audio_frames": torch.tensor([[[10.0]], [[20.0]], [[30.0]]]),
            "audio_padding_mask": torch.tensor([[False], [False], [False]]),
            "acoustic_features": torch.tensor([[100.0], [200.0], [300.0]]),
        }

        negative = phase2.permute_audio(batch, torch.tensor([2, 0, 1]))

        torch.testing.assert_close(negative["input_ids"], batch["input_ids"])
        torch.testing.assert_close(negative["labels"], batch["labels"])
        torch.testing.assert_close(
            negative["audio_frames"].flatten(), torch.tensor([30.0, 10.0, 20.0])
        )
        torch.testing.assert_close(
            negative["acoustic_features"].flatten(),
            torch.tensor([300.0, 100.0, 200.0]),
        )

    def test_counterfactual_loss_is_zero_when_matched_has_required_advantage(self):
        matched = {
            "logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "gate": torch.tensor([[0.20], [0.20]]),
            "correction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        negative = {
            "logits": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "gate": torch.tensor([[0.05], [0.05]]),
            "correction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }

        losses = phase2.counterfactual_losses(
            matched,
            negative,
            torch.tensor([0, 1]),
            prediction_margin=0.1,
            gate_margin=0.05,
        )

        self.assertEqual(float(losses["ranking"]), 0.0)
        self.assertEqual(float(losses["gate_ranking"]), 0.0)
        self.assertGreater(float(losses["negative_residual"]), 0.0)


if __name__ == "__main__":
    unittest.main()
