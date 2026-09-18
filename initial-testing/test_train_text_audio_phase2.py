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


if __name__ == "__main__":
    unittest.main()
