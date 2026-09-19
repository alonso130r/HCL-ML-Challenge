import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name(
    "train_recurrent_dialogue_heavy_regularization.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "train_recurrent_dialogue_heavy_regularization", MODULE_PATH
)
regularized = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regularized)


class HeavyRegularizationTests(unittest.TestCase):
    def test_default_configuration_is_moderately_regularized(self):
        args = regularized.parse_arguments([])

        self.assertEqual(args.feature_mask_probability, 0.0)
        self.assertEqual(args.state_noise_std, 0.0)
        self.assertEqual(args.label_smoothing, 0.02)
        self.assertEqual(args.consistency_weight, 0.15)
        self.assertEqual(args.history_truncation_probability, 0.15)
        self.assertEqual(args.group_mask_probability, 0.08)
        self.assertEqual(args.gate_variance_weight, 0.0)

    def test_modality_conditions_apply_zero_reset_and_shuffle(self):
        batch = {
            "audio_features": torch.arange(24, dtype=torch.float32).reshape(3, 2, 4),
            "valid_mask": torch.ones(3, 2, dtype=torch.bool),
            "labels": torch.tensor([[0, 0], [1, 1], [2, 2]]),
        }
        conditions = torch.tensor([
            regularized.ZERO_AUDIO,
            regularized.RESET_STATE,
            regularized.SHUFFLED_AUDIO,
        ])

        changed = regularized.apply_modality_conditions(batch, conditions)

        self.assertTrue(torch.all(changed["audio_features"][0] == 0))
        self.assertTrue(torch.all(changed["forced_reset_mask"][1]))
        self.assertFalse(
            torch.equal(changed["audio_features"][2], batch["audio_features"][2])
        )

    def test_group_masking_can_remove_a_complete_feature_group(self):
        audio = torch.ones(2, 3, 6)
        generator = torch.Generator().manual_seed(7)

        corrupted = regularized.corrupt_audio_features(
            audio,
            group_slices=[(0, 2), (2, 6)],
            feature_mask_probability=0.0,
            group_mask_probability=1.0,
            noise_std=0.0,
            gain_range=(1.0, 1.0),
            generator=generator,
        )

        for row in corrupted:
            self.assertTrue(torch.any(row == 0))
            self.assertTrue(torch.any(row == 1))

    def test_symmetric_kl_is_zero_for_identical_logits(self):
        logits = torch.tensor([[[1.0, 2.0], [0.5, -0.5]]])
        mask = torch.tensor([[True, True]])

        loss = regularized.symmetric_kl(logits, logits, mask)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_zoneout_one_preserves_previous_state(self):
        cell = regularized.ZoneoutGRUCell(3, 4, zoneout_probability=1.0)
        cell.train()
        inputs = torch.randn(2, 3)
        previous = torch.randn(2, 4)

        output = cell(inputs, previous)

        torch.testing.assert_close(output, previous)

    def test_ema_blends_parameters(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(0.0)
        ema = regularized.ExponentialMovingAverage(model, decay=0.5)
        model.weight.data.fill_(2.0)

        ema.update(model)

        torch.testing.assert_close(
            ema.model.weight, torch.full_like(ema.model.weight, 1.0)
        )

    def test_random_history_mask_has_at_most_one_cut_per_dialogue(self):
        valid = torch.tensor([[True, True, True], [True, True, False]])
        generator = torch.Generator().manual_seed(2)

        mask = regularized.random_history_reset_mask(
            valid, probability=1.0, generator=generator
        )

        self.assertTrue(torch.all(mask.sum(dim=1) == 1))
        self.assertTrue(torch.all(mask <= valid))


if __name__ == "__main__":
    unittest.main()
