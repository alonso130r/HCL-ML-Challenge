import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).with_name("train_recurrent_dialogue_stabilized.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "train_recurrent_dialogue_stabilized", MODULE_PATH
)
stabilized = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stabilized)


def model_args():
    return SimpleNamespace(
        text_projection_dimension=6,
        audio_projection_dimension=5,
        dialogue_state_dimension=7,
        speaker_state_dimension=4,
        dropout=0.0,
        context_max_gate=0.2,
        audio_max_gate=0.2,
        initial_gate_bias=-1.0,
        dialogue_state_dropout=0.0,
        speaker_state_dropout=0.0,
        audio_dropout=0.0,
        dialogue_reset_probability=0.0,
        speaker_reset_probability=0.0,
    )


def make_batch(turns=2):
    return {
        "text_embeddings": torch.randn(1, turns, 3),
        "text_logits": torch.randn(1, turns, 3),
        "audio_features": torch.randn(1, turns, 2),
        "speaker_indices": torch.zeros(1, turns, dtype=torch.long),
        "labels": torch.zeros(1, turns, dtype=torch.long),
        "valid_mask": torch.ones(1, turns, dtype=torch.bool),
    }


class StabilizedRecurrentDialogueTests(unittest.TestCase):
    def test_summarize_runs_includes_zero_audio_evidence(self):
        runs = [
            {
                "test_recurrent_matched": {
                    "weighted_f1": 0.63,
                    "macro_f1": 0.47,
                },
                "test_state_margin": 0.003,
                "test_audio_margin": 0.006,
                "test_zero_audio_margin": 0.004,
            },
            {
                "test_recurrent_matched": {
                    "weighted_f1": 0.65,
                    "macro_f1": 0.49,
                },
                "test_state_margin": 0.005,
                "test_audio_margin": 0.008,
                "test_zero_audio_margin": 0.006,
            },
        ]

        summary = stabilized.summarize_runs(runs)

        self.assertAlmostEqual(summary["weighted_f1"]["mean"], 0.64)
        self.assertAlmostEqual(summary["zero_audio_margin"]["mean"], 0.005)
        self.assertEqual(summary["state_margin"]["count"], 2)

    def test_first_turn_context_residual_is_exactly_zero(self):
        torch.manual_seed(3)
        model = stabilized.StabilizedRecurrentDialogueModel(
            3, 2, 3, model_args()
        ).eval()

        with torch.inference_mode():
            output = model(make_batch())

        torch.testing.assert_close(
            output["context_correction"][:, 0],
            torch.zeros_like(output["context_correction"][:, 0]),
            atol=0,
            rtol=0,
        )

    def test_reset_each_turn_zeroes_every_context_residual(self):
        torch.manual_seed(4)
        model = stabilized.StabilizedRecurrentDialogueModel(
            3, 2, 3, model_args()
        ).eval()

        with torch.inference_mode():
            output = model(make_batch(3), reset_each_turn=True)

        torch.testing.assert_close(
            output["context_correction"],
            torch.zeros_like(output["context_correction"]),
            atol=0,
            rtol=0,
        )

    def test_zero_state_residual_stays_zero_during_training(self):
        torch.manual_seed(12)
        args = model_args()
        args.dropout = 0.5
        model = stabilized.StabilizedRecurrentDialogueModel(3, 2, 3, args).train()
        torch.nn.init.normal_(model.context_correction.weight)

        output = model(make_batch(), reset_each_turn=True)

        torch.testing.assert_close(
            output["context_correction"],
            torch.zeros_like(output["context_correction"]),
            atol=0,
            rtol=0,
        )

    def test_previous_turn_changes_later_context_but_not_current_context(self):
        torch.manual_seed(5)
        model = stabilized.StabilizedRecurrentDialogueModel(
            3, 2, 3, model_args()
        ).eval()
        torch.nn.init.normal_(model.context_correction.weight)
        batch = make_batch()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["text_embeddings"][:, 0] += 20
        changed["audio_features"][:, 0] += 20

        with torch.inference_mode():
            original = model(batch)["context_correction"]
            modified = model(changed)["context_correction"]

        torch.testing.assert_close(original[:, 0], modified[:, 0], atol=0, rtol=0)
        self.assertFalse(torch.allclose(original[:, 1], modified[:, 1]))

    def test_state_ranking_penalty_is_zero_at_requested_margin(self):
        labels = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, True]])
        matched = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        reset = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        penalty = stabilized.state_ranking_loss(
            matched, reset, labels, mask, margin=0.02
        )

        self.assertEqual(float(penalty), 0.0)

    def test_checkpoint_score_requires_text_and_state_margins(self):
        baseline = {"weighted_f1": 0.62}
        matched = {"weighted_f1": 0.625, "macro_f1": 0.48}

        eligible = stabilized.checkpoint_score(
            matched, baseline, state_margin=0.003, minimum_state_margin=0.002
        )
        ineligible = stabilized.checkpoint_score(
            matched, baseline, state_margin=0.001, minimum_state_margin=0.002
        )

        self.assertEqual(eligible[0], 1)
        self.assertEqual(ineligible[0], 0)
        self.assertGreater(eligible, ineligible)


if __name__ == "__main__":
    unittest.main()
