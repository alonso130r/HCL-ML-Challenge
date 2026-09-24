import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("train_recurrent_dialogue.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_recurrent_dialogue", MODULE_PATH)
recurrent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recurrent)


class RecurrentDialogueTests(unittest.TestCase):
    def test_run_seeds_are_reproducible_and_distinct(self):
        self.assertEqual(
            recurrent.make_run_seeds(base_seed=42, runs=4, step=3),
            [42, 45, 48, 51],
        )

    def test_summarize_runs_reports_sample_statistics(self):
        runs = [
            {
                "test_recurrent_matched": {
                    "weighted_f1": 0.60,
                    "macro_f1": 0.40,
                },
                "test_state_margin": 0.01,
                "test_audio_margin": 0.02,
            },
            {
                "test_recurrent_matched": {
                    "weighted_f1": 0.64,
                    "macro_f1": 0.50,
                },
                "test_state_margin": 0.03,
                "test_audio_margin": 0.04,
            },
        ]

        summary = recurrent.summarize_runs(runs)

        self.assertAlmostEqual(summary["weighted_f1"]["mean"], 0.62)
        self.assertAlmostEqual(summary["weighted_f1"]["std"], 0.02)
        self.assertAlmostEqual(summary["macro_f1"]["mean"], 0.45)
        self.assertEqual(summary["state_margin"]["count"], 2)

    def test_load_completed_run_requires_matching_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metrics.json").write_text(
                json.dumps({"seed": 42, "test_state_margin": 0.1}),
                encoding="utf-8",
            )

            self.assertEqual(
                recurrent.load_completed_run(run_dir, 42)["test_state_margin"],
                0.1,
            )
            self.assertIsNone(recurrent.load_completed_run(run_dir, 43))
    def test_group_dialogues_orders_turns_and_assigns_local_speakers(self):
        records = [
            {"dialogue_id": "2", "utterance_id": "1", "speaker": "B"},
            {"dialogue_id": "1", "utterance_id": "2", "speaker": "A"},
            {"dialogue_id": "1", "utterance_id": "0", "speaker": "B"},
            {"dialogue_id": "1", "utterance_id": "1", "speaker": "A"},
        ]

        dialogues = recurrent.group_dialogues(records)

        self.assertEqual(len(dialogues), 2)
        self.assertEqual(
            [item["utterance_id"] for item in dialogues[0]], ["0", "1", "2"]
        )
        self.assertEqual(
            [item["speaker_index"] for item in dialogues[0]], [0, 1, 1]
        )

    def test_different_label_audio_mapping_is_reproducible(self):
        records = [
            {"label": "neutral"},
            {"label": "neutral"},
            {"label": "joy"},
            {"label": "anger"},
        ]

        first = recurrent.different_label_audio_mapping(records, 17)
        second = recurrent.different_label_audio_mapping(records, 17)

        self.assertEqual(first, second)
        self.assertTrue(
            all(records[index]["label"] != records[value]["label"]
                for index, value in enumerate(first))
        )

    def test_future_turn_does_not_change_earlier_prediction(self):
        torch.manual_seed(4)
        args = SimpleNamespace(
            text_projection_dimension=6,
            audio_projection_dimension=5,
            dialogue_state_dimension=7,
            speaker_state_dimension=4,
            dropout=0.0,
            context_max_gate=0.2,
            audio_max_gate=0.2,
            initial_gate_bias=-1.0,
        )
        model = recurrent.RecurrentDialogueModel(3, 2, 3, args).eval()
        base = {
            "text_embeddings": torch.randn(1, 3, 3),
            "text_logits": torch.randn(1, 3, 3),
            "audio_features": torch.randn(1, 3, 2),
            "speaker_indices": torch.tensor([[0, 1, 0]]),
            "valid_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        changed = {key: value.clone() for key, value in base.items()}
        changed["text_embeddings"][:, 2] += 100
        changed["text_logits"][:, 2] += 100
        changed["audio_features"][:, 2] += 100

        with torch.inference_mode():
            original = model(base)["logits"]
            modified = model(changed)["logits"]

        torch.testing.assert_close(original[:, :2], modified[:, :2])

    def test_reset_each_turn_removes_cross_turn_dependence(self):
        torch.manual_seed(8)
        args = SimpleNamespace(
            text_projection_dimension=6,
            audio_projection_dimension=5,
            dialogue_state_dimension=7,
            speaker_state_dimension=4,
            dropout=0.0,
            context_max_gate=0.2,
            audio_max_gate=0.2,
            initial_gate_bias=-1.0,
        )
        model = recurrent.RecurrentDialogueModel(3, 2, 3, args).eval()
        batch = {
            "text_embeddings": torch.randn(1, 2, 3),
            "text_logits": torch.randn(1, 2, 3),
            "audio_features": torch.randn(1, 2, 2),
            "speaker_indices": torch.tensor([[0, 0]]),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        changed = {key: value.clone() for key, value in batch.items()}
        changed["text_embeddings"][:, 0] += 50
        changed["audio_features"][:, 0] += 50

        with torch.inference_mode():
            original = model(batch, reset_each_turn=True)["logits"][:, 1]
            modified = model(changed, reset_each_turn=True)["logits"][:, 1]

        torch.testing.assert_close(original, modified)

    def test_locked_dropout_mask_uses_inverted_dropout_scaling(self):
        torch.manual_seed(3)
        reference = torch.ones(64, 8)

        mask = recurrent.locked_dropout_mask(reference, 0.25, training=True)

        self.assertTrue(torch.any(mask == 0))
        self.assertTrue(torch.any(mask > 0))
        torch.testing.assert_close(
            mask[mask > 0], torch.full_like(mask[mask > 0], 1.0 / 0.75)
        )

    def test_gate_ceiling_penalty_ignores_gates_below_soft_limits(self):
        output = {
            "context_gate": torch.tensor([[[0.10], [0.19]]]),
            "audio_gate": torch.tensor([[[0.05], [0.12]]]),
        }
        mask = torch.tensor([[True, True]])

        penalty = recurrent.gate_ceiling_penalty(
            output, mask, context_ceiling=0.18, audio_ceiling=0.10
        )

        self.assertAlmostEqual(float(penalty), (0.01**2 + 0.02**2) / 2)

    def test_training_resets_can_remove_cross_turn_dependence(self):
        torch.manual_seed(11)
        args = SimpleNamespace(
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
            dialogue_reset_probability=1.0,
            speaker_reset_probability=1.0,
        )
        model = recurrent.RecurrentDialogueModel(3, 2, 3, args).train()
        batch = {
            "text_embeddings": torch.randn(1, 2, 3),
            "text_logits": torch.randn(1, 2, 3),
            "audio_features": torch.randn(1, 2, 2),
            "speaker_indices": torch.tensor([[0, 0]]),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        changed = {key: value.clone() for key, value in batch.items()}
        changed["text_embeddings"][:, 0] += 50
        changed["audio_features"][:, 0] += 50

        original = model(batch)["logits"][:, 1]
        modified = model(changed)["logits"][:, 1]

        torch.testing.assert_close(original, modified)


if __name__ == "__main__":
    unittest.main()
