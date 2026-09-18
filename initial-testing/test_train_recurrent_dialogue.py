import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
