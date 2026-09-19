import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name(
    "train_recurrent_dialogue_frame_attention.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "train_recurrent_dialogue_frame_attention", MODULE_PATH
)
frame_attention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frame_attention)


class FrameAttentionTests(unittest.TestCase):
    def test_uniform_frame_sampling_caps_length_and_keeps_endpoints(self):
        frames = np.arange(30, dtype=np.float32).reshape(10, 3)

        sampled = frame_attention.uniform_sample_frames(frames, 4)

        self.assertEqual(sampled.shape, (4, 3))
        np.testing.assert_array_equal(sampled[0], frames[0])
        np.testing.assert_array_equal(sampled[-1], frames[-1])

    def test_attention_masks_padding_and_normalizes_weights(self):
        attention = frame_attention.TextConditionedFramePool(
            text_dimension=3, frame_dimension=3
        )
        attention.query.weight.data.copy_(torch.eye(3))
        attention.query.bias.data.zero_()
        text = torch.tensor([[[1.0, 0.0, 0.0]]])
        frames = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [100.0, 0.0, 0.0]]]]
        )
        mask = torch.tensor([[[True, True, False]]])

        _, weights = attention(text, frames, mask)

        self.assertEqual(float(weights[0, 0, 2].detach()), 0.0)
        self.assertAlmostEqual(float(weights[0, 0].sum().detach()), 1.0, places=6)

    def test_text_query_changes_the_pooled_audio(self):
        attention = frame_attention.TextConditionedFramePool(
            text_dimension=2, frame_dimension=2
        )
        attention.query.weight.data.copy_(torch.eye(2))
        attention.query.bias.data.zero_()
        frames = torch.tensor([[[[3.0, 0.0], [0.0, 3.0]]]])
        mask = torch.ones(1, 1, 2, dtype=torch.bool)

        first, _ = attention(torch.tensor([[[1.0, 0.0]]]), frames, mask)
        second, _ = attention(torch.tensor([[[0.0, 1.0]]]), frames, mask)

        first = first.detach()
        second = second.detach()
        self.assertGreater(float(first[0, 0, 0]), float(first[0, 0, 1]))
        self.assertGreater(float(second[0, 0, 1]), float(second[0, 0, 0]))

    def test_attention_remains_selective_at_emotion2vec_dimensions(self):
        dimension = 768
        attention = frame_attention.TextConditionedFramePool(
            text_dimension=dimension, frame_dimension=dimension
        )
        attention.query.weight.data.copy_(torch.eye(dimension))
        attention.query.bias.data.zero_()
        text = torch.zeros(1, 1, dimension)
        text[0, 0, 0] = 1
        frames = torch.zeros(1, 1, 2, dimension)
        frames[0, 0, 0, 0] = 1
        frames[0, 0, 1, 1] = 1
        mask = torch.ones(1, 1, 2, dtype=torch.bool)

        _, weights = attention(text, frames, mask)

        self.assertGreater(float(weights[0, 0, 0].detach()), 0.90)

    def test_collate_pads_turns_and_audio_frames(self):
        items = [
            {
                "text_embeddings": np.ones((2, 3), dtype=np.float32),
                "text_logits": np.ones((2, 2), dtype=np.float32),
                "acoustic_features": np.ones((2, 4), dtype=np.float32),
                "emotion_frames": [
                    np.ones((3, 5), dtype=np.float32),
                    np.ones((2, 5), dtype=np.float32),
                ],
                "speaker_indices": np.array([0, 1]),
                "labels": np.array([0, 1]),
                "record_indices": np.array([0, 1]),
            },
            {
                "text_embeddings": np.ones((1, 3), dtype=np.float32),
                "text_logits": np.ones((1, 2), dtype=np.float32),
                "acoustic_features": np.ones((1, 4), dtype=np.float32),
                "emotion_frames": [np.ones((1, 5), dtype=np.float32)],
                "speaker_indices": np.array([0]),
                "labels": np.array([1]),
                "record_indices": np.array([2]),
            },
        ]

        batch = frame_attention.collate_frame_dialogues(items)

        self.assertEqual(batch["emotion_frames"].shape, (2, 2, 3, 5))
        self.assertEqual(batch["frame_mask"].sum().item(), 6)
        self.assertEqual(batch["valid_mask"].sum().item(), 3)


if __name__ == "__main__":
    unittest.main()
