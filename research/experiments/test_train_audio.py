import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("train_audio.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_audio", MODULE_PATH)
train_audio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_audio)


class AudioTrainingTests(unittest.TestCase):
    def test_pad_audio_returns_relative_lengths(self):
        waveforms = [np.ones(4, dtype=np.float32), np.arange(2, dtype=np.float32)]

        padded, lengths = train_audio.pad_audio(waveforms)

        torch.testing.assert_close(
            padded, torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(lengths, torch.tensor([1.0, 0.5]))

    def test_unfreeze_top_layers_leaves_lower_layers_frozen(self):
        encoder = torch.nn.Module()
        encoder.feature_extractor = torch.nn.Linear(2, 2)
        encoder.encoder = torch.nn.Module()
        encoder.encoder.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(4)]
        )

        train_audio.unfreeze_top_layers(encoder, count=2)

        self.assertFalse(any(p.requires_grad for p in encoder.feature_extractor.parameters()))
        self.assertFalse(any(p.requires_grad for p in encoder.encoder.layers[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in encoder.encoder.layers[2].parameters()))
        self.assertTrue(all(p.requires_grad for p in encoder.encoder.layers[3].parameters()))

    def test_unfreeze_top_layers_rejects_excess_count(self):
        encoder = torch.nn.Module()
        encoder.encoder = torch.nn.Module()
        encoder.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

        with self.assertRaisesRegex(ValueError, "only 1"):
            train_audio.unfreeze_top_layers(encoder, count=2)

    def test_frame_padding_mask_uses_relative_audio_lengths(self):
        lengths = torch.tensor([1.0, 0.5, 0.01])

        mask = train_audio.frame_padding_mask(lengths, frame_count=4)

        torch.testing.assert_close(
            mask,
            torch.tensor(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, True, True, True],
                ]
            ),
        )

    def test_attentive_statistics_pooling_ignores_padding(self):
        pooling = train_audio.AttentiveStatisticsPooling(dimension=2)
        torch.nn.init.zeros_(pooling.attention.weight)
        torch.nn.init.zeros_(pooling.attention.bias)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        padding_mask = torch.tensor([[False, False, True]])

        pooled = pooling(hidden, padding_mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0, 1.0, 1.0]]))

    def test_audio_head_pools_encoder_frames_without_temporal_transformer(self):
        encoder = torch.nn.Identity()

        model = train_audio.MeldAudioClassifier(encoder)

        self.assertFalse(hasattr(model, "temporal_encoder"))
        self.assertEqual(model.pool.attention.in_features, 768)
        self.assertEqual(model.classifier[0].in_features, 1536)
        self.assertEqual(model.classifier[-1].out_features, 7)

    def test_wavlm_encoder_passes_raw_sample_attention_mask(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_mask = None

            def forward(self, input_values, attention_mask):
                self.received_mask = attention_mask
                return SimpleNamespace(last_hidden_state=input_values.unsqueeze(-1))

        base = FakeModel()
        encoder = train_audio.WavLmEncoder(base)
        encoder.freeze = False

        encoder(torch.ones(2, 4), torch.tensor([1.0, 0.5]))

        torch.testing.assert_close(
            base.received_mask,
            torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        )


if __name__ == "__main__":
    unittest.main()
