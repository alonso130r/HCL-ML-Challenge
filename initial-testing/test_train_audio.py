import importlib.util
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
