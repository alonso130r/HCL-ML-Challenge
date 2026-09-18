import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np


MODULE_PATH = Path(__file__).with_name("train_text_audio.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_text_audio", MODULE_PATH)
train_text_audio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_text_audio)


class TextAudioFusionTests(unittest.TestCase):
    def test_disable_layerdrop_updates_model_and_encoder_configs(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(layerdrop=0.1)
        model.encoder = torch.nn.Module()
        model.encoder.config = SimpleNamespace(layerdrop=0.1)

        train_text_audio.disable_layerdrop(model)

        self.assertEqual(model.config.layerdrop, 0.0)
        self.assertEqual(model.encoder.config.layerdrop, 0.0)

    def test_disable_pretraining_masking_handles_all_model_configs(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(mask_time_prob=0.05, mask_feature_prob=0.1)
        model.encoder = torch.nn.Module()
        model.encoder.config = SimpleNamespace(
            mask_time_prob=0.05, mask_feature_prob=0.1
        )

        train_text_audio.disable_pretraining_masking(model)

        self.assertEqual(model.config.mask_time_prob, 0.0)
        self.assertEqual(model.config.mask_feature_prob, 0.0)
        self.assertEqual(model.encoder.config.mask_time_prob, 0.0)
        self.assertEqual(model.encoder.config.mask_feature_prob, 0.0)

    def test_sanitize_acoustic_features_accepts_read_only_arrays(self):
        values = np.array([1.0, np.nan, np.inf], dtype=np.float32)
        values.setflags(write=False)

        cleaned = train_text_audio.sanitize_acoustic_features(values)

        np.testing.assert_array_equal(cleaned, np.array([1.0, 0.0, 0.0]))
        self.assertTrue(cleaned.flags.writeable)

    def test_collator_preserves_original_contextual_text_format(self):
        class FakeTokenizer:
            def __init__(self):
                self.received = None

            def __call__(self, texts, **kwargs):
                self.received = texts
                return {
                    "input_ids": torch.tensor([[101, 10, 11, 102]]),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                    "token_type_ids": torch.zeros(1, 4, dtype=torch.long),
                    "special_tokens_mask": torch.tensor([[1, 0, 0, 1]]),
                    "offset_mapping": torch.tensor(
                        [[[0, 0], [0, 8], [9, 40], [0, 0]]]
                    ),
                }

        tokenizer = FakeTokenizer()
        collate = train_text_audio.make_collator(tokenizer, max_length=64)
        item = {
            "context": "Context:\n[Rachel] Hello",
            "current": "[Ross] Fine.",
            "waveform": torch.ones(16).numpy(),
            "acoustic": torch.zeros(88).numpy(),
            "label": 0,
            "index": 0,
        }

        batch = collate([item])

        self.assertEqual(
            tokenizer.received,
            ["Context:\n[Rachel] Hello\nCurrent:\n[Ross] Fine."],
        )
        self.assertIn("current_token_mask", batch)
        self.assertNotIn("offset_mapping", batch)

    def test_current_token_mask_uses_second_bert_segment_and_attention_mask(self):
        token_types = torch.tensor([[0, 0, 1, 1, 1, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 0, 0]])

        mask = train_text_audio.current_token_mask(token_types, attention)

        torch.testing.assert_close(
            mask, torch.tensor([[False, False, True, True, False, False]])
        )

    def test_current_token_mask_falls_back_to_non_special_attended_tokens(self):
        attention = torch.tensor([[1, 1, 1, 0]])
        special = torch.tensor([[1, 0, 1, 1]])

        mask = train_text_audio.current_token_mask(None, attention, special)

        torch.testing.assert_close(mask, torch.tensor([[False, True, False, False]]))

    def test_layer_mixture_selects_requested_hidden_layers(self):
        mixture = train_text_audio.LearnedLayerMixture(layer_indices=(1, 3))
        hidden = tuple(
            torch.full((1, 2, 1), float(value)) for value in range(4)
        )

        mixed = mixture(hidden)

        torch.testing.assert_close(mixed, torch.full((1, 2, 1), 2.0))

    def test_layer_mixture_rejects_missing_hidden_layer(self):
        mixture = train_text_audio.LearnedLayerMixture(layer_indices=(2,))

        with self.assertRaisesRegex(ValueError, "hidden states"):
            mixture((torch.zeros(1, 1, 1),))

    def test_masked_mean_ignores_padding(self):
        sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        padding_mask = torch.tensor([[False, False, True]])

        pooled = train_text_audio.masked_mean(sequence, padding_mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_residual_fusion_initially_stays_close_to_text_logits(self):
        fusion = train_text_audio.GatedResidualFusion(
            text_dimension=4,
            audio_dimension=4,
            acoustic_dimension=3,
            fusion_dimension=4,
            number_of_heads=2,
            number_of_classes=2,
            dropout=0.0,
            initial_gate_bias=-6.0,
        )
        text_hidden = torch.ones(1, 3, 4)
        audio_hidden = torch.ones(1, 2, 4)
        text_padding = torch.tensor([[False, False, True]])
        audio_padding = torch.tensor([[False, False]])
        acoustic = torch.ones(1, 3)
        text_logits = torch.tensor([[2.0, -1.0]])

        output = fusion(
            text_hidden,
            text_padding,
            audio_hidden,
            audio_padding,
            acoustic,
            text_logits,
        )

        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertEqual(output["audio_logits"].shape, (1, 2))
        self.assertEqual(output["gate"].shape, (1, 2))
        self.assertLess(
            float((output["logits"] - text_logits).abs().max().detach()), 0.02
        )
        self.assertLess(float(output["gate"].max().detach()), 0.01)


if __name__ == "__main__":
    unittest.main()
