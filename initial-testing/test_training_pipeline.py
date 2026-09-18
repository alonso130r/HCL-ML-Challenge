import sys
import io
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


TESTING_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTING_DIR))

import cache_embeddings
import train_mlp


class CacheEmbeddingTests(unittest.TestCase):
    def test_read_csv_accepts_non_seekable_tar_stream(self):
        class NonSeekable:
            def __init__(self, value):
                self.value = io.BytesIO(value)

            def read(self, size=-1):
                return self.value.read(size)

        source = NonSeekable(
            b"Emotion,Dialogue_ID,Utterance_ID,Utterance\r\njoy,1,2,Hello\r\n"
        )

        rows = cache_embeddings._read_csv(source)

        self.assertEqual(rows[0]["Emotion"], "joy")

    def test_media_key_accepts_only_canonical_meld_filename(self):
        self.assertEqual(cache_embeddings.media_key("folder/dia48_utt3.mp4"), ("48", "3"))
        self.assertIsNone(
            cache_embeddings.media_key("folder/final_videos_testdia48_utt3.mp4")
        )

    def test_decode_clip_records_corrupt_media(self):
        with mock.patch.object(
            cache_embeddings,
            "decode_audio",
            side_effect=ValueError("invalid media"),
        ):
            audio, error = cache_embeddings.decode_clip(Path("broken.mp4"))

        self.assertIsNone(audio)
        self.assertEqual(error, "ValueError: invalid media")

    def test_cache_round_trip_preserves_features_and_metadata(self):
        records = [
            {
                "label": "joy",
                "dialogue_id": "4",
                "utterance_id": "2",
                "utterance": "Hello",
            }
        ]
        features = np.arange(
            cache_embeddings.FEATURE_DIMENSION, dtype=np.float32
        ).reshape(1, cache_embeddings.FEATURE_DIMENSION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            cache_embeddings.save_cache(
                path,
                "train",
                features,
                records,
                failures=[{"dialogue_id": "9", "error": "broken"}],
                source_count=2,
            )
            loaded = cache_embeddings.load_cache(path)

        np.testing.assert_array_equal(loaded["features"], features)
        self.assertEqual(loaded["labels"].tolist(), ["joy"])
        self.assertEqual(loaded["split"].item(), "train")
        self.assertEqual(loaded["source_count"].item(), 2)
        self.assertEqual(len(__import__("json").loads(loaded["failures_json"].item())), 1)


class TrainMlpTests(unittest.TestCase):
    def test_normalization_uses_training_statistics_and_handles_constants(self):
        features = np.zeros((3, cache_embeddings.FEATURE_DIMENSION), dtype=np.float32)
        features[:, 0] = [1.0, 2.0, 3.0]
        features[:, 768] = [2.0, 4.0, 6.0]
        features[:, 1024] = [5.0, 5.0, 5.0]

        mean, standard_deviation = train_mlp.fit_normalizer(features)
        normalized = train_mlp.apply_normalizer(features, mean, standard_deviation)

        self.assertAlmostEqual(float(normalized[:, 0].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:, 0].std()), 1.0, places=6)
        np.testing.assert_array_equal(normalized[:, 1024], np.zeros(3))
        self.assertEqual(standard_deviation[1024], 1.0)

    def test_context_uses_only_prior_utterances_in_same_dialogue(self):
        features = np.zeros((4, cache_embeddings.FEATURE_DIMENSION), dtype=np.float32)
        features[0, :768] = 1.0
        features[1, :768] = 2.0
        features[2, :768] = 10.0
        features[3, :768] = 4.0
        metadata = {
            "dialogue_ids": ["a", "a", "b", "a"],
            "utterance_ids": ["0", "1", "0", "2"],
            "utterances": ["first", "second", "other", "third"],
        }

        contextual = train_mlp.add_text_context(features, metadata, window=3)

        self.assertEqual(contextual.shape, (4, 2304))
        base = cache_embeddings.FEATURE_DIMENSION
        np.testing.assert_array_equal(contextual[0, base:], np.zeros(768))
        np.testing.assert_array_equal(contextual[1, base:], np.ones(768))
        np.testing.assert_array_equal(contextual[2, base:], np.zeros(768))
        np.testing.assert_array_equal(
            contextual[3, base:], np.full(768, 1.5, dtype=np.float32)
        )

    def test_zero_context_window_preserves_features(self):
        features = np.ones((2, cache_embeddings.FEATURE_DIMENSION), dtype=np.float32)
        metadata = {
            "dialogue_ids": ["a", "a"],
            "utterance_ids": ["0", "1"],
            "utterances": ["first", "second"],
        }

        contextual = train_mlp.add_text_context(features, metadata, window=0)

        self.assertIs(contextual, features)

    def test_class_weights_are_inverse_frequency(self):
        label_ids = np.array([0, 0, 0, 1], dtype=np.int64)

        weights = train_mlp.compute_class_weights(label_ids, number_of_classes=2)

        torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.0]))

    def test_class_weights_reject_missing_training_class(self):
        with self.assertRaisesRegex(ValueError, "missing classes"):
            train_mlp.compute_class_weights(np.array([0, 0]), number_of_classes=2)

    def test_best_checkpoint_prefers_higher_macro_f1(self):
        self.assertTrue(train_mlp.is_better_checkpoint(0.31, 0.30))
        self.assertFalse(train_mlp.is_better_checkpoint(0.30, 0.30))


if __name__ == "__main__":
    unittest.main()
