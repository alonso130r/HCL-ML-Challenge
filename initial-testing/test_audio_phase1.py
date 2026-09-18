import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("audio_phase1.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("audio_phase1", MODULE_PATH)
audio_phase1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audio_phase1)


class AudioPhaseOneTests(unittest.TestCase):
    def test_summarize_sequence_preserves_mean_and_variation(self):
        sequence = np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)

        summary = audio_phase1.summarize_sequence(sequence)

        np.testing.assert_allclose(summary, np.array([2.0, 6.0, 1.0, 2.0]))

    def test_build_prosody_contours_adds_pitch_and_loudness_deltas(self):
        columns = {
            "F0semitoneFrom27.5Hz_sma3nz": np.array([0.0, 3.0, 5.0]),
            "Loudness_sma3": np.array([2.0, 4.0, 3.0]),
        }

        contours = audio_phase1.build_prosody_contours(columns)

        np.testing.assert_allclose(
            contours,
            np.array(
                [
                    [0.0, 2.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 3.0, 2.0, 1.0],
                    [5.0, 3.0, 2.0, -1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_causal_speaker_features_use_only_prior_same_speaker_turns(self):
        features = np.array([[1.0], [100.0], [3.0], [5.0]], dtype=np.float32)
        records = [
            {"dialogue_id": "1", "utterance_id": "0", "speaker": "A"},
            {"dialogue_id": "1", "utterance_id": "1", "speaker": "B"},
            {"dialogue_id": "1", "utterance_id": "2", "speaker": "A"},
            {"dialogue_id": "2", "utterance_id": "0", "speaker": "A"},
        ]

        relative, history = audio_phase1.causal_speaker_features(
            features,
            records,
            global_mean=np.array([10.0], dtype=np.float32),
            global_std=np.array([2.0], dtype=np.float32),
        )

        np.testing.assert_allclose(relative[:, 0], np.array([-4.5, 45.0, 1.0, -2.5]))
        np.testing.assert_array_equal(history, np.array([0.0, 0.0, 1.0, 0.0]))

    def test_combine_feature_views_has_expected_boundaries(self):
        emotion = np.ones((2, 6), dtype=np.float32)
        acoustics = np.full((2, 4), 2.0, dtype=np.float32)
        relative = np.full((2, 4), 3.0, dtype=np.float32)
        history = np.array([0.0, 2.0], dtype=np.float32)

        views = audio_phase1.combine_feature_views(
            emotion, acoustics, relative, history
        )

        self.assertEqual(views["emotion"].shape, (2, 6))
        self.assertEqual(views["hybrid"].shape, (2, 10))
        self.assertEqual(views["speaker_relative"].shape, (2, 15))
        np.testing.assert_array_equal(views["speaker_relative"][:, -1], history)

    def test_directory_only_path_removes_files_and_missing_entries(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "python"
            file_path.write_text("", encoding="utf-8")
            value = f"{directory}:{file_path}:{file_path}.missing"

            cleaned = audio_phase1.directory_only_path(value)

            self.assertEqual(cleaned, directory)


if __name__ == "__main__":
    unittest.main()
