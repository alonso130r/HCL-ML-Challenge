import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("train_text_audio_binned.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_text_audio_binned", MODULE_PATH)
binned = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binned)


class BinnedTextAudioTests(unittest.TestCase):
    def test_pool_sequence_bins_preserves_order_and_statistics(self):
        sequence = np.arange(8, dtype=np.float32)[:, None]

        pooled, valid = binned.pool_sequence_bins(sequence, 4)

        self.assertEqual(pooled.shape, (4, 3))
        np.testing.assert_allclose(
            pooled,
            np.array(
                [[0.5, 0.5, 1.0], [2.5, 0.5, 3.0],
                 [4.5, 0.5, 5.0], [6.5, 0.5, 7.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))

    def test_pool_sequence_bins_marks_empty_bins(self):
        pooled, valid = binned.pool_sequence_bins(
            np.array([[2.0], [6.0]], dtype=np.float32), 4
        )

        np.testing.assert_array_equal(valid, np.array([True, False, True, False]))
        np.testing.assert_array_equal(pooled[~valid], 0.0)

    def test_pool_text_bins_uses_only_current_tokens(self):
        hidden = torch.tensor([[[100.0], [1.0], [3.0], [5.0], [200.0]]])
        current = torch.tensor([[False, True, True, True, False]])

        pooled, valid = binned.pool_text_bins(hidden, current, 2)

        torch.testing.assert_close(pooled, torch.tensor([[[2.0], [5.0]]]))
        torch.testing.assert_close(valid, torch.tensor([[True, True]]))

    def test_shift_bins_rolls_local_audio_but_not_global_features(self):
        state = {
            "bins": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            "valid": torch.tensor([[True, True, False, True]]),
            "global": torch.tensor([[9.0]]),
        }

        shifted = binned.shift_audio_state(state, 2)

        torch.testing.assert_close(
            shifted["bins"], torch.tensor([[[3.0], [4.0], [1.0], [2.0]]])
        )
        torch.testing.assert_close(
            shifted["valid"], torch.tensor([[False, True, True, True]])
        )
        self.assertIs(shifted["global"], state["global"])

    def test_temporal_score_requires_matched_to_beat_all_controls(self):
        matched = {"weighted_f1": 0.640, "macro_f1": 0.490}
        shifted = {"weighted_f1": 0.634, "macro_f1": 0.480}
        shuffled = [{"weighted_f1": 0.632, "macro_f1": 0.470}]
        zeroed = {"weighted_f1": 0.630, "macro_f1": 0.475}

        score = binned.temporal_checkpoint_score(
            matched, shifted, shuffled, zeroed, minimum_margin=0.005
        )

        self.assertEqual(score[0], 1)
        self.assertAlmostEqual(score[1], 0.006)


if __name__ == "__main__":
    unittest.main()
