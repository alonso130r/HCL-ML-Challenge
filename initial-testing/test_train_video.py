import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("train_video.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_video", MODULE_PATH)
train_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_video)


class VideoTrainingTests(unittest.TestCase):
    def test_sample_frame_indices_are_even_and_include_endpoints(self):
        self.assertEqual(train_video.sample_frame_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(train_video.sample_frame_indices(3, 8), [0, 1, 2])

    def test_select_face_uses_largest_detection(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[20:70, 40:100] = 255

        crop = train_video.select_face(image, [(5, 5, 10, 10), (40, 20, 60, 50)])

        self.assertGreater(crop.shape[0], 50)
        self.assertGreater(crop.shape[1], 60)
        self.assertGreater(float(crop.mean()), 100.0)

    def test_select_face_falls_back_to_center_square(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)

        crop = train_video.select_face(image, [])

        self.assertEqual(crop.shape, (60, 60, 3))

    def test_pad_video_embeddings_returns_padding_mask(self):
        sequences = [np.ones((2, 3), dtype=np.float32), np.full((1, 3), 2.0, dtype=np.float32)]

        padded, mask = train_video.pad_video_embeddings(sequences)

        self.assertEqual(tuple(padded.shape), (2, 2, 3))
        torch.testing.assert_close(mask, torch.tensor([[False, False], [False, True]]))
        torch.testing.assert_close(padded[1, 1], torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
