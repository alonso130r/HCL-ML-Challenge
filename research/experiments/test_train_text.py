import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("train_text.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("train_text", MODULE_PATH)
train_text = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_text)


class TextTrainingTests(unittest.TestCase):
    def test_build_examples_adds_speaker_tagged_prior_context_only(self):
        rows = [
            {"Dialogue_ID": "1", "Utterance_ID": "0", "Speaker": "A", "Utterance": "First", "Emotion": "neutral"},
            {"Dialogue_ID": "1", "Utterance_ID": "1", "Speaker": "B", "Utterance": "Second", "Emotion": "joy"},
            {"Dialogue_ID": "2", "Utterance_ID": "0", "Speaker": "C", "Utterance": "Other", "Emotion": "anger"},
            {"Dialogue_ID": "1", "Utterance_ID": "2", "Speaker": "A", "Utterance": "Third", "Emotion": "sadness"},
        ]

        examples = train_text.build_examples(rows, context_window=2)

        self.assertEqual(examples[0]["text"], "Current:\n[A] First")
        self.assertEqual(
            examples[1]["text"],
            "Context:\n[A] First\nCurrent:\n[B] Second",
        )
        self.assertEqual(examples[2]["text"], "Current:\n[C] Other")
        self.assertEqual(
            examples[3]["text"],
            "Context:\n[A] First\n[B] Second\nCurrent:\n[A] Third",
        )
        self.assertEqual(examples[3]["label"], "sadness")

    def test_build_examples_rejects_negative_context_window(self):
        with self.assertRaisesRegex(ValueError, "context window"):
            train_text.build_examples([], context_window=-1)

    def test_sqrt_class_weights_are_less_extreme_than_inverse_frequency(self):
        labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)

        weights = train_text.sqrt_class_weights(labels, number_of_classes=2)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertLess(float(weights[1] / weights[0]), 4.0)


if __name__ == "__main__":
    unittest.main()
