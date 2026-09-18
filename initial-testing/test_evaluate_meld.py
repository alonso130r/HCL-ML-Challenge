import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_meld.py")
SPEC = importlib.util.spec_from_file_location("evaluate_meld", MODULE_PATH)
evaluate_meld = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_meld)


class EvaluateMeldTests(unittest.TestCase):
    def test_seeded_sample_is_reproducible_and_sorted_by_source_row(self):
        rows = [{"Sr No.": str(index)} for index in range(20)]

        first = evaluate_meld.seeded_sample(rows, sample_size=6, seed=42)
        second = evaluate_meld.seeded_sample(rows, sample_size=6, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            [int(row["Sr No."]) for row in first],
            sorted(int(row["Sr No."]) for row in first),
        )

    def test_seeded_sample_rejects_oversized_request(self):
        with self.assertRaisesRegex(ValueError, "exceeds the 2 available"):
            evaluate_meld.seeded_sample([{}, {}], sample_size=3, seed=42)

    def test_media_member_candidates_include_both_meld_naming_patterns(self):
        row = {"Dialogue_ID": "48", "Utterance_ID": "3"}

        self.assertEqual(
            evaluate_meld.media_member_candidates(row),
            (
                "output_repeated_splits_test/dia48_utt3.mp4",
                "output_repeated_splits_test/final_videos_testdia48_utt3.mp4",
            ),
        )

    def test_compute_metrics_uses_fixed_meld_label_order(self):
        actual = ["neutral", "joy", "anger", "anger"]
        predicted = ["neutral", "anger", "anger", "joy"]

        result = evaluate_meld.compute_metrics(actual, predicted)

        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["labels"], evaluate_meld.EMOTION_LABELS)
        self.assertEqual(result["support"]["anger"], 2)
        self.assertEqual(result["support"]["fear"], 0)

    def test_write_predictions_preserves_required_columns(self):
        record = {
            "dialogue_id": "1",
            "utterance_id": "2",
            "utterance": "Hello",
            "expected": "neutral",
            "predicted": "joy",
            "confidence": 0.75,
            "latency_ms": 12.5,
            "status": "ok",
            "error": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "predictions.csv"
            evaluate_meld.write_predictions([record], destination)
            with destination.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(rows[0]["predicted"], "joy")
        self.assertEqual(tuple(rows[0]), evaluate_meld.PREDICTION_COLUMNS)


if __name__ == "__main__":
    unittest.main()
