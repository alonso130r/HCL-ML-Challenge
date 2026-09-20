import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "evaluate_distilled_frame_attention_light.py"
)
SPEC = importlib.util.spec_from_file_location("light_oof", MODULE_PATH)
light_oof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(light_oof)


class LightweightOutOfFoldTests(unittest.TestCase):
    def test_rotation_keeps_outer_and_inner_dialogues_out_of_training(self):
        records = {
            "train": [{"dialogue_id": "base", "label": "neutral"}],
            "dev": [
                {"dialogue_id": "a", "label": "neutral"},
                {"dialogue_id": "b", "label": "joy"},
                {"dialogue_id": "c", "label": "sadness"},
            ],
        }
        assignments = {"a": 0, "b": 1, "c": 2}

        split = light_oof.build_rotation_split(records, assignments, outer_fold=0)

        self.assertEqual(split["inner_fold"], 1)
        self.assertEqual(split["fit_fold"], 2)
        self.assertEqual(
            {row["dialogue_id"] for row in split["train"]}, {"base", "c"}
        )
        self.assertEqual(
            {row["dialogue_id"] for row in split["inner"]}, {"b"}
        )
        self.assertEqual(
            {row["dialogue_id"] for row in split["outer"]}, {"a"}
        )

    def test_oof_summary_uses_concatenated_predictions(self):
        folds = [
            {
                "matched": {
                    "actual": ["neutral", "joy"],
                    "predicted": ["neutral", "neutral"],
                    "text_predicted": ["neutral", "joy"],
                },
                "reset": ["neutral", "neutral"],
                "zero": ["neutral", "neutral"],
                "shuffled": [["neutral", "neutral"]],
            },
            {
                "matched": {
                    "actual": ["joy"],
                    "predicted": ["joy"],
                    "text_predicted": ["neutral"],
                },
                "reset": ["neutral"],
                "zero": ["neutral"],
                "shuffled": [["neutral"]],
            },
        ]

        summary = light_oof.summarize_oof(folds)

        self.assertEqual(summary["examples"], 3)
        self.assertAlmostEqual(summary["matched"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["text"]["accuracy"], 2 / 3)
        self.assertGreater(summary["state_margin"], 0)


if __name__ == "__main__":
    unittest.main()
