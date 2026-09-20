import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_final_training.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("final_training", MODULE_PATH)
final_training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final_training)


class FinalTrainingTests(unittest.TestCase):
    def test_renumbered_dialogues_are_unique_across_source_splits(self):
        records = {
            "train": [
                {"dialogue_id": "0", "utterance_id": "0"},
                {"dialogue_id": "0", "utterance_id": "1"},
                {"dialogue_id": "1", "utterance_id": "0"},
            ],
            "dev": [
                {"dialogue_id": "0", "utterance_id": "0"},
                {"dialogue_id": "2", "utterance_id": "0"},
            ],
        }

        result = final_training.renumber_dialogues(records)

        self.assertEqual(
            [row["dialogue_id"] for row in result["train"]], ["0", "0", "1"]
        )
        self.assertEqual(
            [row["dialogue_id"] for row in result["dev"]], ["2", "3"]
        )
        self.assertEqual(records["dev"][0]["dialogue_id"], "0")

    def test_choose_configuration_applies_macro_tolerance_before_weighted_f1(self):
        candidates = [
            {"learning_rate": 1e-4, "weighted_f1": 0.630, "macro_f1": 0.500, "state_margin": 0.003},
            {"learning_rate": 2e-4, "weighted_f1": 0.640, "macro_f1": 0.494, "state_margin": 0.004},
            {"learning_rate": 3e-4, "weighted_f1": 0.635, "macro_f1": 0.496, "state_margin": 0.005},
        ]

        selected = final_training.choose_configuration(candidates, macro_tolerance=0.005)

        self.assertEqual(selected["learning_rate"], 3e-4)

    def test_choose_configuration_rejects_missing_state_evidence(self):
        candidates = [
            {"learning_rate": 1e-4, "weighted_f1": 0.630, "macro_f1": 0.500, "state_margin": 0.004},
            {"learning_rate": 2e-4, "weighted_f1": 0.650, "macro_f1": 0.510, "state_margin": -0.001},
        ]

        selected = final_training.choose_configuration(
            candidates, macro_tolerance=0.005, minimum_state_margin=0.002
        )

        self.assertEqual(selected["learning_rate"], 1e-4)

    def test_cv_split_holds_out_only_development_dialogues(self):
        records = {
            "train": [
                {"dialogue_id": "0", "record_index": 0},
                {"dialogue_id": "1", "record_index": 1},
            ],
            "dev": [
                {"dialogue_id": "2", "record_index": 0},
                {"dialogue_id": "3", "record_index": 1},
                {"dialogue_id": "4", "record_index": 2},
            ],
        }
        assignments = {"2": 0, "3": 1, "4": 1}

        split = final_training.build_cv_split(records, assignments, fold=0)

        self.assertEqual(
            {row["dialogue_id"] for row in split["train"]}, {"0", "1", "3", "4"}
        )
        self.assertEqual(
            {row["dialogue_id"] for row in split["dev"]}, {"2"}
        )
        self.assertEqual(
            {row["dialogue_id"] for row in split["test"]}, {"2"}
        )

    def test_final_epoch_is_rounded_median_of_selected_folds(self):
        self.assertEqual(final_training.selected_epoch([7, 10, 12]), 10)
        self.assertEqual(final_training.selected_epoch([7, 8]), 8)

    def test_default_plan_is_three_folds_by_three_learning_rates(self):
        args = final_training.parse_arguments([])

        self.assertEqual(args.folds, 3)
        self.assertEqual(args.learning_rates, [1e-4, 2e-4, 3e-4])


if __name__ == "__main__":
    unittest.main()
