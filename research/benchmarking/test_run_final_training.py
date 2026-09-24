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

    def test_choose_configuration_applies_macro_tolerance_then_penalizes_variance(self):
        candidates = [
            {"learning_rate": 1e-4, "weighted_f1": 0.630, "weighted_f1_std": 0.001, "macro_f1": 0.500, "state_margin": 0.003},
            {"learning_rate": 2e-4, "weighted_f1": 0.640, "weighted_f1_std": 0.020, "macro_f1": 0.494, "state_margin": 0.004},
            {"learning_rate": 3e-4, "weighted_f1": 0.635, "weighted_f1_std": 0.010, "macro_f1": 0.496, "state_margin": 0.005},
        ]

        selected = final_training.choose_configuration(candidates, macro_tolerance=0.005)

        self.assertEqual(selected["learning_rate"], 1e-4)

    def test_choose_configuration_rejects_missing_state_evidence(self):
        candidates = [
            {"learning_rate": 1e-4, "weighted_f1": 0.630, "weighted_f1_std": 0.001, "macro_f1": 0.500, "state_margin": 0.004},
            {"learning_rate": 2e-4, "weighted_f1": 0.650, "weighted_f1_std": 0.001, "macro_f1": 0.510, "state_margin": -0.001},
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

    def test_default_plan_is_three_repeats_by_five_folds_and_three_rates(self):
        args = final_training.parse_arguments([])

        self.assertEqual(args.folds, 5)
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.learning_rates, [1e-4, 2e-4, 3e-4])
        self.assertEqual(args.max_epochs, 25)

    def test_aggregate_reports_population_variance_worst_case_and_per_class_means(self):
        reports = [
            {
                "best_epoch": 4,
                "weighted_f1": 0.60,
                "macro_f1": 0.40,
                "state_margin": 0.01,
                "zero_audio_margin": 0.02,
                "audio_margin": 0.03,
                "per_class_f1": {"neutral": 0.8, "sadness": 0.2},
            },
            {
                "best_epoch": 6,
                "weighted_f1": 0.64,
                "macro_f1": 0.44,
                "state_margin": 0.02,
                "zero_audio_margin": 0.04,
                "audio_margin": 0.05,
                "per_class_f1": {"neutral": 0.6, "sadness": 0.4},
            },
        ]

        result = final_training.aggregate_reports(2e-4, reports)

        self.assertAlmostEqual(result["weighted_f1"], 0.62)
        self.assertAlmostEqual(result["weighted_f1_std"], 0.02)
        self.assertEqual(result["weighted_f1_min"], 0.60)
        self.assertEqual(result["macro_f1_min"], 0.40)
        self.assertAlmostEqual(result["per_class_f1"]["neutral"], 0.7)
        self.assertAlmostEqual(result["per_class_f1"]["sadness"], 0.3)
        self.assertEqual(result["selected_epoch"], 5)

    def test_cache_metadata_requires_exact_protocol_inputs(self):
        expected = {
            "protocol_version": "robust-dev-cv-v2",
            "learning_rate": 2e-4,
            "repeat": 2,
            "fold": 4,
            "fold_seed": 20260920,
            "train_seed": 44,
            "folds": 5,
            "repeats": 3,
            "max_epochs": 25,
            "patience": 4,
            "context_window": 2,
        }
        report = {"cache_metadata": dict(expected)}

        final_training.validate_cached_report(report, expected)
        report["cache_metadata"]["train_seed"] = 45
        with self.assertRaisesRegex(ValueError, "train_seed"):
            final_training.validate_cached_report(report, expected)

    def test_only_complete_matching_final_report_is_reusable(self):
        expected = {"protocol_version": "robust-dev-cv-v2", "learning_rate": 2e-4}

        self.assertFalse(final_training.is_completed_final_report({}, expected))
        self.assertFalse(
            final_training.is_completed_final_report(
                {"cache_metadata": expected}, expected
            )
        )
        self.assertTrue(
            final_training.is_completed_final_report(
                {
                    "cache_metadata": expected,
                    "test_recurrent_matched": {"weighted_f1": 0.62},
                },
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
