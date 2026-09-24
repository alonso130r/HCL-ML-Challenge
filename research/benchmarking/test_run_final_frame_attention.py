import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_final_frame_attention.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("final_frame_attention", MODULE_PATH)
final_frame = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final_frame)


class FinalFrameAttentionTests(unittest.TestCase):
    def test_defaults_define_six_lightweight_cv_runs(self):
        args = final_frame.parse_arguments([])

        self.assertEqual(args.folds, 3)
        self.assertEqual(args.seeds, [43, 44])
        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(len(final_frame.cv_schedule(args)), 6)

    def test_selected_epoch_is_rounded_median(self):
        self.assertEqual(final_frame.selected_epoch([4, 7, 8, 10]), 8)
        self.assertEqual(final_frame.selected_epoch([5, 7, 9]), 7)

    def test_checkpoint_requires_audio_and_state_evidence(self):
        self.assertTrue(
            final_frame.checkpoint_is_eligible(0.64, 0.63, 0.003, 0.006, 0.007)
        )
        self.assertFalse(
            final_frame.checkpoint_is_eligible(0.64, 0.63, 0.003, 0.006, 0.001)
        )

    def test_selection_requires_four_eligible_runs(self):
        runs = [
            {"eligible": index < 4, "best_epoch": 7 + index, "warmup_epoch": 3}
            for index in range(6)
        ]

        selected = final_frame.select_epochs(runs, minimum_eligible=4)

        self.assertEqual(selected["eligible_runs"], 4)
        self.assertEqual(selected["warmup_epochs"], 3)


if __name__ == "__main__":
    unittest.main()
