import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("distill_frame_attention_ensemble.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("distill_frame_attention", MODULE_PATH)
distill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distill)


class DistillFrameAttentionTests(unittest.TestCase):
    def test_teacher_evaluation_accepts_explicit_evidence_thresholds(self):
        result = {
            "logits": {},
            "metrics": {"weighted_f1": 0.64, "macro_f1": 0.50},
            "text_metrics": {"weighted_f1": 0.63, "macro_f1": 0.49},
        }
        with (
            mock.patch.object(distill.frame, "make_loader", return_value=[]),
            mock.patch.object(distill, "collect_ensemble_logits", return_value=result),
            mock.patch.object(
                distill.recurrent,
                "different_label_audio_mapping",
                return_value={},
            ),
            mock.patch.object(
                distill.final_frame, "checkpoint_is_eligible", return_value=True
            ) as eligible,
        ):
            report = distill.evaluate_teacher_ensemble(
                [], [], SimpleNamespace(), torch.device("cpu"), 0.001, 0.005
            )

        self.assertTrue(report["eligible"])
        self.assertEqual(eligible.call_args.args[-2:], (0.001, 0.005))

    def test_cached_teacher_logits_do_not_require_cli_namespace_fields(self):
        records = [{}, {}]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "teacher_logits.npz"
            values = np.ones((2, 3), dtype=np.float32)
            np.savez_compressed(
                cache_path, matched=values, reset=values, zero=values
            )

            distill.attach_teacher_logits(
                records,
                [],
                SimpleNamespace(),
                torch.device("cpu"),
                cache_path,
                rebuild_teacher_cache=False,
            )

        self.assertIn("teacher_matched_logits", records[0])

    def test_distillation_loss_is_zero_for_matching_logits(self):
        logits = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        mask = torch.tensor([True, True])

        loss = distill.distillation_loss(logits, logits, mask, temperature=2.0)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_teacher_discovery_requires_six_eligible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (43, 44):
                for fold in (1, 2, 3):
                    run = root / "cv" / f"seed-{seed}" / f"fold-{fold}"
                    run.mkdir(parents=True)
                    (run / "metrics.json").write_text('{"eligible": true}')
                    (run / "best_frame_attention.pt").touch()

            checkpoints = distill.discover_teacher_checkpoints(root)

            self.assertEqual(len(checkpoints), 6)

    def test_distillation_collator_adds_teacher_logits(self):
        item = {
            "text_embeddings": np.ones((1, 3), dtype=np.float32),
            "text_logits": np.ones((1, 2), dtype=np.float32),
            "acoustic_features": np.ones((1, 4), dtype=np.float32),
            "emotion_frames": [np.ones((2, 5), dtype=np.float32)],
            "speaker_indices": np.array([0]),
            "labels": np.array([1]),
            "record_indices": np.array([0]),
            "teacher_matched_logits": np.ones((1, 2), dtype=np.float32),
            "teacher_reset_logits": np.ones((1, 2), dtype=np.float32),
            "teacher_zero_logits": np.ones((1, 2), dtype=np.float32),
        }

        batch = distill.collate_distillation_dialogues([item])

        self.assertEqual(batch["teacher_matched_logits"].shape, (1, 1, 2))
        self.assertEqual(batch["teacher_mask"].sum().item(), 1)

    def test_student_selection_rejects_ineligible_run(self):
        runs = [
            {"seed": 45, "eligible": False, "macro_f1": 0.55, "weighted_f1": 0.65},
            {"seed": 46, "eligible": True, "macro_f1": 0.50, "weighted_f1": 0.63},
        ]

        selected = distill.select_student(runs)

        self.assertEqual(selected["seed"], 46)


if __name__ == "__main__":
    unittest.main()
