#!/usr/bin/env python3
"""Lightweight leakage-safe OOF evaluation of frame-attention distillation.

This runner never reads the MELD test split. For each of three development
folds it trains one teacher and one student, using a different development
fold for checkpoint selection and evaluating once on the untouched outer fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKING = Path(__file__).resolve().parent
INITIAL_TESTING = ROOT / "initial-testing"
sys.path.insert(0, str(BENCHMARKING))
sys.path.insert(0, str(INITIAL_TESTING))

import train_recurrent_dialogue as recurrent  # noqa: E402
import train_recurrent_dialogue_frame_attention as frame  # noqa: E402
import train_recurrent_dialogue_stabilized as stabilized  # noqa: E402
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device  # noqa: E402
from run_final_architecture_cv import assign_dialogue_folds  # noqa: E402

import distill_frame_attention_ensemble as distill  # noqa: E402
import run_final_frame_attention as final_frame  # noqa: E402
import run_final_training as final_base  # noqa: E402


PROTOCOL_VERSION = "light-distillation-oof-v1"


def build_rotation_split(records, assignments, outer_fold):
    folds = sorted(set(assignments.values()))
    if folds != [0, 1, 2] or outer_fold not in folds:
        raise ValueError("rotation requires exactly three folds")
    inner_fold = (outer_fold + 1) % 3
    fit_fold = (outer_fold + 2) % 3

    def dev_fold(fold):
        ids = {key for key, value in assignments.items() if value == fold}
        return final_base.select_dialogues(records["dev"], ids)

    return {
        "train": final_base.copy_records(records["train"] + dev_fold(fit_fold)),
        "inner": dev_fold(inner_fold),
        "outer": dev_fold(outer_fold),
        "inner_fold": inner_fold,
        "fit_fold": fit_fold,
    }


def summarize_oof(folds):
    actual = sum((fold["matched"]["actual"] for fold in folds), [])
    matched = sum((fold["matched"]["predicted"] for fold in folds), [])
    text = sum((fold["matched"]["text_predicted"] for fold in folds), [])
    reset = sum((fold["reset"] for fold in folds), [])
    zero = sum((fold["zero"] for fold in folds), [])
    shuffle_count = len(folds[0]["shuffled"])
    shuffled = [
        sum((fold["shuffled"][index] for fold in folds), [])
        for index in range(shuffle_count)
    ]
    matched_metrics = compute_metrics(actual, matched)
    reset_metrics = compute_metrics(actual, reset)
    zero_metrics = compute_metrics(actual, zero)
    shuffled_metrics = [compute_metrics(actual, values) for values in shuffled]
    weighted = matched_metrics["weighted_f1"]
    return {
        "examples": len(actual),
        "matched": matched_metrics,
        "text": compute_metrics(actual, text),
        "reset": reset_metrics,
        "zero_audio": zero_metrics,
        "shuffled_audio": shuffled_metrics,
        "state_margin": weighted - reset_metrics["weighted_f1"],
        "zero_audio_margin": weighted - zero_metrics["weighted_f1"],
        "audio_margin": weighted
        - max(item["weighted_f1"] for item in shuffled_metrics),
    }


def prediction_bundle(controls, records):
    matched = controls["matched"]
    text_by_index = {
        index: EMOTION_LABELS[int(np.argmax(records[index]["text_logits"]))]
        for index in matched["indices"]
    }
    return {
        "matched": {
            "actual": matched["actual"],
            "predicted": matched["predicted"],
            "text_predicted": [text_by_index[index] for index in matched["indices"]],
            "indices": matched["indices"],
        },
        "reset": controls["reset"]["predicted"],
        "zero": controls["zero"]["predicted"],
        "shuffled": [item["predicted"] for item in controls["shuffled"]],
    }


def write_predictions(path, folds):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("outer_fold", "expected", "predicted", "text_predicted"),
        )
        writer.writeheader()
        for fold_index, fold in enumerate(folds, start=1):
            matched = fold["matched"]
            for actual, predicted, text in zip(
                matched["actual"],
                matched["predicted"],
                matched["text_predicted"],
            ):
                writer.writerow(
                    {
                        "outer_fold": fold_index,
                        "expected": actual,
                        "predicted": predicted,
                        "text_predicted": text,
                    }
                )


def load_model(checkpoint, records, run_args, device):
    model = distill.create_model(records, run_args, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_fold(model, records, run_args, device):
    controls = final_frame.evaluate_controls(model, records, run_args, device)
    bundle = prediction_bundle(controls, records)
    metrics = controls["matched"]["metrics"]
    text_metrics = controls["matched"]["text_metrics"]
    bundle["report"] = {
        "matched": metrics,
        "text": text_metrics,
        "state_margin": controls["state_margin"],
        "zero_audio_margin": controls["zero_audio_margin"],
        "audio_margin": controls["audio_margin"],
        "eligible": final_frame.checkpoint_is_eligible(
            metrics["weighted_f1"],
            text_metrics["weighted_f1"],
            controls["state_margin"],
            controls["zero_audio_margin"],
            controls["audio_margin"],
        ),
    }
    return bundle


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        default=ROOT / "initial-testing/training-output-text/best-model",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=ROOT / "initial-testing/audio-cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarking/results/frame-attention-distillation-oof-light",
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--fold-seed", type=int, default=20260920)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dialogue-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-audio-frames", type=int)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--control-distillation-weight", type=float, default=0.25)
    parser.add_argument("--minimum-state-margin", type=float, default=0.002)
    parser.add_argument("--minimum-audio-margin", type=float, default=0.005)
    parser.add_argument("--retrain", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.audio_warmup_epochs < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("epoch counts and patience must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("learning rate and temperature must be positive")
    for path in (args.raw_archive, args.text_model, args.audio_cache_dir):
        if not path.exists():
            raise FileNotFoundError(path)


def run_fold(outer_fold, split, args, device):
    fold_dir = args.output_dir / f"outer-fold-{outer_fold + 1}"
    teacher_dir = fold_dir / "teacher"
    teacher_checkpoint = teacher_dir / "best_frame_attention.pt"
    teacher_metrics = teacher_dir / "metrics.json"
    teacher_args = argparse.Namespace(**vars(args))
    teacher_args.output_dir = fold_dir
    teacher_args.seeds = [args.seed]
    teacher_args.folds = 3

    if args.retrain or not (teacher_checkpoint.is_file() and teacher_metrics.is_file()):
        final_frame.train_fold(
            {"train": split["train"], "dev": split["inner"]},
            teacher_args,
            device,
            args.seed,
            outer_fold,
            teacher_dir,
        )

    run_args = final_frame.experiment_args(teacher_args, fold_dir, args.seed)
    for key in ("temperature", "distillation_weight", "control_distillation_weight"):
        setattr(run_args, key, getattr(args, key))
    teacher = load_model(teacher_checkpoint, split["train"], run_args, device)
    cache_path = fold_dir / "teacher_train_logits.npz"
    distill.attach_teacher_logits(
        split["train"],
        [teacher],
        run_args,
        device,
        cache_path,
        rebuild_teacher_cache=args.retrain,
    )
    del teacher
    if device.type == "mps":
        torch.mps.empty_cache()

    student_args = argparse.Namespace(**vars(args))
    student_args.output_dir = fold_dir
    student_args.student_seeds = [args.seed]
    student_report = distill.train_student(
        args.seed, split["train"], split["inner"], student_args, device
    )
    student = load_model(student_report["checkpoint"], split["train"], run_args, device)
    result = evaluate_fold(student, split["outer"], run_args, device)
    result["report"].update(
        {
            "outer_fold": outer_fold + 1,
            "inner_fold": split["inner_fold"] + 1,
            "fit_fold": split["fit_fold"] + 1,
            "teacher_checkpoint": str(teacher_checkpoint.resolve()),
            "student_checkpoint": student_report["checkpoint"],
            "student_inner_selection": student_report,
        }
    )
    (fold_dir / "outer_metrics.json").write_text(
        json.dumps(result["report"], indent=2), encoding="utf-8"
    )
    print(
        f"Outer fold {outer_fold + 1}: "
        f"weighted={result['report']['matched']['weighted_f1']:.4f} "
        f"macro={result['report']['matched']['macro_f1']:.4f} "
        f"state={result['report']['state_margin']:+.4f} "
        f"zero={result['report']['zero_audio_margin']:+.4f} "
        f"shuffle={result['report']['audio_margin']:+.4f}",
        flush=True,
    )
    return result


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    started = time.perf_counter()
    device = select_device(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preparation_args = final_frame.experiment_args(args, args.output_dir, args.seed)
    records, normalization = stabilized.prepare_records(preparation_args, device)
    frame.attach_emotion_frames(records, args.max_audio_frames)
    records = final_base.renumber_dialogues(records)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0],
        std=normalization[1],
    )
    assignments = assign_dialogue_folds(records["dev"], 3, args.fold_seed)
    folds = []
    for outer_fold in range(3):
        split = build_rotation_split(records, assignments, outer_fold)
        folds.append(run_fold(outer_fold, split, args, device))
    summary = summarize_oof(folds)
    fold_reports = [fold["report"] for fold in folds]
    summary["protocol_version"] = PROTOCOL_VERSION
    summary["folds"] = fold_reports
    summary["eligible_folds"] = sum(fold["eligible"] for fold in fold_reports)
    summary["runtime_seconds"] = time.perf_counter() - started
    (args.output_dir / "oof_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_predictions(args.output_dir / "oof_predictions.csv", folds)
    print(
        f"OOF weighted F1: {summary['matched']['weighted_f1']:.4f}; "
        f"macro F1: {summary['matched']['macro_f1']:.4f}; "
        f"text weighted F1: {summary['text']['weighted_f1']:.4f}"
    )
    print(
        f"OOF state={summary['state_margin']:+.4f} "
        f"zero={summary['zero_audio_margin']:+.4f} "
        f"shuffle={summary['audio_margin']:+.4f}; "
        f"eligible folds={summary['eligible_folds']}/3"
    )
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
