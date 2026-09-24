#!/usr/bin/env python3
"""Distill six eligible frame-attention teachers into one deployable student."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
INITIAL_TESTING = ROOT / "research/experiments"
sys.path.insert(0, str(INITIAL_TESTING))

import train_recurrent_dialogue as recurrent  # noqa: E402
import train_recurrent_dialogue_frame_attention as frame  # noqa: E402
import train_recurrent_dialogue_stabilized as stabilized  # noqa: E402
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device  # noqa: E402
from run_final_architecture_cv import assign_dialogue_folds  # noqa: E402
from train_text import sqrt_class_weights  # noqa: E402

import run_final_frame_attention as final_frame  # noqa: E402
import run_final_training as final_base  # noqa: E402


def distillation_loss(student_logits, teacher_logits, mask, temperature):
    student = student_logits[mask] / temperature
    teacher = teacher_logits[mask] / temperature
    return temperature**2 * torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=-1),
        torch.softmax(teacher, dim=-1),
        reduction="batchmean",
    )


def discover_teacher_checkpoints(root):
    checkpoints = []
    for metrics_path in sorted((root / "cv").glob("seed-*/fold-*/metrics.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = metrics_path.with_name("best_frame_attention.pt")
        if report.get("eligible") and checkpoint.is_file():
            checkpoints.append(checkpoint)
    if len(checkpoints) != 6:
        raise RuntimeError(
            f"expected six eligible teacher checkpoints, found {len(checkpoints)}"
        )
    return checkpoints


class DistillationDataset(frame.FrameDialogueDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        dialogue = self.dialogues[index]
        for key in (
            "teacher_matched_logits",
            "teacher_reset_logits",
            "teacher_zero_logits",
        ):
            result[key] = np.stack([record[key] for record in dialogue])
        return result


def collate_distillation_dialogues(items):
    result = frame.collate_frame_dialogues(items)
    batch_size, turns = result["valid_mask"].shape
    classes = items[0]["teacher_matched_logits"].shape[1]
    for key in (
        "teacher_matched_logits",
        "teacher_reset_logits",
        "teacher_zero_logits",
    ):
        values = torch.zeros(batch_size, turns, classes)
        for row, item in enumerate(items):
            size = len(item["labels"])
            values[row, :size] = torch.as_tensor(item[key])
        result[key] = values
    result["teacher_mask"] = result["valid_mask"].clone()
    return result


def make_distillation_loader(records, args, training):
    return torch.utils.data.DataLoader(
        DistillationDataset(records),
        batch_size=args.dialogue_batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_distillation_dialogues,
        num_workers=0,
    )


def select_student(runs):
    eligible = [run for run in runs if run["eligible"]]
    if not eligible:
        raise RuntimeError("no distilled student retained audio and state evidence")
    return max(
        eligible,
        key=lambda run: (run["macro_f1"], run["weighted_f1"]),
    )


def create_model(records, args, device):
    sample = records[0]
    return frame.FrameAttentionRecurrentModel(
        len(sample["text_embedding"]),
        sample["emotion_frames"].shape[1],
        len(sample["speaker_acoustic"]),
        len(EMOTION_LABELS),
        args,
    ).to(device)


def load_teachers(checkpoints, records, args, device):
    teachers = []
    for checkpoint in checkpoints:
        model = create_model(records, args, device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        teachers.append(model)
    return teachers


def collect_ensemble_logits(
    teachers, loader, device, reset_each_turn=False, zero_audio=False
):
    logits_by_index = {}
    actual, predicted, text_predicted = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = recurrent.move_batch(batch, device)
            outputs = [
                model(
                    batch,
                    reset_each_turn=reset_each_turn,
                    zero_audio=zero_audio,
                )
                for model in teachers
            ]
            logits = torch.stack([item["logits"] for item in outputs]).mean(0)
            mask = batch["valid_mask"]
            indices = batch["record_indices"][mask].cpu().tolist()
            values = logits[mask].float().cpu().numpy()
            for index, value in zip(indices, values):
                logits_by_index[index] = value
            actual.extend(batch["labels"][mask].cpu().tolist())
            predicted.extend(logits[mask].argmax(dim=-1).cpu().tolist())
            text_predicted.extend(
                batch["text_logits"][mask].argmax(dim=-1).cpu().tolist()
            )
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    return {
        "logits": logits_by_index,
        "metrics": compute_metrics(names(actual), names(predicted)),
        "text_metrics": compute_metrics(names(actual), names(text_predicted)),
    }


def evaluate_teacher_ensemble(
    teachers,
    records,
    model_args,
    device,
    minimum_state_margin,
    minimum_audio_margin,
):
    loader = frame.make_loader(records, model_args, False)
    matched = collect_ensemble_logits(teachers, loader, device)
    reset = collect_ensemble_logits(
        teachers, loader, device, reset_each_turn=True
    )
    zero = collect_ensemble_logits(teachers, loader, device, zero_audio=True)
    shuffled_loader = frame.make_loader(
        records,
        model_args,
        False,
        recurrent.different_label_audio_mapping(records, 43),
    )
    shuffled = collect_ensemble_logits(teachers, shuffled_loader, device)
    weighted = matched["metrics"]["weighted_f1"]
    report = {
        "matched": matched["metrics"],
        "text": matched["text_metrics"],
        "reset": reset["metrics"],
        "zero_audio": zero["metrics"],
        "shuffled_audio": shuffled["metrics"],
        "state_margin": weighted - reset["metrics"]["weighted_f1"],
        "zero_audio_margin": weighted - zero["metrics"]["weighted_f1"],
        "audio_margin": weighted - shuffled["metrics"]["weighted_f1"],
    }
    report["eligible"] = final_frame.checkpoint_is_eligible(
        weighted,
        matched["text_metrics"]["weighted_f1"],
        report["state_margin"],
        report["zero_audio_margin"],
        report["audio_margin"],
        minimum_state_margin,
        minimum_audio_margin,
    )
    return report


def attach_teacher_logits(
    records,
    teachers,
    model_args,
    device,
    cache_path,
    rebuild_teacher_cache=False,
):
    if cache_path.is_file() and not rebuild_teacher_cache:
        with np.load(cache_path) as cache:
            arrays = {key: cache[key] for key in ("matched", "reset", "zero")}
    else:
        loader = frame.make_loader(records, model_args, False)
        collected = {
            "matched": collect_ensemble_logits(teachers, loader, device),
            "reset": collect_ensemble_logits(
                teachers, loader, device, reset_each_turn=True
            ),
            "zero": collect_ensemble_logits(
                teachers, loader, device, zero_audio=True
            ),
        }
        arrays = {
            name: np.stack(
                [result["logits"][index] for index in range(len(records))]
            ).astype(np.float32)
            for name, result in collected.items()
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **arrays)
    if any(len(values) != len(records) for values in arrays.values()):
        raise ValueError("teacher-logit cache does not match training records")
    for index, record in enumerate(records):
        record["teacher_matched_logits"] = arrays["matched"][index]
        record["teacher_reset_logits"] = arrays["reset"][index]
        record["teacher_zero_logits"] = arrays["zero"][index]


def train_distillation_epoch(model, loader, optimizer, weights, args, device):
    model.train()
    rows = []
    for batch in loader:
        batch = recurrent.move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        matched = model(batch)
        model.eval()
        clean = model(batch)
        reset = model(batch, reset_each_turn=True)
        zero = model(batch, zero_audio=True)
        shuffled = model(frame.shuffle_frame_audio(batch))
        model.train()
        base_loss, components = stabilized.calculate_loss(
            matched, clean, reset, shuffled, batch, weights, args
        )
        mask = batch["teacher_mask"]
        matched_kd = distillation_loss(
            matched["logits"],
            batch["teacher_matched_logits"],
            mask,
            args.temperature,
        )
        reset_kd = distillation_loss(
            reset["logits"],
            batch["teacher_reset_logits"],
            mask,
            args.temperature,
        )
        zero_kd = distillation_loss(
            zero["logits"],
            batch["teacher_zero_logits"],
            mask,
            args.temperature,
        )
        loss = (
            base_loss
            + args.distillation_weight * matched_kd
            + args.control_distillation_weight * (reset_kd + zero_kd)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows.append(
            components
            | {
                "distillation": float(matched_kd.detach().cpu()),
                "reset_distillation": float(reset_kd.detach().cpu()),
                "zero_distillation": float(zero_kd.detach().cpu()),
                "distilled_total": float(loss.detach().cpu()),
            }
        )
    return {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }


def train_student(seed, train_records, dev_records, args, device):
    run_dir = args.output_dir / "students" / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_args = final_frame.experiment_args(args, run_dir, seed)
    for key in (
        "temperature",
        "distillation_weight",
        "control_distillation_weight",
    ):
        setattr(run_args, key, getattr(args, key))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = create_model(train_records, run_args, device)
    warmup_loader = frame.make_loader(train_records, run_args, True)
    dev_loader = frame.make_loader(dev_records, run_args, False)
    labels = np.asarray(
        [EMOTION_LABELS.index(record["label"]) for record in train_records]
    )
    weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    warmup = frame.warm_up_audio(
        model, warmup_loader, dev_loader, weights, run_args, device
    )
    train_loader = make_distillation_loader(train_records, run_args, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    checkpoint = run_dir / "best_distilled_frame_attention.pt"
    history, best, best_score, stale = [], None, None, 0
    for epoch in range(1, args.max_epochs + 1):
        losses = train_distillation_epoch(
            model, train_loader, optimizer, weights, run_args, device
        )
        controls = final_frame.evaluate_controls(
            model, dev_records, run_args, device
        )
        metrics = controls["matched"]["metrics"]
        text_weighted = controls["matched"]["text_metrics"]["weighted_f1"]
        eligible = final_frame.checkpoint_is_eligible(
            metrics["weighted_f1"],
            text_weighted,
            controls["state_margin"],
            controls["zero_audio_margin"],
            controls["audio_margin"],
            args.minimum_state_margin,
            args.minimum_audio_margin,
        )
        row = {
            "epoch": epoch,
            "losses": losses,
            "metrics": metrics,
            "eligible": eligible,
            "state_margin": controls["state_margin"],
            "zero_audio_margin": controls["zero_audio_margin"],
            "audio_margin": controls["audio_margin"],
        }
        history.append(row)
        score = (int(eligible), metrics["macro_f1"], metrics["weighted_f1"])
        print(
            f"Student {seed}, epoch {epoch:02d}: "
            f"macro={metrics['macro_f1']:.4f} weighted={metrics['weighted_f1']:.4f} "
            f"state={controls['state_margin']:+.4f} "
            f"zero={controls['zero_audio_margin']:+.4f} "
            f"shuffle={controls['audio_margin']:+.4f} eligible={eligible}",
            flush=True,
        )
        if best_score is None or score > best_score:
            best_score, best, stale = score, row, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "best_epoch": best["epoch"],
        "eligible": best["eligible"],
        "weighted_f1": best["metrics"]["weighted_f1"],
        "macro_f1": best["metrics"]["macro_f1"],
        "state_margin": best["state_margin"],
        "zero_audio_margin": best["zero_audio_margin"],
        "audio_margin": best["audio_margin"],
        "warmup": warmup,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        default=ROOT / "models/text-emotion",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=ROOT / "research/experiments/audio-cache",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=ROOT / "research/benchmarking/results/final-frame-attention-light",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research/benchmarking/results/frame-attention-distillation",
    )
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[45, 46])
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=20260920)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dialogue-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-audio-frames", type=int)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--control-distillation-weight", type=float, default=0.25)
    parser.add_argument("--minimum-state-margin", type=float, default=0.002)
    parser.add_argument("--minimum-audio-margin", type=float, default=0.005)
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.folds != 3 or args.heldout_fold not in range(args.folds):
        raise ValueError("heldout fold must identify one of three folds")
    if len(args.student_seeds) != 2 or len(set(args.student_seeds)) != 2:
        raise ValueError("distillation requires two distinct student seeds")
    if args.temperature <= 0 or args.learning_rate <= 0:
        raise ValueError("temperature and learning rate must be positive")
    for path in (
        args.raw_archive,
        args.text_model,
        args.audio_cache_dir,
        args.teacher_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    started = time.perf_counter()
    device = select_device(torch)
    preparation_args = final_frame.experiment_args(
        args, args.output_dir, args.student_seeds[0]
    )
    records, normalization = stabilized.prepare_records(preparation_args, device)
    frame.attach_emotion_frames(records, args.max_audio_frames)
    records = final_base.renumber_dialogues(records)
    assignments = assign_dialogue_folds(records["dev"], args.folds, args.fold_seed)
    split = final_base.build_cv_split(records, assignments, args.heldout_fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0],
        std=normalization[1],
    )
    checkpoints = discover_teacher_checkpoints(args.teacher_root)
    teachers = load_teachers(
        checkpoints, split["train"], preparation_args, device
    )
    teacher_report = evaluate_teacher_ensemble(
        teachers,
        split["dev"],
        preparation_args,
        device,
        args.minimum_state_margin,
        args.minimum_audio_margin,
    )
    (args.output_dir / "teacher_ensemble_metrics.json").write_text(
        json.dumps(teacher_report, indent=2), encoding="utf-8"
    )
    print(
        f"Teacher ensemble: weighted={teacher_report['matched']['weighted_f1']:.4f} "
        f"macro={teacher_report['matched']['macro_f1']:.4f} "
        f"state={teacher_report['state_margin']:+.4f} "
        f"zero={teacher_report['zero_audio_margin']:+.4f} "
        f"shuffle={teacher_report['audio_margin']:+.4f}",
        flush=True,
    )
    if not teacher_report["eligible"]:
        raise RuntimeError("teacher ensemble failed the multimodal evidence gate")
    attach_teacher_logits(
        split["train"],
        teachers,
        preparation_args,
        device,
        args.output_dir / "teacher_train_logits.npz",
        rebuild_teacher_cache=args.rebuild_teacher_cache,
    )
    del teachers
    if device.type == "mps":
        torch.mps.empty_cache()
    runs = [
        train_student(seed, split["train"], split["dev"], args, device)
        for seed in args.student_seeds
    ]
    selected = select_student(runs)
    selected_dir = args.output_dir / "best-model"
    selected_dir.mkdir(parents=True, exist_ok=True)
    destination = selected_dir / "best_distilled_frame_attention.pt"
    shutil.copy2(selected["checkpoint"], destination)
    report = {
        "teacher": teacher_report,
        "students": runs,
        "selected": selected,
        "checkpoint": str(destination.resolve()),
        "heldout_fold": args.heldout_fold + 1,
    }
    if args.evaluate_test:
        run_args = final_frame.experiment_args(
            args, selected_dir, selected["seed"]
        )
        student = create_model(split["train"], run_args, device)
        student.load_state_dict(
            torch.load(destination, map_location=device, weights_only=True)
        )
        report["test"] = final_frame.evaluate_controls(
            student, records["test"], run_args, device
        )
        report["test"] = {
            "matched": report["test"]["matched"]["metrics"],
            "text": report["test"]["matched"]["text_metrics"],
            "reset": report["test"]["reset"]["metrics"],
            "zero_audio": report["test"]["zero"]["metrics"],
            "shuffled_audio": [
                item["metrics"] for item in report["test"]["shuffled"]
            ],
            "state_margin": report["test"]["state_margin"],
            "zero_audio_margin": report["test"]["zero_audio_margin"],
            "audio_margin": report["test"]["audio_margin"],
        }
    (args.output_dir / "distillation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"Selected student seed {selected['seed']}: "
        f"weighted={selected['weighted_f1']:.4f} "
        f"macro={selected['macro_f1']:.4f}"
    )
    print(f"Best model: {destination}")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
