#!/usr/bin/env python3
"""Light final validation and training for staged full-frame audio fusion."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
INITIAL_TESTING = ROOT / "initial-testing"
sys.path.insert(0, str(INITIAL_TESTING))

import train_recurrent_dialogue as recurrent  # noqa: E402
import train_recurrent_dialogue_frame_attention as frame  # noqa: E402
import train_recurrent_dialogue_stabilized as stabilized  # noqa: E402
from evaluate_meld import EMOTION_LABELS, select_device  # noqa: E402
from run_final_architecture_cv import assign_dialogue_folds  # noqa: E402
from train_text import sqrt_class_weights  # noqa: E402
from train_text_audio_phase2 import speaker_acoustic_normalization  # noqa: E402

import run_final_training as final_base  # noqa: E402


PROTOCOL_VERSION = "light-frame-attention-disagreement-gate-v1"


def selected_epoch(epochs):
    if not epochs:
        raise ValueError("no epochs were supplied")
    return max(1, int(np.median(epochs) + 0.5))


def cv_schedule(args):
    return [
        {"seed": seed, "fold": fold}
        for seed in args.seeds
        for fold in range(args.folds)
    ]


def checkpoint_is_eligible(
    weighted_f1,
    text_weighted_f1,
    state_margin,
    zero_audio_margin,
    shuffle_audio_margin,
    minimum_state_margin=0.002,
    minimum_audio_margin=0.005,
):
    return bool(
        weighted_f1 >= text_weighted_f1
        and state_margin >= minimum_state_margin
        and zero_audio_margin >= minimum_audio_margin
        and shuffle_audio_margin >= minimum_audio_margin
    )


def select_epochs(runs, minimum_eligible=4):
    eligible = [run for run in runs if run["eligible"]]
    if len(eligible) < minimum_eligible:
        raise RuntimeError(
            f"only {len(eligible)}/{len(runs)} CV runs proved audio use; "
            f"need at least {minimum_eligible}"
        )
    return {
        "eligible_runs": len(eligible),
        "total_runs": len(runs),
        "warmup_epochs": selected_epoch(
            [run["warmup_epoch"] for run in eligible]
        ),
        "joint_epochs": selected_epoch([run["best_epoch"] for run in eligible]),
    }


def experiment_args(args, output_dir, seed):
    return SimpleNamespace(
        raw_archive=args.raw_archive,
        text_model=args.text_model,
        audio_cache_dir=args.audio_cache_dir,
        output_dir=output_dir,
        epochs=args.max_epochs,
        patience=args.patience,
        dialogue_batch_size=args.dialogue_batch_size,
        encoder_batch_size=16,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        audio_warmup_epochs=args.audio_warmup_epochs,
        audio_warmup_learning_rate=args.audio_warmup_learning_rate,
        weight_decay=0.01,
        text_projection_dimension=256,
        audio_projection_dimension=128,
        dialogue_state_dimension=128,
        speaker_state_dimension=64,
        dropout=0.2,
        dialogue_state_dropout=0.05,
        speaker_state_dropout=0.10,
        audio_dropout=0.10,
        dialogue_reset_probability=0.01,
        speaker_reset_probability=0.03,
        context_max_gate=0.25,
        audio_max_gate=1.0,
        initial_gate_bias=-2.0,
        direct_audio_mix=True,
        disagreement_gate=True,
        audio_loss_weight=0.3,
        counterfactual_weight=0.5,
        counterfactual_margin=0.1,
        state_counterfactual_weight=0.3,
        state_counterfactual_margin=0.02,
        minimum_dev_state_margin=args.minimum_state_margin,
        negative_residual_weight=0.2,
        correction_penalty_weight=0.01,
        context_gate_soft_ceiling=0.18,
        audio_gate_soft_ceiling=0.70,
        gate_penalty_weight=0.02,
        disagreement_gate_weight=1.0,
        disagreement_gate_warmup_epochs=3,
        disagreement_gate_learning_rate=2e-4,
        context_window=2,
        max_length=256,
        max_audio_frames=args.max_audio_frames,
        shuffle_seeds=[43, 44, 45],
        seed=seed,
        max_samples_per_split=None,
        rebuild_text_cache=False,
    )


def warmup_best_epoch(history):
    best = max(
        history,
        key=lambda row: (row["dev"]["macro_f1"], row["dev"]["weighted_f1"]),
    )
    return best["epoch"]


def evaluate_controls(model, records, args, device):
    loader = frame.make_loader(records, args, False)
    matched = recurrent.evaluate(model, loader, device)
    reset = recurrent.evaluate(model, loader, device, reset_each_turn=True)
    zero = recurrent.evaluate(model, loader, device, zero_audio=True)
    shuffled = []
    for shuffle_seed in args.shuffle_seeds:
        shuffled_loader = frame.make_loader(
            records,
            args,
            False,
            recurrent.different_label_audio_mapping(records, shuffle_seed),
        )
        shuffled.append(recurrent.evaluate(model, shuffled_loader, device))
    weighted = matched["metrics"]["weighted_f1"]
    return {
        "matched": matched,
        "reset": reset,
        "zero": zero,
        "shuffled": shuffled,
        "state_margin": weighted - reset["metrics"]["weighted_f1"],
        "zero_audio_margin": weighted - zero["metrics"]["weighted_f1"],
        "audio_margin": weighted
        - max(item["metrics"]["weighted_f1"] for item in shuffled),
    }


def train_fold(split_records, args, device, seed, fold, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    run_args = experiment_args(args, run_dir, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sample = split_records["train"][0]
    model = frame.FrameAttentionRecurrentModel(
        len(sample["text_embedding"]),
        sample["emotion_frames"].shape[1],
        len(sample["speaker_acoustic"]),
        len(EMOTION_LABELS),
        run_args,
    ).to(device)
    train_loader = frame.make_loader(split_records["train"], run_args, True)
    dev_loader = frame.make_loader(split_records["dev"], run_args, False)
    labels = np.asarray(
        [EMOTION_LABELS.index(row["label"]) for row in split_records["train"]]
    )
    weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    baseline = recurrent.evaluate(model, dev_loader, device)["text_metrics"]
    warmup = frame.warm_up_audio(
        model, train_loader, dev_loader, weights, run_args, device
    )
    gate_warmup = frame.warm_up_disagreement_gate(
        model, train_loader, run_args, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    checkpoint = run_dir / "best_frame_attention.pt"
    history, best_row, best_score, stale = [], None, None, 0
    for epoch in range(1, args.max_epochs + 1):
        losses = frame.train_epoch(
            model, train_loader, optimizer, weights, run_args, device
        )
        controls = evaluate_controls(
            model, split_records["dev"], run_args, device
        )
        metrics = controls["matched"]["metrics"]
        eligible = checkpoint_is_eligible(
            metrics["weighted_f1"],
            baseline["weighted_f1"],
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
            "state_margin": controls["state_margin"],
            "zero_audio_margin": controls["zero_audio_margin"],
            "audio_margin": controls["audio_margin"],
            "eligible": eligible,
        }
        history.append(row)
        score = (
            int(eligible),
            metrics["macro_f1"],
            metrics["weighted_f1"],
            controls["audio_margin"],
        )
        print(
            f"Seed {seed}, fold {fold + 1}, epoch {epoch:02d}: "
            f"macro={metrics['macro_f1']:.4f} weighted={metrics['weighted_f1']:.4f} "
            f"state={controls['state_margin']:+.4f} "
            f"zero={controls['zero_audio_margin']:+.4f} "
            f"shuffle={controls['audio_margin']:+.4f} eligible={eligible}",
            flush=True,
        )
        if best_score is None or score > best_score:
            best_score, best_row, stale = score, row, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "fold": fold + 1,
        "learning_rate": args.learning_rate,
        "warmup_epoch": warmup_best_epoch(warmup),
        "gate_warmup": gate_warmup,
        "best_epoch": best_row["epoch"],
        "eligible": best_row["eligible"],
        "weighted_f1": best_row["metrics"]["weighted_f1"],
        "macro_f1": best_row["metrics"]["macro_f1"],
        "per_class_f1": best_row["metrics"]["per_class_f1"],
        "state_margin": best_row["state_margin"],
        "zero_audio_margin": best_row["zero_audio_margin"],
        "audio_margin": best_row["audio_margin"],
        "history": history,
        "warmup": warmup,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def aggregate_runs(runs):
    def stats(field):
        values = [run[field] for run in runs]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
        }

    return {
        "weighted_f1": stats("weighted_f1"),
        "macro_f1": stats("macro_f1"),
        "state_margin": stats("state_margin"),
        "zero_audio_margin": stats("zero_audio_margin"),
        "audio_margin": stats("audio_margin"),
        "eligible_runs": sum(run["eligible"] for run in runs),
        "per_class_f1": {
            label: float(np.mean([run["per_class_f1"][label] for run in runs]))
            for label in EMOTION_LABELS
        },
    }


def train_audio_fixed(model, loader, weights, run_args, device, epochs):
    optimizer = torch.optim.AdamW(
        model.audio_warmup_parameters(),
        lr=run_args.audio_warmup_learning_rate,
        weight_decay=run_args.weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = recurrent.move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.audio_only_logits(batch)
            loss = recurrent.masked_cross_entropy(
                logits, batch["labels"], batch["valid_mask"], weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.audio_warmup_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": value})
        print(f"Final audio warm-up {epoch:02d}/{epochs}: loss={value:.4f}")
    return history


def train_final(records, args, selected, device):
    final_dir = args.output_dir / "best-model"
    metrics_path = final_dir / "metrics.json"
    checkpoint = final_dir / "best_frame_attention.pt"
    if metrics_path.exists() and checkpoint.exists() and not args.retrain_final:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        if report.get("protocol_version") == PROTOCOL_VERSION:
            print(f"Reusing final model: {checkpoint}")
            return report
    final_dir.mkdir(parents=True, exist_ok=True)
    run_args = experiment_args(args, final_dir, args.seeds[0])
    random.seed(run_args.seed)
    np.random.seed(run_args.seed)
    torch.manual_seed(run_args.seed)
    train_records = final_base.copy_records(records["train"] + records["dev"])
    test_records = final_base.copy_records(records["test"])
    sample = train_records[0]
    model = frame.FrameAttentionRecurrentModel(
        len(sample["text_embedding"]),
        sample["emotion_frames"].shape[1],
        len(sample["speaker_acoustic"]),
        len(EMOTION_LABELS),
        run_args,
    ).to(device)
    train_loader = frame.make_loader(train_records, run_args, True)
    labels = np.asarray(
        [EMOTION_LABELS.index(row["label"]) for row in train_records]
    )
    weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    warmup_history = train_audio_fixed(
        model,
        train_loader,
        weights,
        run_args,
        device,
        selected["warmup_epochs"],
    )
    gate_warmup_history = frame.warm_up_disagreement_gate(
        model, train_loader, run_args, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    joint_history = []
    for epoch in range(1, selected["joint_epochs"] + 1):
        losses = frame.train_epoch(
            model, train_loader, optimizer, weights, run_args, device
        )
        joint_history.append({"epoch": epoch, "losses": losses})
        print(
            f"Final joint epoch {epoch:02d}/{selected['joint_epochs']}: "
            f"loss={losses['total']:.4f}",
            flush=True,
        )
    torch.save(model.state_dict(), checkpoint)
    controls = evaluate_controls(model, test_records, run_args, device)
    recurrent.write_predictions(
        final_dir / "test_predictions.csv", test_records, controls["matched"]
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "selection": selected,
        "learning_rate": args.learning_rate,
        "warmup_epochs": selected["warmup_epochs"],
        "joint_epochs": selected["joint_epochs"],
        "test_text": controls["matched"]["text_metrics"],
        "test_recurrent_matched": controls["matched"]["metrics"],
        "test_reset_state": controls["reset"]["metrics"],
        "test_zero_audio": controls["zero"]["metrics"],
        "test_shuffled_audio": [
            item["metrics"] for item in controls["shuffled"]
        ],
        "test_state_margin": controls["state_margin"],
        "test_zero_audio_margin": controls["zero_audio_margin"],
        "test_audio_margin": controls["audio_margin"],
        "checkpoint": str(checkpoint.resolve()),
        "audio_warmup_history": warmup_history,
        "gate_warmup_history": gate_warmup_history,
        "joint_history": joint_history,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Final test weighted F1: "
        f"{report['test_recurrent_matched']['weighted_f1']:.4f}; "
        f"macro F1: {report['test_recurrent_matched']['macro_f1']:.4f}"
    )
    print(f"Best model: {checkpoint}")
    return report


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
        default=ROOT / "benchmarking/results/final-frame-attention-light",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--fold-seed", type=int, default=20260920)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-learning-rate", type=float, default=2e-4)
    parser.add_argument("--audio-warmup-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dialogue-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-audio-frames", type=int)
    parser.add_argument("--minimum-state-margin", type=float, default=0.002)
    parser.add_argument("--minimum-audio-margin", type=float, default=0.005)
    parser.add_argument("--minimum-eligible-runs", type=int, default=4)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retrain-final", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.folds != 3 or len(args.seeds) != 2:
        raise ValueError("the light protocol requires three folds and two seeds")
    if len(set(args.seeds)) != 2:
        raise ValueError("training seeds must be distinct")
    for name in (
        "audio_warmup_epochs",
        "max_epochs",
        "patience",
        "dialogue_batch_size",
        "gradient_accumulation",
        "minimum_eligible_runs",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if args.learning_rate <= 0 or args.audio_warmup_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    for path in (args.raw_archive, args.text_model, args.audio_cache_dir):
        if not path.exists():
            raise FileNotFoundError(path)


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    started = time.perf_counter()
    device = select_device(torch)
    preparation_args = experiment_args(args, args.output_dir, args.seeds[0])
    records, normalization = stabilized.prepare_records(preparation_args, device)
    frame.attach_emotion_frames(records, args.max_audio_frames)
    records = final_base.renumber_dialogues(records)
    _, absolute_stats, acoustic_stats = speaker_acoustic_normalization(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0],
        std=normalization[1],
    )
    np.savez_compressed(
        args.output_dir / "speaker_acoustic_normalization.npz",
        absolute_mean=absolute_stats[0],
        absolute_std=absolute_stats[1],
        final_mean=acoustic_stats[0],
        final_std=acoustic_stats[1],
    )
    assignments = assign_dialogue_folds(records["dev"], args.folds, args.fold_seed)
    runs = []
    for item in cv_schedule(args):
        seed, fold = item["seed"], item["fold"]
        run_dir = args.output_dir / "cv" / f"seed-{seed}" / f"fold-{fold + 1}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.retrain:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            if report.get("protocol_version") == PROTOCOL_VERSION:
                print(f"Reusing seed {seed}, fold {fold + 1}")
                runs.append(report)
                continue
        split = final_base.build_cv_split(records, assignments, fold)
        print(f"Training seed {seed}, fold {fold + 1}/{args.folds}")
        runs.append(train_fold(split, args, device, seed, fold, run_dir))
    selected = select_epochs(runs, args.minimum_eligible_runs)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "three development folds by two training seeds at fixed LR",
        "caveat": "Frozen text encoder selection used the full development split.",
        "configuration": {
            "folds": args.folds,
            "seeds": args.seeds,
            "fold_seed": args.fold_seed,
            "learning_rate": args.learning_rate,
        },
        "aggregate": aggregate_runs(runs),
        "selected": selected,
        "runs": runs,
    }
    (args.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {selected['warmup_epochs']} audio warm-up epochs and "
        f"{selected['joint_epochs']} joint epochs from "
        f"{selected['eligible_runs']}/{selected['total_runs']} eligible runs"
    )
    train_final(records, args, selected, device)
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
