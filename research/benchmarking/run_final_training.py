#!/usr/bin/env python3
"""Select fusion learning rate by repeated 5-fold CV, then train one final MELD model.

The frozen context-two BERT and audio caches must already exist. Five
dialogue-level folds over the development split compare three fusion learning
rates across three repeats while retaining the full official training split.
The winning rate is retrained for the median selected epoch on train plus dev,
then evaluated once on the official test split.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
INITIAL_TESTING = ROOT / "research/experiments"
sys.path.insert(0, str(INITIAL_TESTING))

import train_recurrent_dialogue as recurrent  # noqa: E402
import train_recurrent_dialogue_stabilized as stabilized  # noqa: E402
from cache_embeddings import load_split_rows  # noqa: E402
from evaluate_meld import EMOTION_LABELS, select_device  # noqa: E402
from run_final_architecture_cv import assign_dialogue_folds  # noqa: E402
from train_text import sqrt_class_weights  # noqa: E402
from train_text_audio_phase2 import (  # noqa: E402
    attach_speaker_relative_acoustics,
    combine_records,
)


PROTOCOL_VERSION = "robust-dev-cv-v2"


def renumber_dialogues(records_by_split):
    result = {}
    next_id = 0
    for split, records in records_by_split.items():
        mapping = {}
        result[split] = []
        for record in records:
            source_id = str(record["dialogue_id"])
            if source_id not in mapping:
                mapping[source_id] = str(next_id)
                next_id += 1
            copied = dict(record)
            copied["dialogue_id"] = mapping[source_id]
            copied["source_split"] = split
            copied["source_dialogue_id"] = source_id
            result[split].append(copied)
    return result


def copy_records(records):
    copied = [dict(record) for record in records]
    for index, record in enumerate(copied):
        record["record_index"] = index
    return copied


def select_dialogues(records, dialogue_ids):
    selected = set(dialogue_ids)
    return copy_records(
        [record for record in records if str(record["dialogue_id"]) in selected]
    )


def choose_configuration(
    candidates, macro_tolerance=0.005, minimum_state_margin=0.002
):
    if not candidates:
        raise ValueError("no CV configurations were completed")
    state_eligible = [
        candidate
        for candidate in candidates
        if candidate["state_margin"] >= minimum_state_margin
    ]
    if not state_eligible:
        raise ValueError("no CV configuration retained positive recurrent-state evidence")
    best_macro = max(candidate["macro_f1"] for candidate in state_eligible)
    eligible = [
        candidate
        for candidate in state_eligible
        if candidate["macro_f1"] >= best_macro - macro_tolerance
    ]
    return max(
        eligible,
        key=lambda candidate: (
            candidate["weighted_f1"] - candidate["weighted_f1_std"],
            candidate["weighted_f1"],
            candidate["macro_f1"],
            -candidate["learning_rate"],
        ),
    )


def selected_epoch(epochs):
    if not epochs:
        raise ValueError("no selected epochs were recorded")
    return max(1, int(np.median(epochs) + 0.5))


def build_cv_split(records, assignments, fold):
    validation_dialogues = {
        dialogue for dialogue, assigned in assignments.items() if assigned == fold
    }
    development_train_dialogues = set(assignments) - validation_dialogues
    validation = select_dialogues(records["dev"], validation_dialogues)
    training = copy_records(
        records["train"]
        + select_dialogues(records["dev"], development_train_dialogues)
    )
    return {
        "train": training,
        "dev": copy_records(validation),
        "test": copy_records(validation),
    }


def load_cached_records(args):
    raw = {}
    missing = []
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        raw[split] = combine_records(
            rows, args.context_window, args.audio_cache_dir, split
        )
        for record in raw[split]:
            cache = (
                args.audio_cache_dir
                / split
                / f"recurrent-text-c{args.context_window}"
                / f"dia{record['dialogue_id']}_utt{record['utterance_id']}.npz"
            )
            record["recurrent_text_path"] = str(cache.resolve())
            if not cache.is_file():
                missing.append(cache)
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"missing {len(missing)} frozen text caches; examples: {examples}"
        )
    attach_speaker_relative_acoustics(raw)
    recurrent.load_cached_features(raw)
    return renumber_dialogues(raw)


def fusion_arguments(
    args, learning_rate, output_dir, epochs=None, patience=None, seed=None
):
    return SimpleNamespace(
        seed=args.seed if seed is None else seed,
        runs=1,
        epochs=args.max_epochs if epochs is None else epochs,
        dialogue_batch_size=args.dialogue_batch_size,
        gradient_accumulation=1,
        learning_rate=learning_rate,
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
        audio_max_gate=0.15,
        initial_gate_bias=-2.0,
        audio_loss_weight=0.3,
        counterfactual_weight=0.5,
        counterfactual_margin=0.1,
        state_counterfactual_weight=0.3,
        state_counterfactual_margin=0.02,
        minimum_dev_state_margin=0.002,
        negative_residual_weight=0.2,
        correction_penalty_weight=0.01,
        context_gate_soft_ceiling=0.18,
        audio_gate_soft_ceiling=0.10,
        gate_penalty_weight=0.2,
        context_window=args.context_window,
        max_length=256,
        patience=args.patience if patience is None else patience,
        shuffle_seeds=[43, 44, 45],
        output_dir=output_dir,
    )


def aggregate_reports(learning_rate, reports):
    def values(key):
        return np.asarray([report[key] for report in reports], dtype=np.float64)

    weighted = values("weighted_f1")
    macro = values("macro_f1")
    labels = reports[0]["per_class_f1"]
    return {
        "learning_rate": learning_rate,
        "weighted_f1": float(weighted.mean()),
        "weighted_f1_std": float(weighted.std(ddof=0)),
        "weighted_f1_min": float(weighted.min()),
        "macro_f1": float(macro.mean()),
        "macro_f1_std": float(macro.std(ddof=0)),
        "macro_f1_min": float(macro.min()),
        "state_margin": float(values("state_margin").mean()),
        "zero_audio_margin": float(values("zero_audio_margin").mean()),
        "audio_margin": float(values("audio_margin").mean()),
        "per_class_f1": {
            label: float(
                np.mean([report["per_class_f1"][label] for report in reports])
            )
            for label in labels
        },
        "selected_epoch": selected_epoch([report["best_epoch"] for report in reports]),
        "runs": reports,
    }


def validate_cached_report(report, expected):
    actual = report.get("cache_metadata")
    if not isinstance(actual, dict):
        raise ValueError("cached run lacks cache_metadata; rerun with --retrain")
    mismatches = [
        key for key, value in expected.items() if key not in actual or actual[key] != value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"cached run metadata mismatch for {joined}; rerun with --retrain"
        )


def is_completed_final_report(report, expected):
    try:
        validate_cached_report(report, expected)
    except ValueError:
        return False
    return isinstance(report.get("test_recurrent_matched"), dict)


def cv_cache_metadata(args, learning_rate, repeat, fold, split_records):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "learning_rate": learning_rate,
        "repeat": repeat + 1,
        "fold": fold + 1,
        "fold_seed": args.fold_seed + repeat,
        "train_seed": args.seed + repeat,
        "folds": args.folds,
        "repeats": args.repeats,
        "epochs": args.max_epochs,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "context_window": args.context_window,
        "dialogue_batch_size": args.dialogue_batch_size,
        "training_samples": len(split_records["train"]),
        "validation_samples": len(split_records["dev"]),
        "training_dialogues": len(recurrent.group_dialogues(split_records["train"])),
        "validation_dialogues": len(recurrent.group_dialogues(split_records["dev"])),
    }


def run_cross_validation(records, args, device):
    reports_by_rate = {rate: [] for rate in args.learning_rates}
    protocol_dir = args.output_dir / PROTOCOL_VERSION
    for repeat in range(args.repeats):
        assignments = assign_dialogue_folds(
            records["dev"], args.folds, args.fold_seed + repeat
        )
        for learning_rate in args.learning_rates:
            for fold in range(args.folds):
                split_records = build_cv_split(records, assignments, fold)
                rate_name = f"lr-{learning_rate:.0e}".replace("+", "")
                run_dir = (
                    protocol_dir / "cv" / f"repeat-{repeat + 1}" / rate_name
                    / f"fold-{fold + 1}"
                )
                metrics_path = run_dir / "metrics.json"
                metadata = cv_cache_metadata(
                    args, learning_rate, repeat, fold, split_records
                )
                if metrics_path.exists() and not args.retrain:
                    report = json.loads(metrics_path.read_text(encoding="utf-8"))
                    validate_cached_report(report, metadata)
                    print(
                        f"Reusing repeat {repeat + 1}, {rate_name}, fold {fold + 1}",
                        flush=True,
                    )
                else:
                    print(
                        f"Training repeat {repeat + 1}/{args.repeats}, {rate_name}, "
                        f"fold {fold + 1}/{args.folds}",
                        flush=True,
                    )
                    run_args = fusion_arguments(
                        args, learning_rate, run_dir, seed=args.seed + repeat
                    )
                    report = stabilized.train(
                        run_args, split_records, device, fold + 1
                    )
                    report["cache_metadata"] = metadata
                    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                reports_by_rate[learning_rate].append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold + 1,
                        "fold_seed": args.fold_seed + repeat,
                        "train_seed": args.seed + repeat,
                        "best_epoch": report["best_epoch"],
                        "weighted_f1": report["test_recurrent_matched"][
                            "weighted_f1"
                        ],
                        "macro_f1": report["test_recurrent_matched"]["macro_f1"],
                        "per_class_f1": report["test_recurrent_matched"][
                            "per_class_f1"
                        ],
                        "state_margin": report["test_state_margin"],
                        "zero_audio_margin": report["test_zero_audio_margin"],
                        "audio_margin": report["test_audio_margin"],
                    }
                )
    summaries = [
        aggregate_reports(rate, reports_by_rate[rate]) for rate in args.learning_rates
    ]
    selected = choose_configuration(
        summaries, args.macro_tolerance, args.minimum_state_margin
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": (
            "full official train plus four development folds; validate on the "
            "fifth development fold, repeated three times"
        ),
        "caveat": (
            "The frozen BERT checkpoint was selected using the full development "
            "split, so this is not fully nested or an unbiased text-encoder estimate."
        ),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "fold_seed": args.fold_seed,
        "macro_tolerance": args.macro_tolerance,
        "minimum_state_margin": args.minimum_state_margin,
        "configurations": summaries,
        "selected": selected,
    }
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "cv_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return selected


def train_final_model(records, args, selected, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    final_dir = args.output_dir / PROTOCOL_VERSION / "best-model"
    final_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = final_dir / "best_recurrent_dialogue_stabilized.pt"
    metrics_path = final_dir / "metrics.json"
    final_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "learning_rate": selected["learning_rate"],
        "train_seed": args.seed,
        "epochs": selected["selected_epoch"],
        "context_window": args.context_window,
    }
    if metrics_path.exists() and checkpoint.exists() and not args.retrain_final:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        if is_completed_final_report(report, final_metadata):
            print(f"Reusing completed final model: {checkpoint}", flush=True)
            return report
    train_records = copy_records(records["train"] + records["dev"])
    test_records = copy_records(records["test"])
    run_args = fusion_arguments(
        args,
        selected["learning_rate"],
        final_dir,
        epochs=selected["selected_epoch"],
        patience=selected["selected_epoch"] + 1,
    )
    sample = train_records[0]
    model = stabilized.StabilizedRecurrentDialogueModel(
        len(sample["text_embedding"]),
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        run_args,
    ).to(device)
    train_loader = recurrent.make_loader(train_records, run_args, True)
    test_loader = recurrent.make_loader(test_records, run_args, False)
    labels = np.asarray([EMOTION_LABELS.index(row["label"]) for row in train_records])
    class_weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run_args.learning_rate, weight_decay=run_args.weight_decay
    )
    history = []
    started = time.perf_counter()
    for epoch in range(1, run_args.epochs + 1):
        losses = stabilized.train_epoch(
            model, train_loader, optimizer, class_weights, run_args, device
        )
        history.append({"epoch": epoch, "losses": losses})
        print(
            f"Final epoch {epoch:02d}/{run_args.epochs}: loss={losses['total']:.4f}",
            flush=True,
        )
    torch.save(model.state_dict(), checkpoint)
    matched = recurrent.evaluate(model, test_loader, device)
    reset = recurrent.evaluate(model, test_loader, device, reset_each_turn=True)
    zero_audio = recurrent.evaluate(model, test_loader, device, zero_audio=True)
    shuffled = []
    for seed in run_args.shuffle_seeds:
        loader = recurrent.make_loader(
            test_records,
            run_args,
            False,
            recurrent.different_label_audio_mapping(test_records, seed),
        )
        shuffled.append(recurrent.evaluate(model, loader, device))
    state_margin = (
        matched["metrics"]["weighted_f1"] - reset["metrics"]["weighted_f1"]
    )
    zero_audio_margin = (
        matched["metrics"]["weighted_f1"] - zero_audio["metrics"]["weighted_f1"]
    )
    audio_margin = matched["metrics"]["weighted_f1"] - max(
        result["metrics"]["weighted_f1"] for result in shuffled
    )
    recurrent.write_predictions(final_dir / "test_predictions.csv", test_records, matched)
    report = {
        "protocol": "development-only CV followed by fixed-epoch train-plus-dev retraining",
        "cache_metadata": final_metadata,
        "selection": selected,
        "training_samples": len(train_records),
        "training_dialogues": len(recurrent.group_dialogues(train_records)),
        "test_samples": len(test_records),
        "epochs": run_args.epochs,
        "learning_rate": run_args.learning_rate,
        "test_text": matched["text_metrics"],
        "test_recurrent_matched": matched["metrics"],
        "test_reset_state": reset["metrics"],
        "test_zero_audio": zero_audio["metrics"],
        "test_shuffled_audio": [result["metrics"] for result in shuffled],
        "test_state_margin": state_margin,
        "test_zero_audio_margin": zero_audio_margin,
        "test_audio_margin": audio_margin,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    (final_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(args.normalization_path, final_dir / "emotion_normalization.npz")
    print(
        f"Final test weighted F1: {matched['metrics']['weighted_f1']:.4f}; "
        f"macro F1: {matched['metrics']['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Best model: {checkpoint}", flush=True)
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=ROOT / "research/experiments/audio-cache",
    )
    parser.add_argument(
        "--normalization-path",
        type=Path,
        default=ROOT
        / "research/experiments/training-output-recurrent-dialogue-stabilized-c2-five"
        / "emotion_normalization.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research/benchmarking/results/final-training",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[1e-4, 2e-4, 3e-4]
    )
    parser.add_argument("--fold-seed", type=int, default=20260919)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dialogue-batch-size", type=int, default=16)
    parser.add_argument("--macro-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-state-margin", type=float, default=0.002)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retrain-final", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.folds != 5 or args.repeats != 3 or len(args.learning_rates) != 3:
        raise ValueError(
            "the robust final protocol requires three repeats, five folds, and three rates"
        )
    if len(set(args.learning_rates)) != 3 or any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("learning rates must be three distinct positive values")
    if args.context_window != 2:
        raise ValueError("the final architecture requires context window two")
    if args.max_epochs < 1 or args.patience < 1 or args.dialogue_batch_size < 1:
        raise ValueError("training counts must be positive")
    if not 0 <= args.macro_tolerance <= 0.05:
        raise ValueError("macro tolerance must be between zero and 0.05")
    if not 0 <= args.minimum_state_margin <= 0.05:
        raise ValueError("minimum state margin must be between zero and 0.05")
    for path in (args.raw_archive, args.audio_cache_dir, args.normalization_path):
        if not path.exists():
            raise FileNotFoundError(path)


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    started = time.perf_counter()
    records = load_cached_records(args)
    print(
        f"Loaded cached records: train={len(records['train'])}, "
        f"dev={len(records['dev'])}, test={len(records['test'])}",
        flush=True,
    )
    device = select_device(torch)
    selected = run_cross_validation(records, args, device)
    print(
        f"Selected learning rate {selected['learning_rate']:.1e} and "
        f"{selected['selected_epoch']} final epochs",
        flush=True,
    )
    train_final_model(records, args, selected, device)
    print(f"Total runtime: {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
