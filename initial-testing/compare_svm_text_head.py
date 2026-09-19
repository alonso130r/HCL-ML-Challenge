#!/usr/bin/env python3
"""Compare neural and calibrated-linear-SVM anchors for the final recurrent model.

This experiment reuses the existing context-two BERT and audio caches. It fits
the SVM and its probability calibration on the training split only, swaps the
resulting log-probabilities into ``text_logits``, then retrains the unchanged
stabilized recurrent fusion model with seed 43.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import train_recurrent_dialogue as recurrent
import train_recurrent_dialogue_stabilized as stabilized
from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text_audio_phase2 import attach_speaker_relative_acoustics, combine_records


ROOT = Path(__file__).resolve().parents[1]


class CalibratedSvmHead:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classes_ = classifier.classes_

    def predict_log_proba(self, values):
        probabilities = self.classifier.predict_proba(values)
        return np.log(np.clip(probabilities, 1e-12, 1.0))


def fit_svm_head(records, class_count, seed=43, folds=3, c=1.0):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    embeddings = np.stack([record["text_embedding"] for record in records])
    labels = np.asarray([record["label_index"] for record in records])
    if set(labels.tolist()) != set(range(class_count)):
        raise ValueError("SVM training records must contain every emotion class")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        ),
    )
    classifier = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=splitter,
        ensemble=False,
    )
    classifier.fit(embeddings, labels)
    if set(classifier.classes_.tolist()) != set(range(class_count)):
        raise ValueError("calibrated SVM did not retain every emotion class")
    return CalibratedSvmHead(classifier)


def replace_text_logits(records, classifier, class_count):
    embeddings = np.stack([record["text_embedding"] for record in records])
    scores = classifier.predict_log_proba(embeddings)
    positions = {int(label): index for index, label in enumerate(classifier.classes_)}
    if set(positions) != set(range(class_count)):
        raise ValueError("SVM output classes do not match the MELD label set")
    scores = scores[:, [positions[index] for index in range(class_count)]]
    for record, score in zip(records, scores):
        record["text_logits"] = score.astype(np.float32)


def choose_winner(neural_dev, svm_dev):
    if (
        svm_dev["weighted_f1"] > neural_dev["weighted_f1"]
        and svm_dev["macro_f1"] >= neural_dev["macro_f1"]
    ):
        return "svm"
    return "neural"


def metrics_for_logits(records):
    actual = [record["label"] for record in records]
    predictions = [
        EMOTION_LABELS[int(np.argmax(record["text_logits"]))] for record in records
    ]
    return compute_metrics(actual, predictions)


def load_cached_records(args):
    records = {}
    missing = []
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        records[split] = combine_records(
            rows, args.context_window, args.audio_cache_dir, split
        )
        for record in records[split]:
            cache_path = (
                args.audio_cache_dir
                / split
                / f"recurrent-text-c{args.context_window}"
                / f"dia{record['dialogue_id']}_utt{record['utterance_id']}.npz"
            )
            record["recurrent_text_path"] = str(cache_path.resolve())
            record["label_index"] = EMOTION_LABELS.index(record["label"])
            if not cache_path.is_file():
                missing.append(cache_path)
    if missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"missing {len(missing)} context-two text caches; examples: {examples}"
        )
    attach_speaker_relative_acoustics(records)
    recurrent.load_cached_features(records)
    return records


def selected_dev_metrics(run_dir):
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    selected = next(
        row for row in history if row["epoch"] == report["best_epoch"]
    )
    return selected["dev_matched"]


def fusion_arguments(args):
    return SimpleNamespace(
        seed=args.seed,
        runs=1,
        epochs=args.epochs,
        dialogue_batch_size=16,
        gradient_accumulation=1,
        learning_rate=2e-4,
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
        patience=args.patience,
        shuffle_seeds=[43, 44, 45],
        output_dir=args.output_dir / "svm-recurrent",
    )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=ROOT / "initial-testing/audio-cache",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=ROOT
        / "initial-testing/training-output-recurrent-dialogue-stabilized-c2-five"
        / "run-02-seed-43",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "initial-testing/results-svm-text-head-comparison",
    )
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.context_window != 2:
        raise ValueError("this comparison is fixed to the selected context-two model")
    if args.calibration_folds < 2 or args.svm_c <= 0:
        raise ValueError("calibration folds must be at least two and SVM C must be positive")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    required = [
        args.raw_archive,
        args.baseline_run_dir / "metrics.json",
        args.baseline_run_dir / "training_history.json",
        args.baseline_run_dir / "best_recurrent_dialogue_stabilized.pt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required input not found: {missing[0]}")


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    records = load_cached_records(args)
    neural_text = {
        split: metrics_for_logits(records[split]) for split in ("dev", "test")
    }
    classifier = fit_svm_head(
        records["train"],
        len(EMOTION_LABELS),
        seed=args.seed,
        folds=args.calibration_folds,
        c=args.svm_c,
    )
    for split in records:
        replace_text_logits(records[split], classifier, len(EMOTION_LABELS))
    svm_text = {
        split: metrics_for_logits(records[split]) for split in ("dev", "test")
    }

    run_args = fusion_arguments(args)
    device = select_device(__import__("torch"))
    svm_report = stabilized.train(run_args, records, device, run_number=1)
    svm_dev = selected_dev_metrics(run_args.output_dir)
    neural_dev = selected_dev_metrics(args.baseline_run_dir)
    neural_report = json.loads(
        (args.baseline_run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    winner = choose_winner(neural_dev, svm_dev)
    checkpoints = {
        "neural": args.baseline_run_dir / "best_recurrent_dialogue_stabilized.pt",
        "svm": run_args.output_dir / "best_recurrent_dialogue_stabilized.pt",
    }
    result = {
        "selection_rule": (
            "SVM must improve development weighted F1 without reducing development macro F1."
        ),
        "winner": winner,
        "best_model": str(checkpoints[winner].resolve()),
        "neural": {
            "text_dev": neural_text["dev"],
            "text_test": neural_text["test"],
            "recurrent_dev": neural_dev,
            "recurrent_test": neural_report["test_recurrent_matched"],
            "checkpoint": str(checkpoints["neural"].resolve()),
        },
        "svm": {
            "c": args.svm_c,
            "calibration_folds": args.calibration_folds,
            "text_dev": svm_text["dev"],
            "text_test": svm_text["test"],
            "recurrent_dev": svm_dev,
            "recurrent_test": svm_report["test_recurrent_matched"],
            "checkpoint": str(checkpoints["svm"].resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "comparison.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Winner: {winner}; best model: {result['best_model']}; "
        f"comparison: {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
