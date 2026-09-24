#!/usr/bin/env python3
"""Fine-tune a speaker-aware text classifier on the MELD splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device


MODEL_NAME = "bert-base-uncased"


def build_examples(
    rows: list[dict[str, str]], context_window: int
) -> list[dict[str, str]]:
    if context_window < 0:
        raise ValueError("context window cannot be negative")
    history: dict[str, list[str]] = defaultdict(list)
    examples = []
    for row in rows:
        label = row["Emotion"].lower()
        if label not in EMOTION_LABELS:
            raise ValueError(f"unknown MELD label: {label}")
        dialogue_id = row["Dialogue_ID"]
        turn = f"[{row['Speaker']}] {row['Utterance']}"
        prior = history[dialogue_id][-context_window:] if context_window else []
        parts = []
        if prior:
            parts.append("Context:\n" + "\n".join(prior))
        parts.append("Current:\n" + turn)
        examples.append(
            {
                "text": "\n".join(parts),
                "label": label,
                "dialogue_id": dialogue_id,
                "utterance_id": row["Utterance_ID"],
                "speaker": row["Speaker"],
                "utterance": row["Utterance"],
            }
        )
        history[dialogue_id].append(turn)
    return examples


def sqrt_class_weights(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=number_of_classes)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"training data is missing classes: {missing.tolist()}")
    weights = np.sqrt(len(labels) / counts).astype(np.float32)
    return weights / weights.mean()


class MeldTextDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict[str, str]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        return {
            "text": example["text"],
            "label": EMOTION_LABELS.index(example["label"]),
            "index": index,
        }


def make_collator(tokenizer, max_length: int):
    def collate(items):
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in items])
        encoded["indices"] = torch.tensor([item["index"] for item in items])
        return encoded

    return collate


def make_loader(
    examples, tokenizer, batch_size: int, max_length: int, shuffle: bool, seed: int
):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        MeldTextDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=make_collator(tokenizer, max_length),
    )


def evaluate(model, loader, device):
    actual_ids = []
    predicted_ids = []
    confidences = []
    indices = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch_indices = batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            actual_ids.extend(labels.tolist())
            predicted_ids.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            indices.extend(batch_indices.tolist())
    actual = [EMOTION_LABELS[index] for index in actual_ids]
    predicted = [EMOTION_LABELS[index] for index in predicted_ids]
    return compute_metrics(actual, predicted), predicted, confidences, indices


def write_predictions(path, examples, predicted, confidences, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = (
            "dialogue_id", "utterance_id", "speaker", "utterance",
            "expected", "predicted", "confidence",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for prediction, confidence, index in zip(predicted, confidences, indices):
            example = examples[index]
            writer.writerow(
                {
                    "dialogue_id": example["dialogue_id"],
                    "utterance_id": example["utterance_id"],
                    "speaker": example["speaker"],
                    "utterance": example["utterance"],
                    "expected": example["label"],
                    "predicted": prediction,
                    "confidence": f"{confidence:.6f}",
                }
            )


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "research/experiments/training-output-text")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def validate_arguments(args) -> None:
    positive = {
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "gradient accumulation": args.gradient_accumulation,
        "max length": args.max_length,
        "patience": args.patience,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if args.context_window < 0:
        raise ValueError("context window cannot be negative")


def main() -> int:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(EMOTION_LABELS),
        id2label={index: label for index, label in enumerate(EMOTION_LABELS)},
        label2id={label: index for index, label in enumerate(EMOTION_LABELS)},
    ).to(device)

    examples = {}
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]
        examples[split] = build_examples(rows, args.context_window)
    train_label_ids = np.array(
        [EMOTION_LABELS.index(item["label"]) for item in examples["train"]]
    )
    class_weights = torch.from_numpy(
        sqrt_class_weights(train_label_ids, len(EMOTION_LABELS))
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    loaders = {
        split: make_loader(
            examples[split], tokenizer, args.batch_size, args.max_length,
            split == "train", args.seed,
        )
        for split in examples
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = (len(loaders["train"]) + args.gradient_accumulation - 1) // args.gradient_accumulation
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "best-model"
    history = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    print(
        f"Device: {device}; training samples: {len(examples['train'])}; "
        f"context window: {args.context_window}; effective batch size: "
        f"{args.batch_size * args.gradient_accumulation}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loaders["train"], start=1):
            labels = batch.pop("labels").to(device)
            batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, _, _, _ = evaluate(model, loaders["dev"], device)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={row['train_loss']:.4f} "
            f"dev_macro_f1={row['dev_macro_f1']:.4f}"
        )
        if row["dev_macro_f1"] > best_f1:
            best_f1 = row["dev_macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    test_metrics, predicted, confidences, indices = evaluate(
        model, loaders["test"], device
    )
    write_predictions(
        args.output_dir / "test_predictions.csv",
        examples["test"], predicted, confidences, indices,
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": vars(args) | {
            "raw_archive": str(args.raw_archive),
            "output_dir": str(args.output_dir),
        },
        "model": args.model,
        "labels": EMOTION_LABELS,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "runtime_seconds": time.perf_counter() - started,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Best development macro F1: {best_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
