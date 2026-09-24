#!/usr/bin/env python3
"""Compare BERT, GoEmotions-RoBERTa, and DeBERTa on MELD text."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import (
    build_examples,
    make_loader,
    sqrt_class_weights,
    write_predictions,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_SPECS = {
    "bert": {"model_id": "bert-base-uncased", "batch_size": 8},
    "roberta_go_emotions": {
        "model_id": "SamLowe/roberta-base-go_emotions",
        "batch_size": 16,
    },
    "deberta": {"model_id": "microsoft/deberta-v3-base", "batch_size": 8},
}


def model_load_settings(model_name):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model_name}")
    return {
        "num_labels": len(EMOTION_LABELS),
        "id2label": {index: label for index, label in enumerate(EMOTION_LABELS)},
        "label2id": {label: index for index, label in enumerate(EMOTION_LABELS)},
        "problem_type": "single_label_classification",
        "ignore_mismatched_sizes": True,
    }


def select_model_run(runs):
    if not runs:
        raise ValueError("no model runs were completed")
    return max(
        runs,
        key=lambda run: (
            run["best_dev"]["macro_f1"],
            run["best_dev"]["weighted_f1"],
            -run["learning_rate"],
        ),
    )


def choose_winner(candidates, macro_tolerance=0.005):
    if not candidates:
        raise ValueError("no encoder candidates were completed")
    best_macro = max(candidate["dev"]["macro_f1"] for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate["dev"]["macro_f1"] >= best_macro - macro_tolerance
    ]
    return max(
        eligible,
        key=lambda candidate: (
            candidate["dev"]["weighted_f1"],
            candidate["dev"]["macro_f1"],
        ),
    )


def synchronize(device):
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def weighted_cross_entropy(logits, labels, weights):
    return torch.nn.functional.cross_entropy(
        logits.float(), labels, weight=weights.float()
    )


def evaluate_with_latency(model, loader, device):
    actual_ids, predicted_ids, confidences, indices, latencies = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch_indices = batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            synchronize(device)
            started = time.perf_counter()
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000 / len(labels))
            confidence, prediction = probabilities.max(dim=-1)
            actual_ids.extend(labels.tolist())
            predicted_ids.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            indices.extend(batch_indices.tolist())
    actual = [EMOTION_LABELS[index] for index in actual_ids]
    predicted = [EMOTION_LABELS[index] for index in predicted_ids]
    metrics = compute_metrics(actual, predicted)
    metrics["median_latency_ms_per_example"] = float(np.median(latencies))
    return metrics, predicted, confidences, indices


def train_one(model_name, learning_rate, examples, args, device):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    spec = MODEL_SPECS[model_name]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model_id"], **model_load_settings(model_name)
    )
    model = model.to(device)
    batch_size = spec["batch_size"]
    gradient_accumulation = args.effective_batch_size // batch_size
    loaders = {
        split: make_loader(
            examples[split],
            tokenizer,
            batch_size,
            args.max_length,
            split == "train",
            args.seed,
        )
        for split in ("train", "dev")
    }
    labels = np.asarray(
        [EMOTION_LABELS.index(example["label"]) for example in examples["train"]]
    )
    weights = torch.from_numpy(sqrt_class_weights(labels, len(EMOTION_LABELS))).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    updates = int(np.ceil(len(loaders["train"]) / gradient_accumulation))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(updates * args.epochs * args.warmup_ratio),
        num_training_steps=updates * args.epochs,
    )
    run_name = f"{model_name}-lr-{learning_rate:.0e}".replace("+", "")
    run_dir = args.output_dir / "runs" / run_name
    checkpoint = run_dir / "best-model"
    history, best_dev, best_epoch, stale = [], None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loaders["train"], start=1):
            batch_labels = batch.pop("labels").to(device)
            batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss = weighted_cross_entropy(
                model(**inputs).logits, batch_labels, weights
            )
            (loss / gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % gradient_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, _, _, _ = evaluate_without_latency(model, loaders["dev"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev": dev_metrics,
        }
        history.append(row)
        print(
            f"{run_name} epoch {epoch}: loss={row['train_loss']:.4f} "
            f"dev_macro={dev_metrics['macro_f1']:.4f} "
            f"dev_weighted={dev_metrics['weighted_f1']:.4f}",
            flush=True,
        )
        score = (dev_metrics["macro_f1"], dev_metrics["weighted_f1"])
        best_score = (
            (-1.0, -1.0)
            if best_dev is None
            else (best_dev["macro_f1"], best_dev["weighted_f1"])
        )
        if score > best_score:
            best_dev, best_epoch, stale = dev_metrics, epoch, 0
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping {run_name} after epoch {epoch}", flush=True)
                break
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    result = {
        "name": model_name,
        "model_id": spec["model_id"],
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "best_dev": best_dev,
        "checkpoint": str(checkpoint.resolve()),
        "training_seconds": time.perf_counter() - started,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    del model, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return result


def evaluate_without_latency(model, loader, device):
    actual_ids, predicted_ids, confidences, indices = [], [], [], []
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


def evaluate_selected(run, examples, args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = Path(run["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    loader = make_loader(
        examples["test"],
        tokenizer,
        MODEL_SPECS[run["name"]]["batch_size"],
        args.max_length,
        False,
        args.seed,
    )
    metrics, predictions, confidences, indices = evaluate_with_latency(
        model, loader, device
    )
    write_predictions(
        checkpoint.parent / "test_predictions.csv",
        examples["test"],
        predictions,
        confidences,
        indices,
    )
    del model, tokenizer
    return metrics


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "research/experiments/results-finetuned-classification-encoder-comparison",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=["roberta_go_emotions", "deberta"],
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--macro-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args(argv)


def validate_arguments(args):
    if args.context_window != 2:
        raise ValueError("this comparison requires context window two")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must be distinct")
    if args.epochs < 1 or args.patience < 1 or args.effective_batch_size < 1:
        raise ValueError("training counts must be positive")
    for name in args.models:
        if args.effective_batch_size % MODEL_SPECS[name]["batch_size"]:
            raise ValueError("effective batch size must divide every physical batch size")
    if not 0 <= args.macro_tolerance <= 0.05:
        raise ValueError("macro tolerance must be between zero and 0.05")
    if not args.raw_archive.is_file():
        raise FileNotFoundError(args.raw_archive)


def main(argv=None):
    args = parse_arguments(argv)
    validate_arguments(args)
    examples = {}
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        if args.max_samples_per_split:
            rows = rows[: args.max_samples_per_split]
        examples[split] = build_examples(rows, args.context_window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(torch)
    selected_models, all_runs = [], {}
    for model_name in args.models:
        runs = [train_one(model_name, args.learning_rate, examples, args, device)]
        all_runs[model_name] = runs
        selected = select_model_run(runs)
        selected["test"] = evaluate_selected(selected, examples, args, device)
        selected_models.append(
            {"name": model_name, "dev": selected["best_dev"], "run": selected}
        )
    winner = choose_winner(selected_models, args.macro_tolerance)
    report = {
        "selection_rule": (
            "Highest development weighted F1 among models within macro tolerance "
            "of the best development macro F1."
        ),
        "configuration": {
            "models": args.models,
            "context_window": args.context_window,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "patience": args.patience,
            "effective_batch_size": args.effective_batch_size,
            "seed": args.seed,
            "macro_tolerance": args.macro_tolerance,
        },
        "runs": all_runs,
        "selected_models": selected_models,
        "winner": winner["name"],
        "best_model": winner["run"]["checkpoint"],
    }
    destination = args.output_dir / "comparison.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Winner: {report['winner']}; best model: {report['best_model']}; "
        f"comparison: {destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
