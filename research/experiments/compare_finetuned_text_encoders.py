#!/usr/bin/env python3
"""Fine-tune and compare BERT-base with E5-large-v2 on MELD text."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import build_examples, sqrt_class_weights, write_predictions


ROOT = Path(__file__).resolve().parents[2]
MODEL_SPECS = {
    "bert": {"model_id": "bert-base-uncased", "pooling": "pooler", "batch_size": 8},
    "e5": {"model_id": "intfloat/e5-large-v2", "pooling": "mean", "batch_size": 2},
}


def masked_mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def format_text(text, model_name):
    if model_name == "e5":
        return f"query: {text}"
    if model_name == "bert":
        return text
    raise ValueError(f"unknown model name: {model_name}")


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


class EncoderClassifier(torch.nn.Module):
    def __init__(self, encoder, pooling, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(
            encoder.config.hidden_size, len(EMOTION_LABELS)
        )

    def forward(self, input_ids, attention_mask, **inputs):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **inputs,
            return_dict=True,
        )
        if self.pooling == "mean":
            pooled = masked_mean_pool(output.last_hidden_state, attention_mask)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        elif self.pooling == "pooler":
            pooled = output.pooler_output
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
        else:
            raise ValueError(f"unknown pooling strategy: {self.pooling}")
        return SimpleNamespace(logits=self.classifier(self.dropout(pooled)))


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {
            "text": self.examples[index]["text"],
            "label": EMOTION_LABELS.index(self.examples[index]["label"]),
            "index": index,
        }


def make_loader(examples, tokenizer, model_name, batch_size, args, training):
    def collate(items):
        encoded = tokenizer(
            [format_text(item["text"], model_name) for item in items],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in items])
        encoded["indices"] = torch.tensor([item["index"] for item in items])
        return encoded

    return torch.utils.data.DataLoader(
        TextDataset(examples),
        batch_size=batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate,
    )


def synchronize(device):
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, device, measure_latency=False):
    actual_ids, predicted_ids, confidences, indices, latencies = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch_indices = batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            if measure_latency:
                synchronize(device)
                started = time.perf_counter()
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)
            if measure_latency:
                synchronize(device)
                latencies.append(
                    (time.perf_counter() - started) * 1000 / len(labels)
                )
            confidence, prediction = probabilities.max(dim=-1)
            actual_ids.extend(labels.tolist())
            predicted_ids.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            indices.extend(batch_indices.tolist())
    actual = [EMOTION_LABELS[index] for index in actual_ids]
    predicted = [EMOTION_LABELS[index] for index in predicted_ids]
    metrics = compute_metrics(actual, predicted)
    if latencies:
        metrics["median_latency_ms_per_example"] = float(np.median(latencies))
    return metrics, predicted, confidences, indices


def save_checkpoint(model, tokenizer, directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory / "encoder")
    tokenizer.save_pretrained(directory / "tokenizer")
    torch.save(model.classifier.state_dict(), directory / "classifier.pt")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_checkpoint(directory, device):
    from transformers import AutoModel, AutoTokenizer

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
    encoder = AutoModel.from_pretrained(directory / "encoder")
    model = EncoderClassifier(encoder, metadata["pooling"]).to(device)
    model.classifier.load_state_dict(
        torch.load(directory / "classifier.pt", map_location=device, weights_only=True)
    )
    return model, tokenizer, metadata


def train_one(model_name, learning_rate, examples, args, device):
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    spec = MODEL_SPECS[model_name]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    encoder = AutoModel.from_pretrained(spec["model_id"])
    if hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    model = EncoderClassifier(encoder, spec["pooling"]).to(device)
    batch_size = spec["batch_size"]
    gradient_accumulation = args.effective_batch_size // batch_size
    loaders = {
        split: make_loader(
            examples[split], tokenizer, model_name, batch_size, args, split == "train"
        )
        for split in ("train", "dev")
    }
    train_labels = np.asarray(
        [EMOTION_LABELS.index(example["label"]) for example in examples["train"]]
    )
    weights = torch.from_numpy(
        sqrt_class_weights(train_labels, len(EMOTION_LABELS))
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
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
            labels = batch.pop("labels").to(device)
            batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss = loss_function(model(**inputs).logits, labels)
            (loss / gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % gradient_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, _, _, _ = evaluate(model, loaders["dev"], device)
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
            save_checkpoint(
                model,
                tokenizer,
                checkpoint,
                {
                    "name": model_name,
                    "model_id": spec["model_id"],
                    "pooling": spec["pooling"],
                    "learning_rate": learning_rate,
                    "best_epoch": epoch,
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    result = {
        "name": model_name,
        "model_id": spec["model_id"],
        "pooling": spec["pooling"],
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
    del model, encoder, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return result


def evaluate_selected(run, examples, args, device):
    checkpoint = Path(run["checkpoint"])
    model, tokenizer, metadata = load_checkpoint(checkpoint, device)
    loader = make_loader(
        examples["test"],
        tokenizer,
        metadata["name"],
        MODEL_SPECS[metadata["name"]]["batch_size"],
        args,
        False,
    )
    metrics, predictions, confidences, indices = evaluate(
        model, loader, device, measure_latency=True
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
        default=ROOT / "research/experiments/results-finetuned-text-encoder-comparison",
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 2e-5])
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
    if len(args.learning_rates) != 2 or len(set(args.learning_rates)) != 2:
        raise ValueError("provide exactly two distinct learning rates")
    if any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("learning rates must be positive")
    if args.epochs < 1 or args.patience < 1 or args.effective_batch_size < 1:
        raise ValueError("training counts must be positive")
    for spec in MODEL_SPECS.values():
        if args.effective_batch_size % spec["batch_size"]:
            raise ValueError("effective batch size must divide both physical batch sizes")
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
    device = select_device(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = []
    all_runs = {}
    for model_name in ("bert", "e5"):
        runs = [
            train_one(model_name, rate, examples, args, device)
            for rate in args.learning_rates
        ]
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
            "context_window": args.context_window,
            "learning_rates": args.learning_rates,
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
