#!/usr/bin/env python3
"""Train and evaluate a fusion MLP from cached MELD embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import FEATURE_DIMENSION, load_cache
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device


TEXT_DIMENSION = 768


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"expected features with {FEATURE_DIMENSION} columns")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = features.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def apply_normalizer(
    features: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / standard_deviation).astype(np.float32, copy=False)


def add_text_context(
    features: np.ndarray,
    metadata: dict[str, list[str]],
    window: int,
) -> np.ndarray:
    if window < 0:
        raise ValueError("context window cannot be negative")
    if window == 0:
        return features
    context = np.zeros((len(features), TEXT_DIMENSION), dtype=np.float32)
    dialogue_indices: dict[str, list[int]] = {}
    for index, dialogue_id in enumerate(metadata["dialogue_ids"]):
        dialogue_indices.setdefault(dialogue_id, []).append(index)
    for indices in dialogue_indices.values():
        indices.sort(key=lambda index: int(metadata["utterance_ids"][index]))
        for position, index in enumerate(indices):
            previous = indices[max(0, position - window):position]
            if previous:
                context[index] = features[previous, :TEXT_DIMENSION].mean(axis=0)
    return np.concatenate((features, context), axis=1)


def compute_class_weights(label_ids: np.ndarray, number_of_classes: int) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=number_of_classes)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"training data is missing classes: {missing.tolist()}")
    return torch.tensor(len(label_ids) / (number_of_classes * counts), dtype=torch.float32)


def is_better_checkpoint(candidate_macro_f1: float, best_macro_f1: float) -> bool:
    return candidate_macro_f1 > best_macro_f1


class FusionMlp(torch.nn.Module):
    def __init__(self, input_dimension: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, len(EMOTION_LABELS)),
        )

    def forward(self, features):
        return self.classifier(features)


def load_split(path: Path):
    with load_cache(path) as cache:
        features = cache["features"].astype(np.float32)
        labels = cache["labels"].tolist()
        metadata = {
            "dialogue_ids": cache["dialogue_ids"].tolist(),
            "utterance_ids": cache["utterance_ids"].tolist(),
            "utterances": cache["utterances"].tolist(),
        }
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"invalid cache feature shape in {path}: {features.shape}")
    unknown = set(labels) - set(EMOTION_LABELS)
    if unknown:
        raise ValueError(f"unknown labels in {path}: {sorted(unknown)}")
    label_ids = np.array([EMOTION_LABELS.index(label) for label in labels], dtype=np.int64)
    return features, label_ids, labels, metadata


def make_loader(features, label_ids, batch_size: int, shuffle: bool, seed: int):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features), torch.from_numpy(label_ids)
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def evaluate(model, loader, device):
    predictions = []
    actual = []
    confidences = []
    latencies = []
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            started = time.perf_counter()
            probabilities = torch.softmax(model(features), dim=-1)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            confidence, prediction = probabilities.max(dim=-1)
            predictions.extend(prediction.cpu().tolist())
            actual.extend(labels.tolist())
            confidences.extend(confidence.cpu().tolist())
            latencies.extend([elapsed_ms / len(labels)] * len(labels))
    actual_names = [EMOTION_LABELS[index] for index in actual]
    predicted_names = [EMOTION_LABELS[index] for index in predictions]
    return compute_metrics(actual_names, predicted_names), predicted_names, confidences, latencies


def write_predictions(path, metadata, expected, predicted, confidence, latency):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "dialogue_id", "utterance_id", "utterance", "expected", "predicted",
        "confidence", "mlp_latency_ms",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index in range(len(expected)):
            writer.writerow(
                {
                    "dialogue_id": metadata["dialogue_ids"][index],
                    "utterance_id": metadata["utterance_ids"][index],
                    "utterance": metadata["utterances"][index],
                    "expected": expected[index],
                    "predicted": predicted[index],
                    "confidence": f"{confidence[index]:.6f}",
                    "mlp_latency_ms": f"{latency[index]:.4f}",
                }
            )


def serializable_metrics(metrics):
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in metrics.items()
    }


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "research/experiments/cache")
    parser.add_argument("--output-dir", type=Path, default=root / "research/experiments/training-output-context")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument(
        "--no-normalization",
        action="store_false",
        dest="normalization",
        help="disable train-statistics feature normalization",
    )
    parser.set_defaults(normalization=True)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_split(args.cache_dir / "train.npz")
    dev = load_split(args.cache_dir / "dev.npz")
    test = load_split(args.cache_dir / "test.npz")
    if args.context_window < 0:
        raise ValueError("context window cannot be negative")
    if args.normalization:
        normalization_mean, normalization_std = fit_normalizer(train[0])
        train_features = apply_normalizer(train[0], normalization_mean, normalization_std)
        dev_features = apply_normalizer(dev[0], normalization_mean, normalization_std)
        test_features = apply_normalizer(test[0], normalization_mean, normalization_std)
    else:
        normalization_mean = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
        normalization_std = np.ones(FEATURE_DIMENSION, dtype=np.float32)
        train_features, dev_features, test_features = train[0], dev[0], test[0]
    train_features = add_text_context(train_features, train[3], args.context_window)
    dev_features = add_text_context(dev_features, dev[3], args.context_window)
    test_features = add_text_context(test_features, test[3], args.context_window)
    input_dimension = train_features.shape[1]
    device = select_device(torch)
    model = FusionMlp(input_dimension, args.dropout).to(device)
    weights = compute_class_weights(train[1], len(EMOTION_LABELS)).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(train_features, train[1], args.batch_size, True, args.seed)
    dev_loader = make_loader(dev_features, dev[1], args.batch_size, False, args.seed)
    test_loader = make_loader(test_features, test[1], args.batch_size, False, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_mlp.pt"
    history = []
    best_f1 = -1.0
    epochs_without_improvement = 0

    print(
        f"Device: {device}; training samples: {len(train[1])}; "
        f"input features: {input_dimension}; context window: {args.context_window}; "
        f"normalization: {args.normalization}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device)
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
        if is_better_checkpoint(row["dev_macro_f1"], best_f1):
            best_f1 = row["dev_macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": EMOTION_LABELS,
                    "feature_dimension": input_dimension,
                    "base_feature_dimension": FEATURE_DIMENSION,
                    "audio_pooling": "speechbrain_masked_temporal_mean",
                    "context_window": args.context_window,
                    "context_feature": "mean_of_prior_standardized_text_embeddings",
                    "normalization": args.normalization,
                    "normalization_mean": torch.from_numpy(normalization_mean),
                    "normalization_std": torch.from_numpy(normalization_std),
                    "epoch": epoch,
                    "dev_macro_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics, predicted, confidence, latency = evaluate(model, test_loader, device)
    write_predictions(
        args.output_dir / "test_predictions.csv",
        test[3],
        test[2],
        predicted,
        confidence,
        latency,
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": vars(args) | {"cache_dir": str(args.cache_dir), "output_dir": str(args.output_dir)},
        "best_epoch": checkpoint["epoch"],
        "best_dev_macro_f1": checkpoint["dev_macro_f1"],
        "test_metrics": serializable_metrics(test_metrics),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best development macro F1: {checkpoint['dev_macro_f1']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
