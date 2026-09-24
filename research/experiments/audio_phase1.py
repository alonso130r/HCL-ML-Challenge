#!/usr/bin/env python3
"""Build and evaluate stronger frozen audio representations for MELD."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_audio import prepare_audio_split
from train_text import sqrt_class_weights
from train_text_audio import extract_egemaps, sanitize_acoustic_features


EMOTION_MODEL = "iic/emotion2vec_plus_base"
PITCH_COLUMN = "F0semitoneFrom27.5Hz_sma3nz"
LOUDNESS_COLUMN = "Loudness_sma3"


def directory_only_path(value: str) -> str:
    return os.pathsep.join(
        entry for entry in value.split(os.pathsep) if Path(entry).is_dir()
    )


def summarize_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = sanitize_acoustic_features(sequence)
    if sequence.ndim != 2 or not len(sequence):
        raise ValueError("expected a nonempty frame-by-feature sequence")
    return np.concatenate((sequence.mean(axis=0), sequence.std(axis=0))).astype(
        np.float32, copy=False
    )


def build_prosody_contours(columns) -> np.ndarray:
    pitch = np.asarray(columns[PITCH_COLUMN], dtype=np.float32)
    loudness = np.asarray(columns[LOUDNESS_COLUMN], dtype=np.float32)
    if pitch.shape != loudness.shape or pitch.ndim != 1 or not len(pitch):
        raise ValueError("pitch and loudness contours must be equal nonempty vectors")
    pitch_delta = np.diff(pitch, prepend=pitch[0])
    loudness_delta = np.diff(loudness, prepend=loudness[0])
    voiced = (pitch > 0).astype(np.float32)
    return sanitize_acoustic_features(
        np.stack((pitch, loudness, pitch_delta, loudness_delta, voiced), axis=1)
    )


def extract_prosody_contours(waveform: np.ndarray) -> np.ndarray:
    import opensmile

    smile = getattr(extract_prosody_contours, "_smile", None)
    if smile is None:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        extract_prosody_contours._smile = smile
    frame = smile.process_signal(waveform, 16000)
    return build_prosody_contours(frame)


def causal_speaker_features(
    features: np.ndarray,
    records,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) != len(records):
        raise ValueError("features and records must have equal length")
    safe_global_std = np.where(global_std < 1e-6, 1.0, global_std)
    relative = np.empty_like(features, dtype=np.float32)
    history_size = np.zeros(len(records), dtype=np.float32)
    histories: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    ordered = sorted(
        range(len(records)),
        key=lambda index: (
            records[index]["dialogue_id"],
            int(records[index]["utterance_id"]),
        ),
    )
    for index in ordered:
        record = records[index]
        key = (record["dialogue_id"], record["speaker"])
        prior = histories[key]
        history_size[index] = len(prior)
        if prior:
            prior_matrix = np.stack(prior)
            center = prior_matrix.mean(axis=0)
            scale = prior_matrix.std(axis=0) if len(prior) > 1 else safe_global_std
            scale = np.where(scale < 1e-6, safe_global_std, scale)
        else:
            center, scale = global_mean, safe_global_std
        relative[index] = (features[index] - center) / scale
        prior.append(features[index])
    return sanitize_acoustic_features(relative), history_size


def combine_feature_views(emotion, acoustics, relative, history):
    if not (len(emotion) == len(acoustics) == len(relative) == len(history)):
        raise ValueError("all feature views must have equal row counts")
    return {
        "emotion": emotion.astype(np.float32, copy=False),
        "hybrid": np.concatenate((emotion, acoustics), axis=1),
        "speaker_relative": np.concatenate(
            (emotion, acoustics, relative, history[:, None]), axis=1
        ),
    }


def attach_metadata(rows, audio_records):
    by_key = {
        (record["dialogue_id"], record["utterance_id"]): record
        for record in audio_records
    }
    result = []
    for row in rows:
        key = (row["Dialogue_ID"], row["Utterance_ID"])
        if key not in by_key:
            continue
        result.append(
            by_key[key]
            | {
                "speaker": row["Speaker"],
                "label": row["Emotion"].lower(),
                "utterance": row["Utterance"],
            }
        )
    return result


class Emotion2VecExtractor:
    def __init__(self, model_id: str, device: str):
        original_path = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = directory_only_path(original_path)
            from funasr import AutoModel
        finally:
            os.environ["PATH"] = original_path

        self.wrapper = AutoModel(
            model=model_id,
            hub="hf",
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        result = self.wrapper.generate(
            input=waveform,
            granularity="frame",
            extract_embedding=True,
        )
        if not result or "feats" not in result[0]:
            raise RuntimeError("emotion2vec did not return frame features")
        features = sanitize_acoustic_features(np.asarray(result[0]["feats"]))
        if features.ndim != 2 or not len(features):
            raise ValueError(f"invalid emotion2vec frame shape: {features.shape}")
        return features


def cache_split_features(records, split_dir, extractor, rebuild):
    feature_dir = split_dir / "phase1"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for position, record in enumerate(records, start=1):
        destination = feature_dir / (
            f"dia{record['dialogue_id']}_utt{record['utterance_id']}.npz"
        )
        if rebuild or not destination.exists():
            waveform = np.load(record["audio_path"]).astype(np.float32)
            emotion_frames = extractor(waveform)
            prosody_frames = extract_prosody_contours(waveform)
            np.savez_compressed(
                destination,
                emotion_frames=emotion_frames.astype(np.float16),
                emotion_summary=summarize_sequence(emotion_frames),
                egemaps=extract_egemaps(waveform),
                prosody_frames=prosody_frames.astype(np.float16),
                prosody_summary=summarize_sequence(prosody_frames),
            )
        prepared.append(record | {"phase1_path": str(destination.resolve())})
        if position % 100 == 0:
            print(f"  cached {position}/{len(records)} Phase 1 clips", flush=True)
    return prepared


def load_summary_matrices(records):
    emotion, acoustics = [], []
    for record in records:
        with np.load(record["phase1_path"]) as cache:
            emotion.append(cache["emotion_summary"].astype(np.float32))
            acoustics.append(
                np.concatenate(
                    (
                        cache["egemaps"].astype(np.float32),
                        cache["prosody_summary"].astype(np.float32),
                    )
                )
            )
    return np.stack(emotion), np.stack(acoustics)


def fit_normalizer(features):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize(features, statistics):
    return sanitize_acoustic_features((features - statistics[0]) / statistics[1])


class DiagnosticProbe(torch.nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, len(EMOTION_LABELS)),
        )

    def forward(self, features):
        return self.network(features)


def make_loader(features, labels, batch_size, training, seed):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(features), torch.from_numpy(labels)
        ),
        batch_size=batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate_probe(model, loader, device):
    actual, predicted = [], []
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            prediction = model(features.to(device)).argmax(dim=-1).cpu()
            actual.extend(labels.tolist())
            predicted.extend(prediction.tolist())
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    return compute_metrics(names(actual), names(predicted))


def train_probe(name, train, dev, test, labels, args, device, output_dir):
    statistics = fit_normalizer(train)
    train, dev, test = (
        normalize(features, statistics) for features in (train, dev, test)
    )
    loaders = {
        "train": make_loader(train, labels["train"], args.batch_size, True, args.seed),
        "dev": make_loader(dev, labels["dev"], args.batch_size, False, args.seed),
        "test": make_loader(test, labels["test"], args.batch_size, False, args.seed),
    }
    model = DiagnosticProbe(train.shape[1], args.dropout).to(device)
    weights = torch.from_numpy(
        sqrt_class_weights(labels["train"], len(EMOTION_LABELS))
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best, stale = -1.0, 0
    checkpoint = output_dir / f"best_{name}.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_probe(model, loaders["dev"], device)
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "dev_macro_f1": metrics["macro_f1"],
                "dev_weighted_f1": metrics["weighted_f1"],
            }
        )
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return {
        "dimension": train.shape[1],
        "best_dev_macro_f1": best,
        "test_metrics": evaluate_probe(model, loaders["test"], device),
        "history": history,
    }


def parse_arguments():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "research/experiments/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "research/experiments/training-output-audio-phase1")
    parser.add_argument("--emotion-model", default=EMOTION_MODEL)
    parser.add_argument("--extract-device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-audio-cache", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = Emotion2VecExtractor(args.emotion_model, args.extract_device)
    records = {}
    started = time.perf_counter()
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        if args.max_samples_per_split is not None:
            rows = rows[:args.max_samples_per_split]
        audio = prepare_audio_split(
            args.raw_archive, split, args.audio_cache_dir,
            args.max_samples_per_split, args.rebuild_audio_cache,
        )
        combined = attach_metadata(rows, audio)
        records[split] = cache_split_features(
            combined, args.audio_cache_dir / split, extractor,
            args.rebuild_feature_cache,
        )

    summaries = {split: load_summary_matrices(value) for split, value in records.items()}
    acoustic_statistics = fit_normalizer(summaries["train"][1])
    views = {}
    labels = {}
    for split in records:
        emotion, acoustics = summaries[split]
        relative, history = causal_speaker_features(
            acoustics, records[split], *acoustic_statistics
        )
        views[split] = combine_feature_views(
            emotion, acoustics, relative, np.log1p(history)
        )
        labels[split] = np.array(
            [EMOTION_LABELS.index(record["label"]) for record in records[split]],
            dtype=np.int64,
        )

    device = select_device(torch)
    results = {}
    for name in ("emotion", "hybrid", "speaker_relative"):
        print(f"Training {name} diagnostic probe on {device}")
        results[name] = train_probe(
            name,
            views["train"][name], views["dev"][name], views["test"][name],
            labels, args, device, args.output_dir,
        )
        metrics = results[name]["test_metrics"]
        print(
            f"  test macro F1={metrics['macro_f1']:.4f} "
            f"weighted F1={metrics['weighted_f1']:.4f}"
        )
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {split: len(value) for split, value in records.items()},
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
