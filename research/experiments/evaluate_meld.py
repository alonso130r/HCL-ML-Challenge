#!/usr/bin/env python3
"""Evaluate the downloaded MELD audio+text checkpoint on a seeded test sample."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


EMOTION_LABELS = ("neutral", "joy", "surprise", "anger", "sadness", "fear", "disgust")
PREDICTION_COLUMNS = (
    "dialogue_id",
    "utterance_id",
    "utterance",
    "expected",
    "predicted",
    "confidence",
    "latency_ms",
    "status",
    "error",
)


def seeded_sample(rows: Sequence[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size < 1:
        raise ValueError("sample size must be at least 1")
    if sample_size > len(rows):
        raise ValueError(f"sample size {sample_size} exceeds the {len(rows)} available rows")
    selected = random.Random(seed).sample(list(enumerate(rows)), sample_size)
    return [row for _, row in sorted(selected, key=lambda item: item[0])]


def media_member_candidates(row: dict[str, str]) -> tuple[str, str]:
    stem = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    directory = "output_repeated_splits_test"
    return (f"{directory}/{stem}", f"{directory}/final_videos_test{stem}")


def _f1_for_label(actual: Sequence[str], predicted: Sequence[str], label: str) -> float:
    true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
    false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
    false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator


def compute_metrics(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, object]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted labels must be non-empty and equal in length")
    unknown = (set(actual) | set(predicted)) - set(EMOTION_LABELS)
    if unknown:
        raise ValueError(f"unknown emotion labels: {sorted(unknown)}")
    support = Counter(actual)
    per_class_f1 = {label: _f1_for_label(actual, predicted, label) for label in EMOTION_LABELS}
    count = len(actual)
    return {
        "labels": EMOTION_LABELS,
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / count,
        "macro_f1": sum(per_class_f1.values()) / len(EMOTION_LABELS),
        "weighted_f1": sum(per_class_f1[label] * support[label] for label in EMOTION_LABELS) / count,
        "per_class_f1": per_class_f1,
        "support": {label: support[label] for label in EMOTION_LABELS},
    }


def write_predictions(records: Iterable[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def prepare_test_sample(raw_archive: Path, sample_size: int, seed: int, workspace: Path) -> list[dict[str, str]]:
    csv_path = workspace / "test_sent_emo.csv"
    test_archive_path = workspace / "test.tar.gz"
    wanted = {
        "MELD.Raw/test_sent_emo.csv": csv_path,
        "MELD.Raw/test.tar.gz": test_archive_path,
    }
    found = set()
    # Stream the 10 GB outer archive once instead of repeatedly seeking through
    # a gzip stream. Only the two test-split members are copied to temporary disk.
    with tarfile.open(raw_archive, "r|gz") as outer:
        for member in outer:
            suffix = member.name.removeprefix("./")
            destination = wanted.get(suffix)
            if destination is None:
                continue
            source = outer.extractfile(member)
            if source is None:
                raise OSError(f"could not read {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            found.add(suffix)
    missing = set(wanted) - found
    if missing:
        raise FileNotFoundError(f"missing from {raw_archive}: {sorted(missing)}")

    with csv_path.open(newline="", encoding="cp1252") as stream:
        rows = list(csv.DictReader(stream))
    selected = seeded_sample(rows, sample_size, seed)

    clips_directory = workspace / "clips"
    clips_directory.mkdir()
    with tarfile.open(test_archive_path, "r:gz") as media_archive:
        members_by_name = {member.name.removeprefix("./"): member for member in media_archive.getmembers()}
        for row in selected:
            member = next(
                (members_by_name[name] for name in media_member_candidates(row) if name in members_by_name),
                None,
            )
            if member is None:
                row["_media_error"] = "clip not found in test archive"
                continue
            source = media_archive.extractfile(member)
            if source is None:
                row["_media_error"] = f"could not read {member.name}"
                continue
            clip_path = clips_directory / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            with source, clip_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            row["_clip_path"] = str(clip_path)
    return selected


def decode_audio(clip_path: Path, sampling_rate: int, max_duration_seconds: float):
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("audio decoding requires the packages listed in requirements.txt") from exc

    samples = []
    with av.open(str(clip_path)) as container:
        if not container.streams.audio:
            raise ValueError("clip has no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            for audio_frame in converted:
                samples.append(audio_frame.to_ndarray().reshape(-1))
        for audio_frame in resampler.resample(None):
            samples.append(audio_frame.to_ndarray().reshape(-1))
    if not samples:
        raise ValueError("decoded audio was empty")
    audio = np.concatenate(samples).astype("float32", copy=False)
    return audio[: int(sampling_rate * max_duration_seconds)]


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_inference_components(model_directory: Path):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise RuntimeError("model inference requires the packages listed in requirements.txt") from exc

    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    if config["classifier"]["num_classes"] != len(EMOTION_LABELS):
        raise ValueError("checkpoint class count does not match MELD's seven labels")
    if config["fusion"]["fusion_input_dim"] != 1536:
        raise ValueError("unsupported fusion input dimension")

    class FusionClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(1536, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(config["classifier"]["dropout"]),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(config["classifier"]["dropout"]),
                torch.nn.Linear(256, 7),
            )

        def forward(self, fused):
            return self.classifier(fused)

    checkpoint_path = model_directory / "pytorch_model.bin"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_shapes = {
        "classifier.0.weight": (512, 1536),
        "classifier.0.bias": (512,),
        "classifier.3.weight": (256, 512),
        "classifier.3.bias": (256,),
        "classifier.6.weight": (7, 256),
        "classifier.6.bias": (7,),
    }
    actual_shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(f"checkpoint architecture mismatch: {actual_shapes}")

    device = select_device(torch)
    classifier = FusionClassifier().to(device)
    classifier.load_state_dict(state, strict=True)
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["text"]["tokenizer"])
    text_encoder = AutoModel.from_pretrained(config["text"]["encoder"]).to(device).eval()
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(config["audio"]["encoder"])
    audio_encoder = AutoModel.from_pretrained(config["audio"]["encoder"]).to(device).eval()
    return torch, config, device, classifier, tokenizer, text_encoder, audio_processor, audio_encoder


def predict(row: dict[str, str], components) -> tuple[str, float]:
    torch, config, device, classifier, tokenizer, text_encoder, audio_processor, audio_encoder = components
    audio = decode_audio(
        Path(row["_clip_path"]),
        config["audio"]["sampling_rate"],
        config["audio"]["max_duration_sec"],
    )
    text_inputs = tokenizer(
        row["Utterance"],
        return_tensors="pt",
        truncation=True,
        max_length=config["text"]["max_sequence_length"],
    ).to(device)
    audio_inputs = audio_processor(
        audio,
        sampling_rate=config["audio"]["sampling_rate"],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        text_embedding = text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        audio_hidden = audio_encoder(**audio_inputs).last_hidden_state
        # The checkpoint expects 768 audio dimensions although its card calls the
        # pooling "mean+std". Elementwise addition is the only documented operation
        # consistent with the checkpoint's 1536-dimensional fusion input.
        audio_embedding = audio_hidden.mean(dim=1) + audio_hidden.std(dim=1, correction=0)
        fused = torch.cat((text_embedding, audio_embedding), dim=-1)
        probabilities = torch.softmax(classifier(fused), dim=-1)[0]
        index = int(probabilities.argmax().item())
    return EMOTION_LABELS[index], float(probabilities[index].item())


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--model-dir", type=Path, default=root / "models/meld-early-fusion-temporal")
    parser.add_argument("--output", type=Path, default=root / "research/experiments/results/predictions.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.raw_archive.is_file():
        raise FileNotFoundError(f"MELD archive not found: {args.raw_archive}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")

    print(f"Loading model from {args.model_dir}")
    components = load_inference_components(args.model_dir)
    print(f"Device: {components[2]}")
    print("Pooling compatibility assumption: audio embedding = temporal mean + temporal std")

    records = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="meld-evaluation-") as temporary_directory:
        print(f"Selecting {args.sample_size} test utterances with seed {args.seed}")
        rows = prepare_test_sample(args.raw_archive, args.sample_size, args.seed, Path(temporary_directory))
        for position, row in enumerate(rows, start=1):
            record = {
                "dialogue_id": row["Dialogue_ID"],
                "utterance_id": row["Utterance_ID"],
                "utterance": row["Utterance"],
                "expected": row["Emotion"].lower(),
                "predicted": "",
                "confidence": "",
                "latency_ms": "",
                "status": "failed",
                "error": row.get("_media_error", ""),
            }
            if "_clip_path" in row:
                inference_started = time.perf_counter()
                try:
                    predicted, confidence = predict(row, components)
                    record.update(
                        predicted=predicted,
                        confidence=f"{confidence:.6f}",
                        latency_ms=f"{(time.perf_counter() - inference_started) * 1000:.2f}",
                        status="ok",
                        error="",
                    )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            print(f"[{position:>3}/{len(rows)}] {record['status']}: dia{record['dialogue_id']}_utt{record['utterance_id']}")

    write_predictions(records, args.output)
    successful = [record for record in records if record["status"] == "ok"]
    failed = len(records) - len(successful)
    print(f"\nPredictions: {args.output}")
    print(f"Successful: {len(successful)}; failed: {failed}")
    if not successful:
        print("No valid predictions were produced. Inspect the CSV error column.")
        return 1
    metrics = compute_metrics(
        [str(record["expected"]) for record in successful],
        [str(record["predicted"]) for record in successful],
    )
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("Per-class F1 and support:")
    for label in EMOTION_LABELS:
        print(f"  {label:<8} {metrics['per_class_f1'][label]:.4f}  n={metrics['support'][label]}")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
