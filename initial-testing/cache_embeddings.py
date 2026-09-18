#!/usr/bin/env python3
"""Cache BERT text and SpeechBrain emotion audio embeddings for MELD."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from evaluate_meld import EMOTION_LABELS, decode_audio, select_device


TEXT_MODEL = "bert-base-uncased"
AUDIO_MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
TEXT_DIMENSION = 768
AUDIO_DIMENSION = 768
FEATURE_DIMENSION = TEXT_DIMENSION + AUDIO_DIMENSION
MEDIA_PATTERN = re.compile(r"dia(\d+)_utt(\d+)\.mp4$")


def media_key(member_name: str) -> tuple[str, str] | None:
    filename = Path(member_name).name
    if not filename.startswith("dia"):
        return None
    match = MEDIA_PATTERN.fullmatch(filename)
    return None if match is None else (match.group(1), match.group(2))


def save_cache(
    path: Path,
    split: str,
    features: np.ndarray,
    records: list[dict[str, str]],
    failures: list[dict[str, str]] | None = None,
    source_count: int | None = None,
) -> None:
    if features.shape != (len(records), FEATURE_DIMENSION):
        raise ValueError(
            f"expected cache shape ({len(records)}, {FEATURE_DIMENSION}), got {features.shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split=np.array(split),
        features=features.astype(np.float32, copy=False),
        labels=np.array([record["label"] for record in records]),
        dialogue_ids=np.array([record["dialogue_id"] for record in records]),
        utterance_ids=np.array([record["utterance_id"] for record in records]),
        utterances=np.array([record["utterance"] for record in records]),
        text_model=np.array(TEXT_MODEL),
        audio_model=np.array(AUDIO_MODEL),
        audio_pooling=np.array("speechbrain_masked_temporal_mean"),
        failures_json=np.array(json.dumps(failures or [])),
        source_count=np.array(source_count if source_count is not None else len(records)),
    )


def load_cache(path: Path):
    return np.load(path, allow_pickle=False)


def decode_clip(clip: Path):
    try:
        return decode_audio(clip, 16000, 6.0), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_csv(stream) -> list[dict[str, str]]:
    text = stream.read().decode("cp1252")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def load_split_rows(raw_archive: Path, split: str) -> list[dict[str, str]]:
    csv_name = f"{split}_sent_emo.csv"
    with tarfile.open(raw_archive, "r|gz") as outer:
        for member in outer:
            if split == "train" and member.name.endswith("train.tar.gz"):
                nested_stream = outer.extractfile(member)
                if nested_stream is None:
                    break
                with tarfile.open(fileobj=nested_stream, mode="r|gz") as nested:
                    for nested_member in nested:
                        if nested_member.name.endswith(csv_name):
                            source = nested.extractfile(nested_member)
                            if source is not None:
                                return _read_csv(source)
            elif split != "train" and member.name.endswith(csv_name):
                source = outer.extractfile(member)
                if source is not None:
                    return _read_csv(source)
    raise FileNotFoundError(f"could not find {csv_name} in {raw_archive}")


def normalize_rows(rows: list[dict[str, str]], maximum: int | None) -> list[dict[str, str]]:
    selected = rows if maximum is None else rows[:maximum]
    normalized = []
    for row in selected:
        label = row["Emotion"].lower()
        if label not in EMOTION_LABELS:
            raise ValueError(f"unknown MELD label: {label}")
        normalized.append(
            {
                "label": label,
                "dialogue_id": row["Dialogue_ID"],
                "utterance_id": row["Utterance_ID"],
                "utterance": row["Utterance"],
            }
        )
    return normalized


def load_encoders():
    import torch
    from speechbrain.inference.interfaces import foreign_class
    from transformers import AutoModel, AutoTokenizer

    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    text_encoder = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    audio_encoder = foreign_class(
        source=AUDIO_MODEL,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": str(device)},
    )
    return torch, device, tokenizer, text_encoder, audio_encoder


def encode_batch(batch, components) -> np.ndarray:
    torch, device, tokenizer, text_encoder, audio_encoder = components
    texts = [item["record"]["utterance"] for item in batch]
    audio_lengths = torch.tensor(
        [len(item["audio"]) for item in batch], dtype=torch.long, device=device
    )
    maximum_audio_length = int(audio_lengths.max())
    audio = torch.zeros((len(batch), maximum_audio_length), device=device)
    for index, item in enumerate(batch):
        samples = torch.from_numpy(item["audio"]).to(device)
        audio[index, : len(samples)] = samples
    relative_audio_lengths = audio_lengths / maximum_audio_length
    text_inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        text_features = text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        audio_features = audio_encoder.encode_batch(audio, relative_audio_lengths)
        features = torch.cat((text_features, audio_features), dim=-1)
    if features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"unexpected fused feature shape: {tuple(features.shape)}")
    return features.float().cpu().numpy()


def cache_split(
    raw_archive: Path,
    split: str,
    output: Path,
    batch_size: int,
    maximum: int | None,
    components,
) -> None:
    records = normalize_rows(load_split_rows(raw_archive, split), maximum)
    wanted = {
        (record["dialogue_id"], record["utterance_id"]): index
        for index, record in enumerate(records)
    }
    features_by_index = {}
    failures_by_index = {}
    pending = []

    def flush() -> None:
        if not pending:
            return
        encoded = encode_batch(pending, components)
        for item, feature in zip(pending, encoded):
            features_by_index[item["index"]] = feature
        pending.clear()
        print(f"  encoded {len(features_by_index)}/{len(records)}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"meld-{split}-") as directory:
        temp_directory = Path(directory)
        with tarfile.open(raw_archive, "r|gz") as outer:
            nested_name = f"{split}.tar.gz"
            for member in outer:
                if not member.name.endswith(nested_name):
                    continue
                nested_stream = outer.extractfile(member)
                if nested_stream is None:
                    raise OSError(f"could not read {member.name}")
                with tarfile.open(fileobj=nested_stream, mode="r|gz") as nested:
                    for media_member in nested:
                        key = media_key(media_member.name)
                        if key is None:
                            continue
                        index = wanted.get(key)
                        if index is None or index in features_by_index:
                            continue
                        source = nested.extractfile(media_member)
                        if source is None:
                            raise OSError(f"could not read {media_member.name}")
                        clip = temp_directory / f"{split}-{key[0]}-{key[1]}.mp4"
                        with source, clip.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                        audio, error = decode_clip(clip)
                        clip.unlink(missing_ok=True)
                        if error is not None:
                            failures_by_index[index] = error
                            print(
                                f"  skipped corrupt media: dia{key[0]}_utt{key[1]} ({error})",
                                flush=True,
                            )
                            continue
                        pending.append({"index": index, "record": records[index], "audio": audio})
                        if len(pending) >= batch_size:
                            flush()
                    flush()
                break

    missing = sorted(
        set(range(len(records))) - set(features_by_index) - set(failures_by_index)
    )
    for index in missing:
        failures_by_index[index] = "media file not found in split archive"
    successful_indices = sorted(features_by_index)
    successful_records = [records[index] for index in successful_indices]
    feature_matrix = np.stack([features_by_index[index] for index in successful_indices])
    failures = [
        records[index] | {"error": failures_by_index[index]}
        for index in sorted(failures_by_index)
    ]
    save_cache(
        output,
        split,
        feature_matrix,
        successful_records,
        failures=failures,
        source_count=len(records),
    )
    failure_path = output.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(
        f"Saved {len(successful_records)} {split} embeddings to {output}; "
        f"excluded {len(failures)} source records"
    )


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/cache")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("batch size must be at least 1")
    pending_splits = []
    for split in args.splits:
        output = args.output_dir / f"{split}.npz"
        if output.exists() and not args.force:
            print(f"Skipping completed split: {output}")
        else:
            pending_splits.append((split, output))
    if not pending_splits:
        return 0
    components = load_encoders()
    print(f"Device: {components[1]}")
    for split, output in pending_splits:
        cache_split(
            args.raw_archive,
            split,
            output,
            args.batch_size,
            args.max_samples_per_split,
            components,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
