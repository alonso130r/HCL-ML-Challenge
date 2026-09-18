#!/usr/bin/env python3
"""Cache facial-expression frame embeddings and train a MELD video classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import load_split_rows, media_key
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_audio import AttentiveStatisticsPooling


VIDEO_MODEL = "trpakov/vit-face-expression"
EMBEDDING_DIMENSION = 768


def sample_frame_indices(total_frames: int, requested_frames: int) -> list[int]:
    if total_frames < 1 or requested_frames < 1:
        raise ValueError("frame counts must be positive")
    count = min(total_frames, requested_frames)
    return np.linspace(0, total_frames - 1, count, dtype=int).tolist()


def select_face(image: np.ndarray, boxes, margin: float = 0.15) -> np.ndarray:
    height, width = image.shape[:2]
    if not boxes:
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return image[top : top + side, left : left + side]
    x, y, box_width, box_height = max(boxes, key=lambda box: box[2] * box[3])
    padding_x = int(box_width * margin)
    padding_y = int(box_height * margin)
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(width, x + box_width + padding_x)
    bottom = min(height, y + box_height + padding_y)
    return image[top:bottom, left:right]


def pad_video_embeddings(sequences):
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("video batch cannot contain empty sequences")
    maximum = max(len(sequence) for sequence in sequences)
    dimension = sequences[0].shape[1]
    padded = torch.zeros((len(sequences), maximum, dimension), dtype=torch.float32)
    padding_mask = torch.ones((len(sequences), maximum), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        tensor = torch.from_numpy(sequence.astype(np.float32, copy=False))
        padded[index, : len(sequence)] = tensor
        padding_mask[index, : len(sequence)] = False
    return padded, padding_mask


def decode_sampled_faces(clip: Path, frame_count: int, face_detector):
    import cv2

    frames = []
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise ValueError("could not open video stream")
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError("decoded video was empty")
    crops = []
    detections = 0
    for index in sample_frame_indices(len(frames), frame_count):
        image = frames[index]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(boxes):
            detections += 1
        crops.append(select_face(image, list(boxes)))
    return crops, detections


def load_visual_encoder(device):
    import cv2
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(VIDEO_MODEL)
    encoder = AutoModelForImageClassification.from_pretrained(VIDEO_MODEL).to(device).eval()
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"could not load face detector: {cascade}")
    return processor, encoder, detector


def encode_faces(crops, processor, encoder, device):
    from PIL import Image

    images = [Image.fromarray(crop) for crop in crops]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = encoder(**inputs, output_hidden_states=True)
        embeddings = output.hidden_states[-1][:, 0, :]
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"unexpected visual embedding shape: {tuple(embeddings.shape)}")
    return embeddings.float().cpu().numpy()


def normalized_records(rows, maximum):
    selected = rows if maximum is None else rows[:maximum]
    records = []
    for row in selected:
        label = row["Emotion"].lower()
        if label not in EMOTION_LABELS:
            raise ValueError(f"unknown MELD label: {label}")
        records.append(
            {
                "dialogue_id": row["Dialogue_ID"],
                "utterance_id": row["Utterance_ID"],
                "utterance": row["Utterance"],
                "label": label,
            }
        )
    return records


def prepare_video_split(
    raw_archive, split, cache_dir, frame_count, maximum, rebuild, components
):
    split_dir = cache_dir / split
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("video_model") == VIDEO_MODEL
            and manifest.get("frame_count") == frame_count
            and manifest.get("maximum") == maximum
        ):
            return manifest["records"]
    split_dir.mkdir(parents=True, exist_ok=True)
    records = normalized_records(load_split_rows(raw_archive, split), maximum)
    wanted = {
        (record["dialogue_id"], record["utterance_id"]): record
        for record in records
    }
    completed, failures = [], []
    processor, encoder, face_detector = components
    with tempfile.TemporaryDirectory(prefix=f"meld-video-{split}-") as directory:
        temporary = Path(directory)
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
                        record = wanted.get(key) if key else None
                        if record is None:
                            continue
                        source = nested.extractfile(media_member)
                        if source is None:
                            failures.append(record | {"error": "could not read media"})
                            continue
                        clip = temporary / f"dia{key[0]}_utt{key[1]}.mp4"
                        with source, clip.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                        try:
                            crops, detections = decode_sampled_faces(
                                clip, frame_count, face_detector
                            )
                            embeddings = encode_faces(crops, processor, encoder, encoder.device)
                            embedding_path = split_dir / f"dia{key[0]}_utt{key[1]}.npy"
                            np.save(embedding_path, embeddings.astype(np.float16))
                            completed.append(
                                record
                                | {
                                    "embedding_path": str(embedding_path.resolve()),
                                    "sampled_frames": len(crops),
                                    "detected_face_frames": detections,
                                }
                            )
                        except Exception as exc:
                            failures.append(record | {"error": f"{type(exc).__name__}: {exc}"})
                        finally:
                            clip.unlink(missing_ok=True)
                        if len(completed) % 250 == 0 and completed:
                            print(f"  cached {len(completed)}/{len(records)} {split} videos", flush=True)
                break
    manifest = {
        "video_model": VIDEO_MODEL,
        "frame_count": frame_count,
        "maximum": maximum,
        "records": completed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (split_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Cached {len(completed)} {split} videos; excluded {len(failures)}")
    return completed


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        embeddings = np.load(record["embedding_path"]).astype(np.float32)
        return embeddings, EMOTION_LABELS.index(record["label"]), index


def collate_video(items):
    sequences, labels, indices = zip(*items)
    padded, padding_mask = pad_video_embeddings(list(sequences))
    return padded, padding_mask, torch.tensor(labels), torch.tensor(indices)


def make_loader(records, batch_size, training, seed):
    sampler = None
    if training:
        counts = Counter(record["label"] for record in records)
        sample_weights = torch.tensor(
            [1.0 / np.sqrt(counts[record["label"]]) for record in records],
            dtype=torch.double,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(records),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return torch.utils.data.DataLoader(
        VideoDataset(records),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_video,
    )


class VideoEmotionClassifier(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.pool = AttentiveStatisticsPooling(EMBEDDING_DIMENSION)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_DIMENSION * 2, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, len(EMOTION_LABELS)),
        )

    def forward(self, embeddings, padding_mask):
        return self.classifier(self.pool(embeddings, padding_mask))


def evaluate(model, loader, device):
    actual_ids, predicted_ids, confidences, indices, all_logits = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for embeddings, padding_mask, labels, batch_indices in loader:
            logits = model(embeddings.to(device), padding_mask.to(device))
            probabilities = torch.softmax(logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            actual_ids.extend(labels.tolist())
            predicted_ids.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            indices.extend(batch_indices.tolist())
            all_logits.append(logits.float().cpu().numpy())
    actual = [EMOTION_LABELS[index] for index in actual_ids]
    predicted = [EMOTION_LABELS[index] for index in predicted_ids]
    return compute_metrics(actual, predicted), predicted, confidences, indices, np.concatenate(all_logits)


def write_predictions(path, records, predicted, confidences, indices):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("dialogue_id", "utterance_id", "utterance", "expected", "predicted", "confidence")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prediction, confidence, index in zip(predicted, confidences, indices):
            record = records[index]
            writer.writerow(
                {
                    "dialogue_id": record["dialogue_id"],
                    "utterance_id": record["utterance_id"],
                    "utterance": record["utterance"],
                    "expected": record["label"],
                    "predicted": prediction,
                    "confidence": f"{confidence:.6f}",
                }
            )


def parse_arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--video-cache-dir", type=Path, default=root / "initial-testing/video-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/training-output-video")
    parser.add_argument("--frames-per-clip", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-video-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if min(args.frames_per_clip, args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError("frame count, epochs, batch size, and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    components = load_visual_encoder(device)
    records = {
        split: prepare_video_split(
            args.raw_archive, split, args.video_cache_dir, args.frames_per_clip,
            args.max_samples_per_split, args.rebuild_video_cache, components,
        )
        for split in ("train", "dev", "test")
    }
    del components
    if device.type == "mps":
        torch.mps.empty_cache()
    loaders = {
        split: make_loader(records[split], args.batch_size, split == "train", args.seed)
        for split in records
    }
    model = VideoEmotionClassifier(args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = torch.nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_video.pt"
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    started = time.perf_counter()
    print(f"Training video head on {device}; samples: {len(records['train'])}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for embeddings, padding_mask, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings.to(device), padding_mask.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics, _, _, _, _ = evaluate(model, loaders["dev"], device)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(row)
        print(f"Epoch {epoch:02d}: loss={row['train_loss']:.4f} dev_macro_f1={row['dev_macro_f1']:.4f}")
        if row["dev_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = row["dev_macro_f1"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    dev_metrics, _, _, dev_indices, dev_logits = evaluate(model, loaders["dev"], device)
    test_metrics, predicted, confidences, test_indices, test_logits = evaluate(model, loaders["test"], device)
    write_predictions(args.output_dir / "test_predictions.csv", records["test"], predicted, confidences, test_indices)
    for split, logits, indices in (
        ("dev", dev_logits, dev_indices), ("test", test_logits, test_indices)
    ):
        np.savez_compressed(
            args.output_dir / f"{split}_logits.npz",
            logits=logits,
            labels=np.array([EMOTION_LABELS.index(records[split][i]["label"]) for i in indices]),
            indices=np.array(indices),
        )
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    detection_total = sum(record["detected_face_frames"] for split in records.values() for record in split)
    sampled_total = sum(record["sampled_frames"] for split in records.values() for record in split)
    report = {
        "configuration": vars(args) | {
            "raw_archive": str(args.raw_archive),
            "video_cache_dir": str(args.video_cache_dir),
            "output_dir": str(args.output_dir),
        },
        "video_model": VIDEO_MODEL,
        "face_detection_rate": detection_total / sampled_total,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "final_dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Face detection rate: {report['face_detection_rate']:.4f}")
    print(f"Best development macro F1: {best_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
