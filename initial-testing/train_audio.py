#!/usr/bin/env python3
"""Fine-tune WavLM Base Plus for seven-class MELD emotion recognition."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from cache_embeddings import decode_clip, load_split_rows, media_key
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import sqrt_class_weights


AUDIO_MODEL = "microsoft/wavlm-base-plus"


def pad_audio(waveforms: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    if not waveforms or any(len(waveform) == 0 for waveform in waveforms):
        raise ValueError("audio batch cannot contain empty waveforms")
    maximum = max(len(waveform) for waveform in waveforms)
    padded = torch.zeros((len(waveforms), maximum), dtype=torch.float32)
    lengths = torch.empty(len(waveforms), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        padded[index, : len(waveform)] = torch.from_numpy(waveform.astype(np.float32, copy=False))
        lengths[index] = len(waveform) / maximum
    return padded, lengths


def unfreeze_top_layers(encoder: torch.nn.Module, count: int) -> None:
    layers = encoder.encoder.layers
    if count < 0 or count > len(layers):
        raise ValueError(f"cannot unfreeze {count} layers; encoder has only {len(layers)}")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if count:
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True


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


def prepare_audio_split(raw_archive, split, cache_dir, maximum=None, rebuild=False):
    split_dir = cache_dir / split
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    split_dir.mkdir(parents=True, exist_ok=True)
    records = normalized_records(load_split_rows(raw_archive, split), maximum)
    wanted = {
        (record["dialogue_id"], record["utterance_id"]): record
        for record in records
    }
    completed = []
    failures = []
    with tempfile.TemporaryDirectory(prefix=f"meld-audio-{split}-") as directory:
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
                        waveform, error = decode_clip(clip)
                        clip.unlink(missing_ok=True)
                        if error:
                            failures.append(record | {"error": error})
                            continue
                        audio_path = split_dir / f"dia{key[0]}_utt{key[1]}.npy"
                        np.save(audio_path, waveform.astype(np.float32, copy=False))
                        completed.append(record | {"audio_path": str(audio_path.resolve())})
                        if len(completed) % 250 == 0:
                            print(f"  cached {len(completed)}/{len(records)} {split} clips", flush=True)
                break
    manifest_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    (split_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Cached {len(completed)} {split} clips; excluded {len(failures)}")
    return completed


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return np.load(record["audio_path"]), EMOTION_LABELS.index(record["label"]), index


def collate_audio(items):
    waveforms, labels, indices = zip(*items)
    padded, lengths = pad_audio(list(waveforms))
    return padded, lengths, torch.tensor(labels), torch.tensor(indices)


def make_loader(records, batch_size, shuffle, seed):
    return torch.utils.data.DataLoader(
        AudioDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_audio,
    )


def frame_padding_mask(lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
    valid = torch.ceil(lengths * frame_count).long().clamp(min=1, max=frame_count)
    positions = torch.arange(frame_count, device=lengths.device).unsqueeze(0)
    return positions >= valid.unsqueeze(1)


class AttentiveStatisticsPooling(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.attention = torch.nn.Linear(dimension, 1)

    def forward(self, hidden, padding_mask):
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (hidden * weights).sum(dim=1)
        variance = (((hidden - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1)
        return torch.cat((mean, variance.clamp_min(1e-8).sqrt()), dim=-1)


class WavLmEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.freeze = True

    def forward(self, waveforms, lengths):
        sample_count = waveforms.shape[1]
        padding_mask = frame_padding_mask(lengths, sample_count)
        attention_mask = (~padding_mask).long()

        def encode():
            return self.model(
                input_values=waveforms,
                attention_mask=attention_mask,
            ).last_hidden_state

        if self.freeze:
            with torch.no_grad():
                return encode()
        return encode()


class MeldAudioClassifier(torch.nn.Module):
    def __init__(self, audio_encoder, dropout=0.2):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.pool = AttentiveStatisticsPooling(768)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1536, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, len(EMOTION_LABELS)),
        )

    def forward(self, waveforms, lengths):
        hidden = self.audio_encoder(waveforms, lengths)
        padding_mask = frame_padding_mask(lengths, hidden.shape[1])
        pooled = self.pool(hidden, padding_mask)
        return self.classifier(pooled)

    def head_parameters(self):
        encoder_parameters = {id(parameter) for parameter in self.audio_encoder.parameters()}
        return [
            parameter for parameter in self.parameters()
            if id(parameter) not in encoder_parameters
        ]


def load_model(device, dropout):
    from transformers import AutoModel

    audio_encoder = WavLmEncoder(AutoModel.from_pretrained(AUDIO_MODEL))
    model = MeldAudioClassifier(audio_encoder, dropout).to(device)
    unfreeze_top_layers(model.audio_encoder.model, 0)
    return model


def evaluate(model, loader, device):
    actual_ids, predicted_ids, confidences, indices, all_logits = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for waveforms, lengths, labels, batch_indices in loader:
            logits = model(waveforms.to(device), lengths.to(device))
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


def train_epoch(
    model, loader, optimizer, loss_function, accumulation, device,
    train_encoder,
):
    model.train()
    if not train_encoder:
        model.audio_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, (waveforms, lengths, labels, _) in enumerate(loader, start=1):
        logits = model(waveforms.to(device), lengths.to(device))
        loss = loss_function(logits, labels.to(device))
        (loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if step % accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses)


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
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "initial-testing/training-output-audio-wavlm",
    )
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-head-macro-f1", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-audio-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.head_epochs < 0 or args.finetune_epochs < 0 or args.head_epochs + args.finetune_epochs < 1:
        raise ValueError("at least one training epoch is required")
    if min(args.batch_size, args.gradient_accumulation, args.patience) < 1:
        raise ValueError("batch size, gradient accumulation, and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = {
        split: prepare_audio_split(
            args.raw_archive, split, args.audio_cache_dir,
            args.max_samples_per_split, args.rebuild_audio_cache,
        )
        for split in ("train", "dev", "test")
    }
    label_ids = np.array([EMOTION_LABELS.index(record["label"]) for record in records["train"]])
    weights = torch.from_numpy(sqrt_class_weights(label_ids, len(EMOTION_LABELS)))
    loaders = {
        split: make_loader(records[split], args.batch_size, split == "train", args.seed)
        for split in records
    }
    device = select_device(torch)
    model = load_model(device, args.dropout)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_audio.pt"
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    started = time.perf_counter()
    phases = (("head", args.head_epochs), ("finetune", args.finetune_epochs))
    epoch = 0
    stop = False
    for phase, phase_epochs in phases:
        if phase_epochs == 0 or stop:
            continue
        if phase == "finetune" and best_f1 < args.minimum_head_macro_f1:
            print(
                f"Skipping finetune phase: best frozen-head macro F1 "
                f"{best_f1:.4f} is below {args.minimum_head_macro_f1:.4f}"
            )
            break
        stale = 0
        if phase == "head":
            optimizer = torch.optim.AdamW(
                model.head_parameters(), lr=args.head_learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            if checkpoint_path.exists():
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=device, weights_only=True)
                )
            unfreeze_top_layers(model.audio_encoder.model, args.unfreeze_layers)
            model.audio_encoder.freeze = False
            encoder_parameters = [
                p for p in model.audio_encoder.parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.head_parameters(), "lr": args.head_learning_rate},
                    {"params": encoder_parameters, "lr": args.encoder_learning_rate},
                ],
                weight_decay=args.weight_decay,
            )
        print(f"Starting {phase} phase for {phase_epochs} epochs on {device}")
        for _ in range(phase_epochs):
            epoch += 1
            loss = train_epoch(
                model, loaders["train"], optimizer, loss_function,
                args.gradient_accumulation, device, phase == "finetune",
            )
            dev_metrics, _, _, _, _ = evaluate(model, loaders["dev"], device)
            row = {"epoch": epoch, "phase": phase, "train_loss": loss, "dev_macro_f1": dev_metrics["macro_f1"]}
            history.append(row)
            print(f"Epoch {epoch:02d} ({phase}): loss={loss:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f}")
            if dev_metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = dev_metrics["macro_f1"], epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping after epoch {epoch}")
                    stop = True
                    break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    dev_metrics, _, _, dev_indices, dev_logits = evaluate(model, loaders["dev"], device)
    test_metrics, predicted, confidences, test_indices, test_logits = evaluate(model, loaders["test"], device)
    write_predictions(args.output_dir / "test_predictions.csv", records["test"], predicted, confidences, test_indices)
    np.savez_compressed(
        args.output_dir / "dev_logits.npz", logits=dev_logits,
        labels=np.array([EMOTION_LABELS.index(records["dev"][i]["label"]) for i in dev_indices]),
        indices=np.array(dev_indices),
    )
    np.savez_compressed(
        args.output_dir / "test_logits.npz", logits=test_logits,
        labels=np.array([EMOTION_LABELS.index(records["test"][i]["label"]) for i in test_indices]),
        indices=np.array(test_indices),
    )
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = {
        "configuration": vars(args) | {
            "raw_archive": str(args.raw_archive),
            "audio_cache_dir": str(args.audio_cache_dir),
            "output_dir": str(args.output_dir),
        },
        "audio_model": AUDIO_MODEL,
        "audio_head": {
            "input_dimension": 768,
            "hidden_dimension": 256,
            "pooling": "masked_attentive_mean_and_standard_deviation",
        },
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "final_dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best development macro F1: {best_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
