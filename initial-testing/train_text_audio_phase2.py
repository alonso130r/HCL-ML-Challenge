#!/usr/bin/env python3
"""Train evidence-gated text and cached emotion2vec fusion on MELD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from audio_phase1 import (
    attach_metadata,
    causal_speaker_features,
    fit_normalizer,
    load_summary_matrices,
    normalize,
)
from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import build_examples, sqrt_class_weights


def masked_mean_std(sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    count = valid.sum(dim=1).clamp_min(1.0)
    mean = (sequence * valid).sum(dim=1) / count
    variance = ((sequence - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count
    return torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=-1)


def symmetric_contrastive_loss(
    text: torch.Tensor, audio: torch.Tensor, temperature: float
) -> torch.Tensor:
    text = torch.nn.functional.normalize(text, dim=-1)
    audio = torch.nn.functional.normalize(audio, dim=-1)
    similarities = text @ audio.T / temperature
    targets = torch.arange(len(text), device=text.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(similarities, targets)
        + torch.nn.functional.cross_entropy(similarities.T, targets)
    )


def make_derangement(size: int, seed: int) -> list[int]:
    if size < 2:
        raise ValueError("a derangement needs at least two samples")
    values = list(range(size))
    generator = random.Random(seed)
    while True:
        generator.shuffle(values)
        if all(index != value for index, value in enumerate(values)):
            return values.copy()


def checkpoint_score(matched, controls, minimum_margin: float):
    margin = matched["weighted_f1"] - max(item["weighted_f1"] for item in controls)
    return int(margin >= minimum_margin), margin, matched["macro_f1"]


def combine_records(rows, context_window: int, audio_cache_dir: Path, split: str):
    examples = build_examples(rows, context_window)
    metadata = []
    for row in rows:
        path = audio_cache_dir / split / "phase1" / (
            f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.npz"
        )
        if path.exists():
            metadata.append(
                {
                    "dialogue_id": row["Dialogue_ID"],
                    "utterance_id": row["Utterance_ID"],
                    "phase1_path": str(path.resolve()),
                }
            )
    audio = attach_metadata(rows, metadata)
    by_key = {
        (item["dialogue_id"], item["utterance_id"]): item for item in audio
    }
    records = []
    for example in examples:
        key = (example["dialogue_id"], example["utterance_id"])
        if key in by_key:
            records.append(example | by_key[key])
    return records


def attach_speaker_relative_acoustics(records_by_split):
    summaries = {
        split: load_summary_matrices(records) for split, records in records_by_split.items()
    }
    absolute_stats = fit_normalizer(summaries["train"][1])
    combined = {}
    for split, records in records_by_split.items():
        acoustics = summaries[split][1]
        relative, history = causal_speaker_features(
            acoustics, records, *absolute_stats
        )
        combined[split] = np.concatenate(
            (normalize(acoustics, absolute_stats), relative, np.log1p(history[:, None])),
            axis=1,
        ).astype(np.float32)
    final_stats = fit_normalizer(combined["train"])
    for split, records in records_by_split.items():
        values = normalize(combined[split], final_stats)
        for record, acoustic in zip(records, values):
            record["speaker_acoustic"] = acoustic
    return final_stats


class PhaseTwoDataset(torch.utils.data.Dataset):
    def __init__(self, records, audio_indices=None):
        self.records = records
        self.audio_indices = (
            list(range(len(records))) if audio_indices is None else audio_indices
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        text_record = self.records[index]
        audio_record = self.records[self.audio_indices[index]]
        with np.load(audio_record["phase1_path"]) as cache:
            frames = cache["emotion_frames"].astype(np.float32)
        return {
            "text": text_record["text"],
            "frames": frames,
            "acoustic": audio_record["speaker_acoustic"],
            "label": EMOTION_LABELS.index(text_record["label"]),
            "index": index,
        }


def crop_frames(frames: np.ndarray, maximum: int, training: bool, rng) -> np.ndarray:
    if len(frames) <= maximum:
        return frames
    start = rng.randrange(len(frames) - maximum + 1) if training else (len(frames) - maximum) // 2
    return frames[start : start + maximum]


def make_collator(tokenizer, max_length, max_audio_frames, training, seed):
    rng = random.Random(seed)

    def collate(items):
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        texts = [item["text"] for item in items]
        current_starts = torch.tensor(
            [text.rfind("Current:\n") + len("Current:\n") for text in texts]
        ).unsqueeze(1)
        encoded["current_token_mask"] = (
            offsets[:, :, 0].ge(current_starts)
            & offsets[:, :, 1].gt(offsets[:, :, 0])
            & encoded["attention_mask"].bool()
            & ~encoded["special_tokens_mask"].bool()
        )
        frames = [
            crop_frames(item["frames"], max_audio_frames, training, rng) for item in items
        ]
        longest = max(len(value) for value in frames)
        feature_size = frames[0].shape[1]
        padded = np.zeros((len(frames), longest, feature_size), dtype=np.float32)
        padding_mask = np.ones((len(frames), longest), dtype=bool)
        for row, values in enumerate(frames):
            padded[row, : len(values)] = values
            padding_mask[row, : len(values)] = False
        encoded.update(
            {
                "audio_frames": torch.from_numpy(padded),
                "audio_padding_mask": torch.from_numpy(padding_mask),
                "acoustic_features": torch.from_numpy(
                    np.stack([item["acoustic"] for item in items])
                ),
                "labels": torch.tensor([item["label"] for item in items]),
                "indices": torch.tensor([item["index"] for item in items]),
            }
        )
        return encoded

    return collate


def make_loader(records, tokenizer, args, training, audio_indices=None):
    return torch.utils.data.DataLoader(
        PhaseTwoDataset(records, audio_indices),
        batch_size=args.batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collator(
            tokenizer, args.max_length, args.max_audio_frames, training, args.seed
        ),
        num_workers=0,
    )


class PhaseTwoFusion(torch.nn.Module):
    def __init__(self, text_model, audio_input_dimension, acoustic_dimension, args):
        super().__init__()
        self.text_model = text_model
        dimension = args.fusion_dimension
        self.max_gate = args.max_gate
        self.modality_dropout = args.modality_dropout
        self.audio_projection = torch.nn.Linear(audio_input_dimension, dimension)
        self.position = torch.nn.Parameter(torch.zeros(1, args.max_audio_frames, dimension))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=args.attention_heads,
            dim_feedforward=dimension * 4,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=args.temporal_layers, enable_nested_tensor=False
        )
        self.text_query = torch.nn.Linear(text_model.config.hidden_size, dimension)
        self.cross_attention = torch.nn.MultiheadAttention(
            dimension, args.attention_heads, dropout=args.dropout, batch_first=True
        )
        self.acoustic_projection = torch.nn.Sequential(
            torch.nn.Linear(acoustic_dimension, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.audio_fusion = torch.nn.Sequential(
            torch.nn.Linear(dimension * 4, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.audio_classifier = torch.nn.Linear(dimension, len(EMOTION_LABELS))
        self.correction = torch.nn.Linear(dimension, len(EMOTION_LABELS))
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(len(EMOTION_LABELS) + 2, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.zeros_(self.correction.weight)
        torch.nn.init.zeros_(self.correction.bias)
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.constant_(self.gate[-1].bias, args.initial_gate_bias)
        self.text_contrast = torch.nn.Linear(text_model.config.hidden_size, dimension)
        self.audio_contrast = torch.nn.Linear(dimension, dimension)

    def forward(self, batch, audio_mode="matched"):
        text_inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        with torch.no_grad():
            text_output = self.text_model(
                **text_inputs, output_hidden_states=True, return_dict=True
            )
        text_logits = text_output.logits
        token_mask = batch["current_token_mask"].unsqueeze(-1)
        text_hidden = text_output.hidden_states[-1]
        current_text = (text_hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        frames = self.audio_projection(batch["audio_frames"])
        frames = frames + self.position[:, : frames.shape[1]]
        frames = self.temporal_encoder(
            frames, src_key_padding_mask=batch["audio_padding_mask"]
        )
        query = self.text_query(current_text).unsqueeze(1)
        attended, _ = self.cross_attention(
            query,
            frames,
            frames,
            key_padding_mask=batch["audio_padding_mask"],
            need_weights=False,
        )
        statistics = masked_mean_std(frames, batch["audio_padding_mask"])
        acoustic = self.acoustic_projection(batch["acoustic_features"])
        audio_embedding = self.audio_fusion(
            torch.cat((attended.squeeze(1), statistics, acoustic), dim=-1)
        )
        audio_logits = self.audio_classifier(audio_embedding)
        probabilities = torch.softmax(text_logits.detach(), dim=-1)
        confidence = probabilities.max(dim=-1, keepdim=True).values
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(probabilities.shape[-1])
        gate_inputs = torch.cat((audio_logits, confidence, entropy), dim=-1)
        gate = self.max_gate * torch.sigmoid(self.gate(gate_inputs))
        correction = self.correction(audio_embedding)
        if audio_mode == "zeroed":
            gate = torch.zeros_like(gate)
        elif audio_mode != "matched":
            raise ValueError(f"unknown audio mode: {audio_mode}")
        if self.training and self.modality_dropout > 0:
            keep = torch.rand_like(gate).ge(self.modality_dropout).to(gate.dtype)
            gate = gate * keep
        return {
            "logits": text_logits + gate * correction,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "gate": gate,
            "correction": correction,
            "text_embedding": self.text_contrast(current_text),
            "audio_embedding": self.audio_contrast(audio_embedding),
        }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def calculate_loss(output, labels, class_weights, args):
    fused = torch.nn.functional.cross_entropy(
        output["logits"], labels, weight=class_weights
    )
    audio = torch.nn.functional.cross_entropy(
        output["audio_logits"], labels, weight=class_weights
    )
    contrastive = symmetric_contrastive_loss(
        output["text_embedding"], output["audio_embedding"], args.temperature
    )
    correction = output["correction"].square().mean()
    total = (
        fused
        + args.audio_loss_weight * audio
        + args.contrastive_weight * contrastive
        + args.correction_penalty_weight * correction
    )
    return total, {
        "total": float(total.detach().cpu()),
        "fused": float(fused.detach().cpu()),
        "audio": float(audio.detach().cpu()),
        "contrastive": float(contrastive.detach().cpu()),
        "correction": float(correction.detach().cpu()),
    }


def train_epoch(model, loader, optimizer, class_weights, args, device):
    model.train()
    model.text_model.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        output = model(batch)
        loss, components = calculate_loss(output, batch["labels"], class_weights, args)
        (loss / args.gradient_accumulation).backward()
        losses.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {
        key: sum(row[key] for row in losses) / len(losses) for key in losses[0]
    }


def evaluate(model, loader, device, audio_mode="matched"):
    actual, fused, text, confidence, indices, gates = [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch, audio_mode=audio_mode)
            probabilities = torch.softmax(output["logits"], dim=-1)
            batch_confidence, prediction = probabilities.max(dim=-1)
            actual.extend(batch["labels"].cpu().tolist())
            fused.extend(prediction.cpu().tolist())
            text.extend(output["text_logits"].argmax(dim=-1).cpu().tolist())
            confidence.extend(batch_confidence.cpu().tolist())
            indices.extend(batch["indices"].cpu().tolist())
            gates.extend(output["gate"].squeeze(-1).cpu().tolist())
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    return {
        "fused_metrics": compute_metrics(names(actual), names(fused)),
        "text_metrics": compute_metrics(names(actual), names(text)),
        "predicted": names(fused),
        "confidence": confidence,
        "indices": indices,
        "mean_gate": float(np.mean(gates)),
    }


def write_predictions(path, records, result):
    by_index = {
        index: (prediction, confidence)
        for index, prediction, confidence in zip(
            result["indices"], result["predicted"], result["confidence"]
        )
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "dialogue_id", "utterance_id", "speaker", "utterance",
            "expected", "predicted", "confidence",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            prediction, confidence = by_index[index]
            writer.writerow(
                {
                    "dialogue_id": record["dialogue_id"],
                    "utterance_id": record["utterance_id"],
                    "speaker": record["speaker"],
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
    parser.add_argument("--text-model", type=Path, default=root / "initial-testing/training-output-text/best-model")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/training-output-text-audio-phase2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fusion-dimension", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--max-gate", type=float, default=0.5)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--audio-loss-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--correction-penalty-weight", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--minimum-audio-margin", type=float, default=0.005)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-audio-frames", type=int, default=400)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def validate_arguments(args):
    for name in ("epochs", "batch_size", "gradient_accumulation", "patience", "max_audio_frames"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if args.fusion_dimension % args.attention_heads:
        raise ValueError("fusion dimension must be divisible by attention heads")
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("modality dropout must be in [0, 1)")
    if not 0 < args.max_gate <= 1:
        raise ValueError("max gate must be in (0, 1]")
    if not args.text_model.exists():
        raise FileNotFoundError(f"text checkpoint not found: {args.text_model}")


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model)
    for parameter in text_model.parameters():
        parameter.requires_grad = False

    records = {}
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]
        records[split] = combine_records(
            rows, args.context_window, args.audio_cache_dir, split
        )
        if not records[split]:
            raise FileNotFoundError(
                f"no Phase 1 cache found for {split}; run audio_phase1.py first"
            )
    acoustic_stats = attach_speaker_relative_acoustics(records)
    with np.load(records["train"][0]["phase1_path"]) as cache:
        audio_dimension = cache["emotion_frames"].shape[1]
    acoustic_dimension = len(records["train"][0]["speaker_acoustic"])
    model = PhaseTwoFusion(
        text_model, audio_dimension, acoustic_dimension, args
    ).to(device)
    loaders = {
        split: make_loader(records[split], tokenizer, args, split == "train")
        for split in records
    }
    dev_controls = [
        make_loader(
            records["dev"], tokenizer, args, False,
            make_derangement(len(records["dev"]), seed),
        )
        for seed in args.shuffle_seeds
    ]
    train_labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in records["train"]]
    )
    class_weights = torch.from_numpy(
        sqrt_class_weights(train_labels, len(EMOTION_LABELS))
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_text_audio_phase2.pt"
    baseline = evaluate(model, loaders["dev"], device, "zeroed")["text_metrics"]
    print(
        f"Device: {device}; samples: {len(records['train'])}; "
        f"text dev weighted F1: {baseline['weighted_f1']:.4f}"
    )
    history, best_score, best_epoch, stale = [], (-1, -float("inf"), -1.0), 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, loaders["train"], optimizer, class_weights, args, device)
        matched = evaluate(model, loaders["dev"], device)
        zeroed = evaluate(model, loaders["dev"], device, "zeroed")
        shuffled = [evaluate(model, loader, device) for loader in dev_controls]
        controls = [zeroed["fused_metrics"]] + [item["fused_metrics"] for item in shuffled]
        score = checkpoint_score(
            matched["fused_metrics"], controls, args.minimum_audio_margin
        )
        selection_score = (score[0], score[2], score[1])
        row = {
            "epoch": epoch,
            "losses": losses,
            "dev_matched": matched["fused_metrics"],
            "dev_zeroed": zeroed["fused_metrics"],
            "dev_shuffled": [item["fused_metrics"] for item in shuffled],
            "matched_weighted_margin": score[1],
            "mean_gate": matched["mean_gate"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={losses['total']:.4f} "
            f"matched_macro={score[2]:.4f} "
            f"matched_weighted={matched['fused_metrics']['weighted_f1']:.4f} "
            f"audio_margin={score[1]:+.4f}"
        )
        if selection_score > best_score:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_matched = evaluate(model, loaders["test"], device)
    test_zeroed = evaluate(model, loaders["test"], device, "zeroed")
    test_shuffled = []
    for seed in args.shuffle_seeds:
        loader = make_loader(
            records["test"], tokenizer, args, False,
            make_derangement(len(records["test"]), seed),
        )
        test_shuffled.append(evaluate(model, loader, device))
    test_controls = [test_zeroed["fused_metrics"]] + [
        item["fused_metrics"] for item in test_shuffled
    ]
    test_score = checkpoint_score(
        test_matched["fused_metrics"], test_controls, args.minimum_audio_margin
    )
    write_predictions(
        args.output_dir / "test_predictions.csv", records["test"], test_matched
    )
    np.savez_compressed(
        args.output_dir / "acoustic_normalization.npz",
        mean=acoustic_stats[0], std=acoustic_stats[1],
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {split: len(value) for split, value in records.items()},
        "best_epoch": best_epoch,
        "development_text_baseline": baseline,
        "test_text": test_matched["text_metrics"],
        "test_matched": test_matched["fused_metrics"],
        "test_zeroed": test_zeroed["fused_metrics"],
        "test_shuffled": [item["fused_metrics"] for item in test_shuffled],
        "test_worst_case_weighted_margin": test_score[1],
        "passes_audio_evidence_gate": bool(test_score[0]),
        "mean_test_gate": test_matched["mean_gate"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test_matched['text_metrics']['weighted_f1']:.4f}")
    print(f"Matched test weighted F1: {test_matched['fused_metrics']['weighted_f1']:.4f}")
    print(f"Matched test macro F1: {test_matched['fused_metrics']['macro_f1']:.4f}")
    print(f"Worst-case audio margin: {test_score[1]:+.4f}")
    print(f"Passes audio evidence gate: {bool(test_score[0])}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
