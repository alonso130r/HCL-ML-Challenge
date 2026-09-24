#!/usr/bin/env python3
"""Train coarse temporally aligned text-audio fusion on cached MELD features."""

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

from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import sqrt_class_weights
from train_text_audio_phase2 import (
    attach_speaker_relative_acoustics,
    combine_records,
    hard_negative_indices,
    make_derangement,
)


def pool_sequence_bins(sequence: np.ndarray, number_of_bins: int, offset=0.0):
    if sequence.ndim != 2 or not len(sequence):
        raise ValueError("expected a nonempty frame-by-feature sequence")
    positions = np.arange(len(sequence), dtype=np.float32) / len(sequence)
    assignments = np.floor((positions + offset) * number_of_bins).astype(np.int64)
    assignments = np.clip(assignments, 0, number_of_bins - 1)
    pooled = np.zeros((number_of_bins, sequence.shape[1] * 3), dtype=np.float32)
    valid = np.zeros(number_of_bins, dtype=bool)
    for index in range(number_of_bins):
        values = sequence[assignments == index]
        if len(values):
            valid[index] = True
            pooled[index] = np.concatenate(
                (values.mean(axis=0), values.std(axis=0), values.max(axis=0))
            )
    return pooled, valid


def pool_text_bins(hidden, current_mask, number_of_bins):
    batch, _, dimension = hidden.shape
    pooled = hidden.new_zeros((batch, number_of_bins, dimension))
    valid = torch.zeros(
        (batch, number_of_bins), dtype=torch.bool, device=hidden.device
    )
    for row in range(batch):
        token_indices = torch.nonzero(current_mask[row], as_tuple=False).flatten()
        if not len(token_indices):
            continue
        assignments = torch.div(
            torch.arange(len(token_indices), device=hidden.device) * number_of_bins,
            len(token_indices),
            rounding_mode="floor",
        ).clamp_max(number_of_bins - 1)
        for index in range(number_of_bins):
            selected = token_indices[assignments == index]
            if len(selected):
                pooled[row, index] = hidden[row, selected].mean(dim=0)
                valid[row, index] = True
    return pooled, valid


def masked_mean(sequence, valid):
    weights = valid.unsqueeze(-1).to(sequence.dtype)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def shift_audio_state(state, amount):
    return {
        "bins": torch.roll(state["bins"], shifts=-amount, dims=1),
        "valid": torch.roll(state["valid"], shifts=-amount, dims=1),
        "global": state["global"],
    }


def temporal_checkpoint_score(
    matched, shifted, shuffled, zeroed, minimum_margin
):
    controls = [shifted, zeroed] + list(shuffled)
    margin = matched["weighted_f1"] - max(
        item["weighted_f1"] for item in controls
    )
    return int(margin >= minimum_margin), margin, matched["macro_f1"]


class BinnedDataset(torch.utils.data.Dataset):
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
            emotion = cache["emotion_frames"].astype(np.float32)
            prosody = cache["prosody_frames"].astype(np.float32)
        return {
            "text": text_record["text"],
            "emotion": emotion,
            "prosody": prosody,
            "acoustic": audio_record["speaker_acoustic"],
            "audio_frame_count": len(emotion),
            "label": EMOTION_LABELS.index(text_record["label"]),
            "index": index,
        }


def make_collator(tokenizer, args, training):
    generator = random.Random(args.seed)

    def collate(items):
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        starts = torch.tensor(
            [
                item["text"].rfind("Current:\n") + len("Current:\n")
                for item in items
            ]
        ).unsqueeze(1)
        encoded["current_token_mask"] = (
            offsets[:, :, 0].ge(starts)
            & offsets[:, :, 1].gt(offsets[:, :, 0])
            & encoded["attention_mask"].bool()
            & ~encoded["special_tokens_mask"].bool()
        )
        audio_bins, bin_masks = [], []
        for item in items:
            jitter = (
                generator.uniform(-args.boundary_jitter, args.boundary_jitter)
                / args.number_of_bins
                if training
                else 0.0
            )
            emotion, emotion_valid = pool_sequence_bins(
                item["emotion"], args.number_of_bins, jitter
            )
            prosody, prosody_valid = pool_sequence_bins(
                item["prosody"], args.number_of_bins, jitter
            )
            audio_bins.append(np.concatenate((emotion, prosody), axis=1))
            bin_masks.append(emotion_valid & prosody_valid)
        encoded.update(
            {
                "audio_bins": torch.from_numpy(np.stack(audio_bins)),
                "audio_bin_mask": torch.from_numpy(np.stack(bin_masks)),
                "acoustic_features": torch.from_numpy(
                    np.stack([item["acoustic"] for item in items])
                ),
                "audio_frame_counts": torch.tensor(
                    [item["audio_frame_count"] for item in items]
                ),
                "labels": torch.tensor([item["label"] for item in items]),
                "indices": torch.tensor([item["index"] for item in items]),
            }
        )
        return encoded

    return collate


def make_loader(records, tokenizer, args, training, audio_indices=None):
    return torch.utils.data.DataLoader(
        BinnedDataset(records, audio_indices),
        batch_size=args.batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collator(tokenizer, args, training),
        num_workers=0,
    )


def permute_audio(batch, indices):
    result = dict(batch)
    for key in (
        "audio_bins", "audio_bin_mask", "acoustic_features", "audio_frame_counts"
    ):
        result[key] = batch[key].index_select(0, indices)
    return result


class BinnedFusionModel(torch.nn.Module):
    def __init__(self, text_model, audio_dimension, acoustic_dimension, args):
        super().__init__()
        self.text_model = text_model
        self.number_of_bins = args.number_of_bins
        self.local_max_gate = args.local_max_gate
        self.global_max_gate = args.global_max_gate
        dimension = args.fusion_dimension
        self.text_projection = torch.nn.Linear(text_model.config.hidden_size, dimension)
        self.audio_projection = torch.nn.Sequential(
            torch.nn.Linear(audio_dimension, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.acoustic_projection = torch.nn.Sequential(
            torch.nn.Linear(acoustic_dimension, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
        )
        self.local_interaction = torch.nn.Sequential(
            torch.nn.Linear(dimension * 6, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.local_attention = torch.nn.Linear(dimension, 1)
        self.global_fusion = torch.nn.Sequential(
            torch.nn.Linear(dimension * 2, dimension),
            torch.nn.LayerNorm(dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.local_correction = torch.nn.Linear(dimension, len(EMOTION_LABELS))
        self.global_correction = torch.nn.Linear(dimension, len(EMOTION_LABELS))
        self.audio_classifier = torch.nn.Linear(dimension * 2, len(EMOTION_LABELS))
        gate_dimension = len(EMOTION_LABELS) + 3
        self.local_gate = torch.nn.Sequential(
            torch.nn.Linear(gate_dimension, 32), torch.nn.GELU(), torch.nn.Linear(32, 1)
        )
        self.global_gate = torch.nn.Sequential(
            torch.nn.Linear(gate_dimension, 32), torch.nn.GELU(), torch.nn.Linear(32, 1)
        )
        self.text_contrast = torch.nn.Linear(text_model.config.hidden_size, dimension)
        self.audio_contrast = torch.nn.Linear(dimension, dimension)
        for head in (self.local_correction, self.global_correction):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        for gate in (self.local_gate, self.global_gate):
            torch.nn.init.zeros_(gate[-1].weight)
            torch.nn.init.constant_(gate[-1].bias, args.initial_gate_bias)

    def encode_text(self, batch):
        inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        with torch.no_grad():
            output = self.text_model(
                **inputs, output_hidden_states=True, return_dict=True
            )
        hidden = output.hidden_states[-1]
        bins, valid = pool_text_bins(
            hidden, batch["current_token_mask"], self.number_of_bins
        )
        current = masked_mean(bins, valid)
        return {
            "logits": output.logits,
            "bins": self.text_projection(bins),
            "valid": valid,
            "current": current,
        }

    def encode_audio(self, batch):
        bins = self.audio_projection(batch["audio_bins"])
        valid = batch["audio_bin_mask"]
        utterance = masked_mean(bins, valid)
        acoustic = self.acoustic_projection(batch["acoustic_features"])
        return {
            "bins": bins,
            "valid": valid,
            "global": self.global_fusion(torch.cat((utterance, acoustic), dim=-1)),
        }

    def fuse(self, text, audio, audio_mode="matched"):
        bins = audio["bins"]
        previous = torch.cat((bins[:, :1], bins[:, :-1]), dim=1)
        following = torch.cat((bins[:, 1:], bins[:, -1:]), dim=1)
        current_audio = bins
        text_bins = text["bins"]
        interaction = self.local_interaction(
            torch.cat(
                (
                    text_bins,
                    previous,
                    current_audio,
                    following,
                    text_bins * current_audio,
                    (text_bins - current_audio).abs(),
                ),
                dim=-1,
            )
        )
        valid = text["valid"] & audio["valid"]
        scores = self.local_attention(interaction).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local = (interaction * weights.unsqueeze(-1)).sum(dim=1)
        audio_logits = self.audio_classifier(torch.cat((local, audio["global"]), dim=-1))
        probabilities = torch.softmax(text["logits"].detach(), dim=-1)
        confidence = probabilities.max(dim=-1, keepdim=True).values
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / math.log(probabilities.shape[-1])
        coverage = valid.float().mean(dim=-1, keepdim=True)
        gate_input = torch.cat((audio_logits, confidence, entropy, coverage), dim=-1)
        local_gate = self.local_max_gate * torch.sigmoid(self.local_gate(gate_input))
        global_gate = self.global_max_gate * torch.sigmoid(self.global_gate(gate_input))
        if audio_mode == "zeroed":
            local_gate = torch.zeros_like(local_gate)
            global_gate = torch.zeros_like(global_gate)
        elif audio_mode != "matched":
            raise ValueError(f"unknown audio mode: {audio_mode}")
        local_correction = self.local_correction(local)
        global_correction = self.global_correction(audio["global"])
        logits = (
            text["logits"]
            + local_gate * local_correction
            + global_gate * global_correction
        )
        return {
            "logits": logits,
            "text_logits": text["logits"],
            "audio_logits": audio_logits,
            "local_gate": local_gate,
            "global_gate": global_gate,
            "local_correction": local_correction,
            "global_correction": global_correction,
            "text_embedding": self.text_contrast(text["current"]),
            "audio_embedding": self.audio_contrast(audio["global"]),
        }

    def forward(self, batch, audio_mode="matched"):
        return self.fuse(
            self.encode_text(batch), self.encode_audio(batch), audio_mode
        )


def correct_class_support(output, labels):
    return torch.log_softmax(output["logits"], dim=-1).gather(
        1, labels.unsqueeze(1)
    )


def ranking_loss(matched, control, labels, margin):
    advantage = correct_class_support(matched, labels) - correct_class_support(
        control, labels
    )
    return torch.relu(margin - advantage).mean()


def symmetric_contrastive_loss(text, audio, temperature):
    text = torch.nn.functional.normalize(text, dim=-1)
    audio = torch.nn.functional.normalize(audio, dim=-1)
    similarities = text @ audio.T / temperature
    targets = torch.arange(len(text), device=text.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(similarities, targets)
        + torch.nn.functional.cross_entropy(similarities.T, targets)
    )


def calculate_loss(matched, shifted, shuffled, labels, class_weights, args):
    fused = torch.nn.functional.cross_entropy(
        matched["logits"], labels, weight=class_weights
    )
    audio = torch.nn.functional.cross_entropy(
        matched["audio_logits"], labels, weight=class_weights
    )
    contrastive = symmetric_contrastive_loss(
        matched["text_embedding"], matched["audio_embedding"], args.temperature
    )
    shift_ranking = ranking_loss(
        matched, shifted, labels, args.temporal_margin
    )
    shuffle_ranking = ranking_loss(
        matched, shuffled, labels, args.counterfactual_margin
    )
    shifted_local_residual = (
        shifted["local_gate"] * shifted["local_correction"]
    ).square().mean()
    shuffled_residual = (
        shuffled["local_gate"] * shuffled["local_correction"]
    ).square().mean() + (
        shuffled["global_gate"] * shuffled["global_correction"]
    ).square().mean()
    gate_ranking = torch.relu(
        args.gate_margin - matched["local_gate"] + shifted["local_gate"]
    ).mean() + torch.relu(
        args.gate_margin
        - (matched["local_gate"] + matched["global_gate"])
        + (shuffled["local_gate"] + shuffled["global_gate"])
    ).mean()
    correction = (
        matched["local_correction"].square().mean()
        + matched["global_correction"].square().mean()
    )
    total = (
        fused
        + args.audio_loss_weight * audio
        + args.contrastive_weight * contrastive
        + args.counterfactual_weight * shuffle_ranking
        + args.temporal_weight * shift_ranking
        + args.negative_residual_weight
        * (shifted_local_residual + shuffled_residual)
        + args.gate_ranking_weight * gate_ranking
        + args.negative_gate_weight
        * (
            shifted["local_gate"].mean()
            + shuffled["local_gate"].mean()
            + shuffled["global_gate"].mean()
        )
        + args.correction_penalty_weight * correction
    )
    components = {
        "total": total,
        "fused": fused,
        "audio": audio,
        "contrastive": contrastive,
        "shift_ranking": shift_ranking,
        "shuffle_ranking": shuffle_ranking,
        "negative_residual": shifted_local_residual + shuffled_residual,
        "gate_ranking": gate_ranking,
        "correction": correction,
        "matched_local_gate": matched["local_gate"].mean(),
        "matched_global_gate": matched["global_gate"].mean(),
        "shifted_local_gate": shifted["local_gate"].mean(),
        "shuffled_local_gate": shuffled["local_gate"].mean(),
        "shuffled_global_gate": shuffled["global_gate"].mean(),
    }
    return total, {
        key: float(value.detach().cpu()) for key, value in components.items()
    }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, class_weights, args, device):
    model.train()
    model.text_model.eval()
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        text = model.encode_text(batch)
        audio = model.encode_audio(batch)
        matched = model.fuse(text, audio)
        shifted = model.fuse(text, shift_audio_state(audio, args.shift_bins))
        if len(batch["labels"]) > 1:
            negative_indices = hard_negative_indices(
                batch["labels"], batch["audio_frame_counts"]
            )
            shuffled_audio = model.encode_audio(permute_audio(batch, negative_indices))
            shuffled = model.fuse(text, shuffled_audio)
        else:
            shuffled = shifted
        loss, components = calculate_loss(
            matched, shifted, shuffled, batch["labels"], class_weights, args
        )
        (loss / args.gradient_accumulation).backward()
        rows.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def evaluate_modes(model, loader, device, modes):
    accumulators = {
        mode: {"actual": [], "predicted": [], "text": [], "confidence": [],
               "indices": [], "local_gate": [], "global_gate": []}
        for mode in modes
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            text = model.encode_text(batch)
            audio = model.encode_audio(batch)
            for mode in modes:
                selected = shift_audio_state(audio, modes[mode]) if modes[mode] else audio
                output = model.fuse(
                    text, selected, "zeroed" if mode == "zeroed" else "matched"
                )
                probabilities = torch.softmax(output["logits"], dim=-1)
                confidence, prediction = probabilities.max(dim=-1)
                target = accumulators[mode]
                target["actual"].extend(batch["labels"].cpu().tolist())
                target["predicted"].extend(prediction.cpu().tolist())
                target["text"].extend(output["text_logits"].argmax(dim=-1).cpu().tolist())
                target["confidence"].extend(confidence.cpu().tolist())
                target["indices"].extend(batch["indices"].cpu().tolist())
                target["local_gate"].extend(output["local_gate"].squeeze(-1).cpu().tolist())
                target["global_gate"].extend(output["global_gate"].squeeze(-1).cpu().tolist())
    results = {}
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    for mode, values in accumulators.items():
        results[mode] = {
            "fused_metrics": compute_metrics(
                names(values["actual"]), names(values["predicted"])
            ),
            "text_metrics": compute_metrics(names(values["actual"]), names(values["text"])),
            "predicted": names(values["predicted"]),
            "confidence": values["confidence"],
            "indices": values["indices"],
            "mean_local_gate": float(np.mean(values["local_gate"])),
            "mean_global_gate": float(np.mean(values["global_gate"])),
        }
    return results


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
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--text-model", type=Path, default=root / "models/text-emotion")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "research/experiments/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "research/experiments/training-output-text-audio-binned")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fusion-dimension", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--number-of-bins", type=int, default=8)
    parser.add_argument("--shift-bins", type=int, default=3)
    parser.add_argument("--boundary-jitter", type=float, default=0.5)
    parser.add_argument("--local-max-gate", type=float, default=0.15)
    parser.add_argument("--global-max-gate", type=float, default=0.15)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--audio-loss-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--counterfactual-weight", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--temporal-margin", type=float, default=0.05)
    parser.add_argument("--gate-ranking-weight", type=float, default=0.1)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--negative-residual-weight", type=float, default=0.5)
    parser.add_argument("--negative-gate-weight", type=float, default=0.2)
    parser.add_argument("--correction-penalty-weight", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--minimum-audio-margin", type=float, default=0.005)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def validate_arguments(args):
    for name in ("epochs", "batch_size", "gradient_accumulation", "number_of_bins", "patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if not 0 < args.shift_bins < args.number_of_bins:
        raise ValueError("shift bins must be between zero and the number of bins")
    for name in ("local_max_gate", "global_max_gate"):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', ' ')} must be in (0, 1]")
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
    sample = BinnedDataset(records["train"])[0]
    emotion, _ = pool_sequence_bins(sample["emotion"], args.number_of_bins)
    prosody, _ = pool_sequence_bins(sample["prosody"], args.number_of_bins)
    model = BinnedFusionModel(
        text_model,
        emotion.shape[1] + prosody.shape[1],
        len(sample["acoustic"]),
        args,
    ).to(device)
    loaders = {
        split: make_loader(records[split], tokenizer, args, split == "train")
        for split in records
    }
    dev_shuffled_loaders = [
        make_loader(
            records["dev"], tokenizer, args, False,
            make_derangement(len(records["dev"]), seed),
        )
        for seed in args.shuffle_seeds
    ]
    labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in records["train"]]
    )
    class_weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_text_audio_binned.pt"
    baseline = evaluate_modes(
        model, loaders["dev"], device, {"zeroed": 0}
    )["zeroed"]["text_metrics"]
    print(
        f"Device: {device}; samples: {len(records['train'])}; bins: "
        f"{args.number_of_bins}; text dev weighted F1: {baseline['weighted_f1']:.4f}"
    )
    history, best_score, best_epoch, stale = [], (-1, -float("inf"), -1.0), 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, loaders["train"], optimizer, class_weights, args, device)
        dev = evaluate_modes(
            model,
            loaders["dev"],
            device,
            {"matched": 0, "shifted": args.shift_bins, "zeroed": 0},
        )
        shuffled = [
            evaluate_modes(model, loader, device, {"shuffled": 0})["shuffled"]
            for loader in dev_shuffled_loaders
        ]
        score = temporal_checkpoint_score(
            dev["matched"]["fused_metrics"],
            dev["shifted"]["fused_metrics"],
            [item["fused_metrics"] for item in shuffled],
            dev["zeroed"]["fused_metrics"],
            args.minimum_audio_margin,
        )
        row = {
            "epoch": epoch,
            "losses": losses,
            "dev_matched": dev["matched"]["fused_metrics"],
            "dev_shifted": dev["shifted"]["fused_metrics"],
            "dev_zeroed": dev["zeroed"]["fused_metrics"],
            "dev_shuffled": [item["fused_metrics"] for item in shuffled],
            "temporal_margin": score[1],
            "matched_local_gate": dev["matched"]["mean_local_gate"],
            "matched_global_gate": dev["matched"]["mean_global_gate"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={losses['total']:.4f} "
            f"matched_macro={score[2]:.4f} "
            f"matched_weighted={dev['matched']['fused_metrics']['weighted_f1']:.4f} "
            f"temporal_margin={score[1]:+.4f}"
        )
        selection = (score[0], score[1], score[2])
        if selection > best_score:
            best_score, best_epoch, stale = selection, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = evaluate_modes(
        model,
        loaders["test"],
        device,
        {"matched": 0, "shifted": args.shift_bins, "zeroed": 0},
    )
    test_shuffled = []
    for seed in args.shuffle_seeds:
        loader = make_loader(
            records["test"], tokenizer, args, False,
            make_derangement(len(records["test"]), seed),
        )
        test_shuffled.append(
            evaluate_modes(model, loader, device, {"shuffled": 0})["shuffled"]
        )
    test_score = temporal_checkpoint_score(
        test["matched"]["fused_metrics"],
        test["shifted"]["fused_metrics"],
        [item["fused_metrics"] for item in test_shuffled],
        test["zeroed"]["fused_metrics"],
        args.minimum_audio_margin,
    )
    write_predictions(
        args.output_dir / "test_predictions.csv", records["test"], test["matched"]
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
        "test_text": test["matched"]["text_metrics"],
        "test_matched": test["matched"]["fused_metrics"],
        "test_shifted": test["shifted"]["fused_metrics"],
        "test_zeroed": test["zeroed"]["fused_metrics"],
        "test_shuffled": [item["fused_metrics"] for item in test_shuffled],
        "test_temporal_margin": test_score[1],
        "passes_temporal_evidence_gate": bool(test_score[0]),
        "matched_local_gate": test["matched"]["mean_local_gate"],
        "matched_global_gate": test["matched"]["mean_global_gate"],
        "shifted_local_gate": test["shifted"]["mean_local_gate"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test['matched']['text_metrics']['weighted_f1']:.4f}")
    print(f"Matched test weighted F1: {test['matched']['fused_metrics']['weighted_f1']:.4f}")
    print(f"Matched test macro F1: {test['matched']['fused_metrics']['macro_f1']:.4f}")
    print(f"Shifted test weighted F1: {test['shifted']['fused_metrics']['weighted_f1']:.4f}")
    print(f"Worst-case temporal margin: {test_score[1]:+.4f}")
    print(f"Passes temporal evidence gate: {bool(test_score[0])}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
