#!/usr/bin/env python3
"""Train causal dialogue and speaker recurrence over cached MELD encodings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audio_phase1 import fit_normalizer, load_summary_matrices, normalize
from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_text import sqrt_class_weights
from train_text_audio_phase2 import (
    attach_speaker_relative_acoustics,
    combine_records,
)


def group_dialogues(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dialogue_id"]].append(dict(record))
    result = []
    for dialogue_id in sorted(grouped, key=lambda value: int(value)):
        dialogue = sorted(
            grouped[dialogue_id], key=lambda item: int(item["utterance_id"])
        )
        speakers = {}
        for item in dialogue:
            if item["speaker"] not in speakers:
                speakers[item["speaker"]] = len(speakers)
            item["speaker_index"] = speakers[item["speaker"]]
        result.append(dialogue)
    return result


def different_label_audio_mapping(records, seed):
    by_label = defaultdict(list)
    for index, record in enumerate(records):
        by_label[record["label"]].append(index)
    labels = list(by_label)
    if len(labels) < 2:
        raise ValueError("audio shuffling requires at least two emotion labels")
    generator = random.Random(seed)
    mapping = []
    for record in records:
        candidates = [label for label in labels if label != record["label"]]
        label = generator.choice(candidates)
        mapping.append(generator.choice(by_label[label]))
    return mapping


def locked_dropout_mask(reference, probability, training):
    if not training or probability <= 0:
        return torch.ones_like(reference)
    if probability >= 1:
        raise ValueError("locked dropout probability must be below one")
    keep = torch.rand_like(reference).ge(probability).to(reference.dtype)
    return keep / (1.0 - probability)


def gate_ceiling_penalty(output, mask, context_ceiling, audio_ceiling):
    context_excess = torch.relu(output["context_gate"][mask] - context_ceiling)
    audio_excess = torch.relu(output["audio_gate"][mask] - audio_ceiling)
    return (context_excess.square() + audio_excess.square()).mean()


def current_token_mask(encoded, offsets, texts):
    starts = torch.tensor(
        [text.rfind("Current:\n") + len("Current:\n") for text in texts]
    ).unsqueeze(1)
    return (
        offsets[:, :, 0].ge(starts)
        & offsets[:, :, 1].gt(offsets[:, :, 0])
        & encoded["attention_mask"].bool()
        & ~encoded["special_tokens_mask"].bool()
    )


def cache_text_features(records, tokenizer, model, device, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, record in enumerate(records):
        destination = directory / (
            f"dia{record['dialogue_id']}_utt{record['utterance_id']}.npz"
        )
        record["recurrent_text_path"] = str(destination.resolve())
        if args.rebuild_text_cache or not destination.exists():
            pending.append((index, record, destination))
    model.eval()
    for start in range(0, len(pending), args.encoder_batch_size):
        items = pending[start : start + args.encoder_batch_size]
        texts = [item[1]["text"] for item in items]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        mask = current_token_mask(encoded, offsets, texts).to(device)
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1]
        count = mask.sum(dim=1, keepdim=True)
        embedding = (hidden * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
        embedding = torch.where(count.eq(0), hidden[:, 0], embedding)
        for row, (_, _, destination) in enumerate(items):
            np.savez_compressed(
                destination,
                embedding=embedding[row].float().cpu().numpy().astype(np.float16),
                logits=output.logits[row].float().cpu().numpy(),
            )
        completed = min(start + len(items), len(pending))
        if completed and (completed % 500 == 0 or completed == len(pending)):
            print(f"  cached text {completed}/{len(pending)}", flush=True)


def load_cached_features(records_by_split):
    emotion = {
        split: load_summary_matrices(records)[0]
        for split, records in records_by_split.items()
    }
    emotion_stats = fit_normalizer(emotion["train"])
    for split, records in records_by_split.items():
        emotion_values = normalize(emotion[split], emotion_stats)
        for index, (record, emotion_value) in enumerate(zip(records, emotion_values)):
            with np.load(record["recurrent_text_path"]) as cache:
                record["text_embedding"] = cache["embedding"].astype(np.float32)
                record["text_logits"] = cache["logits"].astype(np.float32)
            record["audio_features"] = np.concatenate(
                (emotion_value, record["speaker_acoustic"])
            ).astype(np.float32)
            record["record_index"] = index
    return emotion_stats


class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, records, audio_mapping=None):
        self.records = records
        self.dialogues = group_dialogues(records)
        self.audio_mapping = audio_mapping

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        audio = []
        for record in dialogue:
            source = record
            if self.audio_mapping is not None:
                source = self.records[self.audio_mapping[record["record_index"]]]
            audio.append(source["audio_features"])
        return {
            "text_embeddings": np.stack(
                [record["text_embedding"] for record in dialogue]
            ),
            "text_logits": np.stack([record["text_logits"] for record in dialogue]),
            "audio_features": np.stack(audio),
            "speaker_indices": np.array(
                [record["speaker_index"] for record in dialogue], dtype=np.int64
            ),
            "labels": np.array(
                [EMOTION_LABELS.index(record["label"]) for record in dialogue],
                dtype=np.int64,
            ),
            "record_indices": np.array(
                [record["record_index"] for record in dialogue], dtype=np.int64
            ),
        }


def collate_dialogues(items):
    batch = len(items)
    length = max(len(item["labels"]) for item in items)
    text_dimension = items[0]["text_embeddings"].shape[1]
    class_count = items[0]["text_logits"].shape[1]
    audio_dimension = items[0]["audio_features"].shape[1]
    result = {
        "text_embeddings": torch.zeros(batch, length, text_dimension),
        "text_logits": torch.zeros(batch, length, class_count),
        "audio_features": torch.zeros(batch, length, audio_dimension),
        "speaker_indices": torch.zeros(batch, length, dtype=torch.long),
        "labels": torch.zeros(batch, length, dtype=torch.long),
        "record_indices": torch.zeros(batch, length, dtype=torch.long),
        "valid_mask": torch.zeros(batch, length, dtype=torch.bool),
    }
    for row, item in enumerate(items):
        size = len(item["labels"])
        for key in (
            "text_embeddings", "text_logits", "audio_features", "speaker_indices",
            "labels", "record_indices",
        ):
            result[key][row, :size] = torch.as_tensor(item[key])
        result["valid_mask"][row, :size] = True
    return result


def make_loader(records, args, training, audio_mapping=None):
    return torch.utils.data.DataLoader(
        DialogueDataset(records, audio_mapping),
        batch_size=args.dialogue_batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_dialogues,
        num_workers=0,
    )


class RecurrentDialogueModel(torch.nn.Module):
    def __init__(self, text_dimension, audio_dimension, number_of_classes, args):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.dialogue_dimension = args.dialogue_state_dimension
        self.speaker_dimension = args.speaker_state_dimension
        self.context_max_gate = args.context_max_gate
        self.audio_max_gate = args.audio_max_gate
        self.dialogue_state_dropout = getattr(args, "dialogue_state_dropout", 0.0)
        self.speaker_state_dropout = getattr(args, "speaker_state_dropout", 0.0)
        self.audio_dropout = getattr(args, "audio_dropout", 0.0)
        self.dialogue_reset_probability = getattr(
            args, "dialogue_reset_probability", 0.0
        )
        self.speaker_reset_probability = getattr(
            args, "speaker_reset_probability", 0.0
        )
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(text_dimension, args.text_projection_dimension),
            torch.nn.LayerNorm(args.text_projection_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.audio_projection = torch.nn.Sequential(
            torch.nn.Linear(audio_dimension, args.audio_projection_dimension),
            torch.nn.LayerNorm(args.audio_projection_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        current_dimension = (
            args.text_projection_dimension + args.audio_projection_dimension
        )
        self.dialogue_cell = torch.nn.GRUCell(
            current_dimension + self.speaker_dimension, self.dialogue_dimension
        )
        self.speaker_cell = torch.nn.GRUCell(
            current_dimension + self.dialogue_dimension, self.speaker_dimension
        )
        self.dialogue_norm = torch.nn.LayerNorm(self.dialogue_dimension)
        self.speaker_norm = torch.nn.LayerNorm(self.speaker_dimension)
        context_input = (
            args.text_projection_dimension
            + self.dialogue_dimension
            + self.speaker_dimension
        )
        self.context_hidden = torch.nn.Sequential(
            torch.nn.Linear(context_input, args.text_projection_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
        )
        self.context_correction = torch.nn.Linear(
            args.text_projection_dimension, number_of_classes
        )
        self.audio_correction = torch.nn.Linear(
            args.audio_projection_dimension, number_of_classes
        )
        self.audio_classifier = torch.nn.Linear(
            args.audio_projection_dimension, number_of_classes
        )
        gate_input = number_of_classes + 3
        self.context_gate = torch.nn.Sequential(
            torch.nn.Linear(gate_input, 32), torch.nn.GELU(), torch.nn.Linear(32, 1)
        )
        self.audio_gate = torch.nn.Sequential(
            torch.nn.Linear(gate_input, 32), torch.nn.GELU(), torch.nn.Linear(32, 1)
        )
        for head in (self.context_correction, self.audio_correction):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        for gate in (self.context_gate, self.audio_gate):
            torch.nn.init.zeros_(gate[-1].weight)
            torch.nn.init.constant_(gate[-1].bias, args.initial_gate_bias)

    def forward(self, batch, reset_each_turn=False, zero_audio=False):
        text = self.text_projection(batch["text_embeddings"])
        audio_input = batch["audio_features"]
        if self.training and self.audio_dropout > 0:
            keep_audio = torch.rand(
                *audio_input.shape[:2], 1,
                device=audio_input.device,
                dtype=audio_input.dtype,
            ).ge(self.audio_dropout)
            audio_input = audio_input * keep_audio
        audio = self.audio_projection(audio_input)
        if zero_audio:
            audio = torch.zeros_like(audio)
        batch_size, turns, _ = text.shape
        speaker_count = int(batch["speaker_indices"].max().item()) + 1
        dialogue_state = text.new_zeros(batch_size, self.dialogue_dimension)
        speaker_states = text.new_zeros(
            batch_size, speaker_count, self.speaker_dimension
        )
        dialogue_dropout = locked_dropout_mask(
            dialogue_state, self.dialogue_state_dropout, self.training
        )
        speaker_dropout = locked_dropout_mask(
            speaker_states[:, 0], self.speaker_state_dropout, self.training
        )
        logits, audio_logits = [], []
        context_gates, audio_gates = [], []
        context_corrections, audio_corrections = [], []
        for turn in range(turns):
            valid = batch["valid_mask"][:, turn]
            if reset_each_turn:
                dialogue_state = torch.zeros_like(dialogue_state)
                speaker_states = torch.zeros_like(speaker_states)
            elif self.training and turn > 0:
                reset_dialogue = torch.rand(
                    batch_size, 1, device=text.device
                ).lt(self.dialogue_reset_probability)
                dialogue_state = torch.where(
                    reset_dialogue, torch.zeros_like(dialogue_state), dialogue_state
                )
            speaker_index = batch["speaker_indices"][:, turn]
            gather_index = speaker_index[:, None, None].expand(
                -1, 1, self.speaker_dimension
            )
            previous_speaker = speaker_states.gather(1, gather_index).squeeze(1)
            if self.training and turn > 0:
                reset_speaker = torch.rand(
                    batch_size, 1, device=text.device
                ).lt(self.speaker_reset_probability)
                previous_speaker = torch.where(
                    reset_speaker,
                    torch.zeros_like(previous_speaker),
                    previous_speaker,
                )
            dialogue_view = self.dialogue_norm(dialogue_state) * dialogue_dropout
            speaker_view = self.speaker_norm(previous_speaker) * speaker_dropout
            current = torch.cat((text[:, turn], audio[:, turn]), dim=-1)
            new_dialogue = self.dialogue_cell(
                torch.cat((current, speaker_view), dim=-1), dialogue_view
            )
            new_speaker = self.speaker_cell(
                torch.cat((current, dialogue_view), dim=-1), speaker_view
            )
            normalized_dialogue = self.dialogue_norm(new_dialogue) * dialogue_dropout
            normalized_speaker = self.speaker_norm(new_speaker) * speaker_dropout
            context = self.context_hidden(
                torch.cat(
                    (text[:, turn], normalized_dialogue, normalized_speaker), dim=-1
                )
            )
            base_logits = batch["text_logits"][:, turn]
            probabilities = torch.softmax(base_logits.detach(), dim=-1)
            confidence = probabilities.max(dim=-1, keepdim=True).values
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True
            ) / math.log(self.number_of_classes)
            audio_prediction = self.audio_classifier(audio[:, turn])
            history = speaker_view.norm(dim=-1, keepdim=True)
            gate_input = torch.cat(
                (audio_prediction, confidence, entropy, history), dim=-1
            )
            context_gate = self.context_max_gate * torch.sigmoid(
                self.context_gate(gate_input)
            )
            audio_gate = self.audio_max_gate * torch.sigmoid(
                self.audio_gate(gate_input)
            )
            if zero_audio:
                audio_gate = torch.zeros_like(audio_gate)
            context_correction = self.context_correction(context)
            audio_correction = self.audio_correction(audio[:, turn])
            logits.append(
                base_logits
                + context_gate * context_correction
                + audio_gate * audio_correction
            )
            audio_logits.append(audio_prediction)
            context_gates.append(context_gate)
            audio_gates.append(audio_gate)
            context_corrections.append(context_correction)
            audio_corrections.append(audio_correction)
            valid_column = valid.unsqueeze(1)
            dialogue_state = torch.where(
                valid_column, new_dialogue, dialogue_state
            )
            updated_speakers = speaker_states.scatter(
                1, gather_index, new_speaker.unsqueeze(1)
            )
            speaker_states = torch.where(
                valid[:, None, None], updated_speakers, speaker_states
            )
        stack = lambda values: torch.stack(values, dim=1)
        return {
            "logits": stack(logits),
            "text_logits": batch["text_logits"],
            "audio_logits": stack(audio_logits),
            "context_gate": stack(context_gates),
            "audio_gate": stack(audio_gates),
            "context_correction": stack(context_corrections),
            "audio_correction": stack(audio_corrections),
        }


def shuffle_valid_audio(batch):
    result = dict(batch)
    shuffled = batch["audio_features"].clone()
    valid = batch["valid_mask"]
    labels = batch["labels"][valid]
    audio = batch["audio_features"][valid]
    if len(audio) > 1:
        indices = []
        for index in range(len(labels)):
            candidates = torch.nonzero(labels != labels[index], as_tuple=False).flatten()
            if not len(candidates):
                candidates = torch.arange(len(labels), device=labels.device)
                candidates = candidates[candidates != index]
            indices.append(candidates[index % len(candidates)])
        shuffled[valid] = audio[torch.stack(indices)]
    result["audio_features"] = shuffled
    return result


def masked_cross_entropy(logits, labels, mask, weights):
    return torch.nn.functional.cross_entropy(
        logits[mask], labels[mask], weight=weights
    )


def calculate_loss(
    matched, shuffled, batch, class_weights, args, counterfactual_matched=None
):
    mask, labels = batch["valid_mask"], batch["labels"]
    comparison = matched if counterfactual_matched is None else counterfactual_matched
    fused = masked_cross_entropy(matched["logits"], labels, mask, class_weights)
    audio = masked_cross_entropy(
        matched["audio_logits"], labels, mask, class_weights
    )
    targets = labels[mask].unsqueeze(1)
    matched_support = torch.log_softmax(comparison["logits"][mask], dim=-1).gather(
        1, targets
    )
    shuffled_support = torch.log_softmax(shuffled["logits"][mask], dim=-1).gather(
        1, targets
    )
    ranking = torch.relu(
        args.counterfactual_margin - matched_support + shuffled_support
    ).mean()
    if getattr(args, "direct_audio_mix", False):
        shuffled_audio_residual = shuffled["audio_influence"][mask].square().mean()
        correction = matched["context_correction"][mask].square().mean()
    else:
        shuffled_audio_residual = (
            shuffled["audio_gate"][mask] * shuffled["audio_correction"][mask]
        ).square().mean()
        correction = (
            matched["context_correction"][mask].square().mean()
            + matched["audio_correction"][mask].square().mean()
        )
    gate_penalty = gate_ceiling_penalty(
        matched,
        mask,
        args.context_gate_soft_ceiling,
        args.audio_gate_soft_ceiling,
    )
    total = (
        fused
        + args.audio_loss_weight * audio
        + args.counterfactual_weight * ranking
        + args.negative_residual_weight * shuffled_audio_residual
        + args.correction_penalty_weight * correction
        + args.gate_penalty_weight * gate_penalty
    )
    components = {
        "total": total,
        "fused": fused,
        "audio": audio,
        "counterfactual": ranking,
        "negative_residual": shuffled_audio_residual,
        "correction": correction,
        "gate_penalty": gate_penalty,
        "context_gate": matched["context_gate"][mask].mean(),
        "audio_gate": matched["audio_gate"][mask].mean(),
    }
    return total, {
        key: float(value.detach().cpu()) for key, value in components.items()
    }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, class_weights, args, device):
    model.train()
    rows = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        matched = model(batch)
        model.eval()
        counterfactual_matched = model(batch)
        shuffled = model(shuffle_valid_audio(batch))
        model.train()
        loss, components = calculate_loss(
            matched,
            shuffled,
            batch,
            class_weights,
            args,
            counterfactual_matched=counterfactual_matched,
        )
        (loss / args.gradient_accumulation).backward()
        rows.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def evaluate(model, loader, device, reset_each_turn=False, zero_audio=False):
    actual, predicted, text, indices = [], [], [], []
    context_gates, audio_gates = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch, reset_each_turn=reset_each_turn, zero_audio=zero_audio
            )
            mask = batch["valid_mask"]
            actual.extend(batch["labels"][mask].cpu().tolist())
            predicted.extend(output["logits"][mask].argmax(dim=-1).cpu().tolist())
            text.extend(output["text_logits"][mask].argmax(dim=-1).cpu().tolist())
            indices.extend(batch["record_indices"][mask].cpu().tolist())
            context_gates.extend(output["context_gate"][mask].squeeze(-1).cpu().tolist())
            audio_gates.extend(output["audio_gate"][mask].squeeze(-1).cpu().tolist())
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    return {
        "metrics": compute_metrics(names(actual), names(predicted)),
        "text_metrics": compute_metrics(names(actual), names(text)),
        "actual": names(actual),
        "predicted": names(predicted),
        "indices": indices,
        "mean_context_gate": float(np.mean(context_gates)),
        "mean_audio_gate": float(np.mean(audio_gates)),
    }


def write_predictions(path, records, result):
    by_index = dict(zip(result["indices"], result["predicted"]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "dialogue_id", "utterance_id", "speaker", "utterance",
            "expected", "predicted",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "dialogue_id": record["dialogue_id"],
                    "utterance_id": record["utterance_id"],
                    "speaker": record["speaker"],
                    "utterance": record["utterance"],
                    "expected": record["label"],
                    "predicted": by_index[index],
                }
            )


def make_run_seeds(base_seed, runs, step):
    if runs < 1 or step < 1:
        raise ValueError("runs and seed step must be positive")
    return [base_seed + index * step for index in range(runs)]


def summarize_runs(runs):
    fields = {
        "weighted_f1": [
            run["test_recurrent_matched"]["weighted_f1"] for run in runs
        ],
        "macro_f1": [run["test_recurrent_matched"]["macro_f1"] for run in runs],
        "state_margin": [run["test_state_margin"] for run in runs],
        "audio_margin": [run["test_audio_margin"] for run in runs],
    }
    return {
        name: {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in fields.items()
    }


def load_completed_run(output_dir, expected_seed):
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    return report if report.get("seed") == expected_seed else None


def parse_arguments():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--text-model", type=Path, default=root / "models/text-emotion")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "research/experiments/audio-cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "research/experiments/training-output-recurrent-dialogue-regularized",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dialogue-batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-projection-dimension", type=int, default=256)
    parser.add_argument("--audio-projection-dimension", type=int, default=128)
    parser.add_argument("--dialogue-state-dimension", type=int, default=256)
    parser.add_argument("--speaker-state-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dialogue-state-dropout", type=float, default=0.10)
    parser.add_argument("--speaker-state-dropout", type=float, default=0.15)
    parser.add_argument("--audio-dropout", type=float, default=0.10)
    parser.add_argument("--dialogue-reset-probability", type=float, default=0.03)
    parser.add_argument("--speaker-reset-probability", type=float, default=0.05)
    parser.add_argument("--context-max-gate", type=float, default=0.25)
    parser.add_argument("--audio-max-gate", type=float, default=0.15)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--audio-loss-weight", type=float, default=0.3)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--negative-residual-weight", type=float, default=0.2)
    parser.add_argument("--correction-penalty-weight", type=float, default=0.01)
    parser.add_argument("--context-gate-soft-ceiling", type=float, default=0.18)
    parser.add_argument("--audio-gate-soft-ceiling", type=float, default=0.10)
    parser.add_argument("--gate-penalty-weight", type=float, default=0.2)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    parser.add_argument("--retrain-completed-runs", action="store_true")
    return parser.parse_args()


def validate_arguments(args):
    for name in (
        "epochs", "dialogue_batch_size", "encoder_batch_size",
        "gradient_accumulation", "patience", "runs", "seed_step",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if not args.text_model.exists():
        raise FileNotFoundError(f"text checkpoint not found: {args.text_model}")
    for name in (
        "dropout", "dialogue_state_dropout", "speaker_state_dropout",
        "audio_dropout", "dialogue_reset_probability", "speaker_reset_probability",
    ):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be in [0, 1)")


def train_one_run(args, records, device, seed, output_dir, run_number):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    sample = records["train"][0]
    model = RecurrentDialogueModel(
        len(sample["text_embedding"]),
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        run_args,
    ).to(device)
    loaders = {
        split: make_loader(records[split], run_args, split == "train")
        for split in records
    }
    dev_shuffled = [
        make_loader(
            records["dev"], run_args, False,
            different_label_audio_mapping(records["dev"], seed),
        )
        for seed in run_args.shuffle_seeds
    ]
    train_labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in records["train"]]
    )
    class_weights = torch.from_numpy(
        sqrt_class_weights(train_labels, len(EMOTION_LABELS))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run_args.learning_rate,
        weight_decay=run_args.weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_recurrent_dialogue.pt"
    baseline = evaluate(model, loaders["dev"], device)["text_metrics"]
    print(
        f"Run {run_number:02d}/{args.runs}; seed={seed}; device={device}; "
        f"dialogues: {len(loaders['train'].dataset)}; "
        f"utterances: {len(records['train'])}; text dev weighted F1: "
        f"{baseline['weighted_f1']:.4f}"
    )
    history, best_score, best_epoch, stale = [], (-1, -1.0, -1.0), 0, 0
    started = time.perf_counter()
    for epoch in range(1, run_args.epochs + 1):
        losses = train_epoch(
            model, loaders["train"], optimizer, class_weights, run_args, device
        )
        matched = evaluate(model, loaders["dev"], device)
        reset = evaluate(model, loaders["dev"], device, reset_each_turn=True)
        shuffled = [evaluate(model, loader, device) for loader in dev_shuffled]
        audio_margin = matched["metrics"]["weighted_f1"] - max(
            item["metrics"]["weighted_f1"] for item in shuffled
        )
        state_margin = (
            matched["metrics"]["weighted_f1"] - reset["metrics"]["weighted_f1"]
        )
        eligible = int(
            matched["metrics"]["weighted_f1"] >= baseline["weighted_f1"]
        )
        score = (
            eligible,
            matched["metrics"]["weighted_f1"],
            matched["metrics"]["macro_f1"],
        )
        history.append(
            {
                "epoch": epoch,
                "losses": losses,
                "dev_matched": matched["metrics"],
                "dev_reset_state": reset["metrics"],
                "dev_shuffled": [item["metrics"] for item in shuffled],
                "audio_margin": audio_margin,
                "state_margin": state_margin,
                "mean_context_gate": matched["mean_context_gate"],
                "mean_audio_gate": matched["mean_audio_gate"],
            }
        )
        print(
            f"Epoch {epoch:02d}: loss={losses['total']:.4f} "
            f"macro={matched['metrics']['macro_f1']:.4f} "
            f"weighted={matched['metrics']['weighted_f1']:.4f} "
            f"state_margin={state_margin:+.4f} audio_margin={audio_margin:+.4f}"
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= run_args.patience:
                print(f"Early stopping after epoch {epoch}")
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = evaluate(model, loaders["test"], device)
    test_reset = evaluate(model, loaders["test"], device, reset_each_turn=True)
    test_zero_audio = evaluate(model, loaders["test"], device, zero_audio=True)
    test_shuffled = []
    for shuffle_seed in run_args.shuffle_seeds:
        loader = make_loader(
            records["test"], run_args, False,
            different_label_audio_mapping(records["test"], shuffle_seed),
        )
        test_shuffled.append(evaluate(model, loader, device))
    audio_margin = test["metrics"]["weighted_f1"] - max(
        item["metrics"]["weighted_f1"] for item in test_shuffled
    )
    state_margin = (
        test["metrics"]["weighted_f1"]
        - test_reset["metrics"]["weighted_f1"]
    )
    write_predictions(
        output_dir / "test_predictions.csv", records["test"], test
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(run_args).items()
        },
        "run_number": run_number,
        "seed": seed,
        "samples": {split: len(value) for split, value in records.items()},
        "dialogues": {
            split: len(group_dialogues(value)) for split, value in records.items()
        },
        "best_epoch": best_epoch,
        "test_text": test["text_metrics"],
        "test_recurrent_matched": test["metrics"],
        "test_reset_state": test_reset["metrics"],
        "test_zero_audio": test_zero_audio["metrics"],
        "test_shuffled_audio": [item["metrics"] for item in test_shuffled],
        "test_state_margin": state_margin,
        "test_audio_margin": audio_margin,
        "mean_context_gate": test["mean_context_gate"],
        "mean_audio_gate": test["mean_audio_gate"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test['text_metrics']['weighted_f1']:.4f}")
    print(f"Recurrent test weighted F1: {test['metrics']['weighted_f1']:.4f}")
    print(f"Recurrent test macro F1: {test['metrics']['macro_f1']:.4f}")
    print(f"State margin over reset: {state_margin:+.4f}")
    print(f"Audio margin over worst shuffle: {audio_margin:+.4f}")
    print(f"Run outputs: {output_dir}")
    del optimizer, model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return report


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModelForSequenceClassification.from_pretrained(
        args.text_model
    ).to(device)
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
        cache_text_features(
            records[split], tokenizer, text_model, device,
            args.audio_cache_dir / split / f"recurrent-text-c{args.context_window}",
            args,
        )
    del text_model, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    attach_speaker_relative_acoustics(records)
    emotion_stats = load_cached_features(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=emotion_stats[0], std=emotion_stats[1],
    )
    reports = []
    total_started = time.perf_counter()
    seeds = make_run_seeds(args.seed, args.runs, args.seed_step)
    for run_number, seed in enumerate(seeds, start=1):
        run_dir = args.output_dir / f"run-{run_number:02d}-seed-{seed}"
        completed = (
            None
            if args.retrain_completed_runs
            else load_completed_run(run_dir, seed)
        )
        if completed is not None:
            print(
                f"Run {run_number:02d}/{args.runs}; seed={seed}; "
                "reusing completed result"
            )
            reports.append(completed)
        else:
            reports.append(
                train_one_run(
                    args, records, device, seed, run_dir, run_number
                )
            )
        aggregate = {
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "completed_runs": len(reports),
            "requested_runs": args.runs,
            "seeds": seeds,
            "summary": summarize_runs(reports),
            "runs": reports,
            "runtime_seconds": time.perf_counter() - total_started,
        }
        (args.output_dir / "aggregate_metrics.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
    summary = summarize_runs(reports)
    print(
        f"Completed {len(reports)} runs; weighted F1 "
        f"{summary['weighted_f1']['mean']:.4f} ± "
        f"{summary['weighted_f1']['std']:.4f}; macro F1 "
        f"{summary['macro_f1']['mean']:.4f} ± "
        f"{summary['macro_f1']['std']:.4f}"
    )
    print(f"Aggregate outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
