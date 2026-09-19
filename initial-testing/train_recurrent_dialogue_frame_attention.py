#!/usr/bin/env python3
"""Train context-two recurrence with text-conditioned emotion2vec frame pooling."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import train_recurrent_dialogue as base
import train_recurrent_dialogue_stabilized as stabilized
from evaluate_meld import EMOTION_LABELS, select_device
from train_text import sqrt_class_weights


def uniform_sample_frames(frames, maximum_frames):
    if frames.ndim != 2 or not len(frames):
        raise ValueError("emotion frames must be a nonempty matrix")
    if maximum_frames < 1:
        raise ValueError("maximum frames must be positive")
    if len(frames) <= maximum_frames:
        return frames.astype(np.float32, copy=False)
    indices = np.linspace(0, len(frames) - 1, maximum_frames)
    indices = np.rint(indices).astype(np.int64)
    return frames[indices].astype(np.float32, copy=False)


def attach_emotion_frames(records_by_split, maximum_frames):
    frame_dimension = None
    for split, records in records_by_split.items():
        for position, record in enumerate(records, start=1):
            with np.load(record["phase1_path"]) as cache:
                frames = uniform_sample_frames(
                    cache["emotion_frames"], maximum_frames
                )
            if frame_dimension is None:
                frame_dimension = frames.shape[1]
            elif frames.shape[1] != frame_dimension:
                raise ValueError("emotion2vec frame dimensions are inconsistent")
            record["emotion_frames"] = frames.astype(np.float16)
            if position % 1000 == 0 or position == len(records):
                print(
                    f"  loaded {split} frames {position}/{len(records)}",
                    flush=True,
                )
    return frame_dimension


class FrameDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, records, audio_mapping=None):
        self.records = records
        self.dialogues = base.group_dialogues(records)
        self.audio_mapping = audio_mapping

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        audio_records = []
        for record in dialogue:
            source = record
            if self.audio_mapping is not None:
                source = self.records[self.audio_mapping[record["record_index"]]]
            audio_records.append(source)
        return {
            "text_embeddings": np.stack(
                [record["text_embedding"] for record in dialogue]
            ),
            "text_logits": np.stack(
                [record["text_logits"] for record in dialogue]
            ),
            "acoustic_features": np.stack(
                [record["speaker_acoustic"] for record in audio_records]
            ),
            "emotion_frames": [
                record["emotion_frames"] for record in audio_records
            ],
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


def collate_frame_dialogues(items):
    batch_size = len(items)
    turns = max(len(item["labels"]) for item in items)
    maximum_frames = max(
        len(frames) for item in items for frames in item["emotion_frames"]
    )
    text_dimension = items[0]["text_embeddings"].shape[1]
    class_count = items[0]["text_logits"].shape[1]
    acoustic_dimension = items[0]["acoustic_features"].shape[1]
    frame_dimension = items[0]["emotion_frames"][0].shape[1]
    result = {
        "text_embeddings": torch.zeros(batch_size, turns, text_dimension),
        "text_logits": torch.zeros(batch_size, turns, class_count),
        "acoustic_features": torch.zeros(batch_size, turns, acoustic_dimension),
        "emotion_frames": torch.zeros(
            batch_size, turns, maximum_frames, frame_dimension
        ),
        "frame_mask": torch.zeros(
            batch_size, turns, maximum_frames, dtype=torch.bool
        ),
        "speaker_indices": torch.zeros(batch_size, turns, dtype=torch.long),
        "labels": torch.zeros(batch_size, turns, dtype=torch.long),
        "record_indices": torch.zeros(batch_size, turns, dtype=torch.long),
        "valid_mask": torch.zeros(batch_size, turns, dtype=torch.bool),
    }
    for row, item in enumerate(items):
        size = len(item["labels"])
        for key in (
            "text_embeddings",
            "text_logits",
            "acoustic_features",
            "speaker_indices",
            "labels",
            "record_indices",
        ):
            result[key][row, :size] = torch.as_tensor(item[key])
        for turn, frames in enumerate(item["emotion_frames"]):
            frame_count = len(frames)
            result["emotion_frames"][row, turn, :frame_count] = torch.as_tensor(
                frames.astype(np.float32)
            )
            result["frame_mask"][row, turn, :frame_count] = True
        result["valid_mask"][row, :size] = True
    return result


def make_loader(records, args, training, audio_mapping=None):
    return torch.utils.data.DataLoader(
        FrameDialogueDataset(records, audio_mapping),
        batch_size=args.dialogue_batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_frame_dialogues,
        num_workers=0,
    )


class TextConditionedFramePool(torch.nn.Module):
    def __init__(self, text_dimension, frame_dimension):
        super().__init__()
        self.frame_dimension = frame_dimension
        self.query = torch.nn.Linear(text_dimension, frame_dimension)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, text, frames, frame_mask):
        normalized_frames = torch.nn.functional.layer_norm(
            frames, (self.frame_dimension,)
        )
        query = torch.nn.functional.normalize(self.query(text), dim=-1)
        keys = torch.nn.functional.normalize(normalized_frames, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        scores = scale * torch.einsum("btd,btfd->btf", query, keys)
        safe_mask = frame_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        safe_mask[:, :, 0] |= empty
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * frame_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("btf,btfd->btd", weights, normalized_frames)
        return pooled, weights


class FrameAttentionRecurrentModel(stabilized.StabilizedRecurrentDialogueModel):
    def __init__(
        self,
        text_dimension,
        frame_dimension,
        acoustic_dimension,
        number_of_classes,
        args,
    ):
        super().__init__(
            text_dimension,
            frame_dimension + acoustic_dimension,
            number_of_classes,
            args,
        )
        self.frame_pool = TextConditionedFramePool(
            text_dimension, frame_dimension
        )

    def forward(self, batch, reset_each_turn=False, zero_audio=False):
        pooled, attention = self.frame_pool(
            batch["text_embeddings"],
            batch["emotion_frames"],
            batch["frame_mask"],
        )
        recurrent_batch = dict(batch)
        recurrent_batch["audio_features"] = torch.cat(
            (pooled, batch["acoustic_features"]), dim=-1
        )
        output = super().forward(
            recurrent_batch,
            reset_each_turn=reset_each_turn,
            zero_audio=zero_audio,
        )
        output["frame_attention"] = attention
        return output


def shuffle_frame_audio(batch):
    result = dict(batch)
    valid = batch["valid_mask"]
    labels = batch["labels"][valid]
    if len(labels) <= 1:
        return result
    indices = []
    for index in range(len(labels)):
        candidates = torch.nonzero(
            labels != labels[index], as_tuple=False
        ).flatten()
        if not len(candidates):
            candidates = torch.arange(len(labels), device=labels.device)
            candidates = candidates[candidates != index]
        indices.append(candidates[index % len(candidates)])
    indices = torch.stack(indices)
    for key in ("emotion_frames", "frame_mask", "acoustic_features"):
        values = batch[key].clone()
        values[valid] = batch[key][valid][indices]
        result[key] = values
    return result


def train_epoch(model, loader, optimizer, class_weights, args, device):
    model.train()
    rows = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = base.move_batch(batch, device)
        matched = model(batch)
        model.eval()
        clean = model(batch)
        reset = model(batch, reset_each_turn=True)
        shuffled = model(shuffle_frame_audio(batch))
        model.train()
        loss, components = stabilized.calculate_loss(
            matched,
            clean,
            reset,
            shuffled,
            batch,
            class_weights,
            args,
        )
        (loss / args.gradient_accumulation).backward()
        rows.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--text-model", type=Path, default=root / "initial-testing/training-output-text/best-model")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/training-output-recurrent-dialogue-frame-attention")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dialogue-batch-size", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-projection-dimension", type=int, default=256)
    parser.add_argument("--audio-projection-dimension", type=int, default=128)
    parser.add_argument("--dialogue-state-dimension", type=int, default=128)
    parser.add_argument("--speaker-state-dimension", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dialogue-state-dropout", type=float, default=0.05)
    parser.add_argument("--speaker-state-dropout", type=float, default=0.10)
    parser.add_argument("--audio-dropout", type=float, default=0.10)
    parser.add_argument("--dialogue-reset-probability", type=float, default=0.01)
    parser.add_argument("--speaker-reset-probability", type=float, default=0.03)
    parser.add_argument("--context-max-gate", type=float, default=0.25)
    parser.add_argument("--audio-max-gate", type=float, default=0.15)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--audio-loss-weight", type=float, default=0.3)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--state-counterfactual-weight", type=float, default=0.3)
    parser.add_argument("--state-counterfactual-margin", type=float, default=0.02)
    parser.add_argument("--minimum-dev-state-margin", type=float, default=0.002)
    parser.add_argument("--negative-residual-weight", type=float, default=0.2)
    parser.add_argument("--correction-penalty-weight", type=float, default=0.01)
    parser.add_argument("--context-gate-soft-ceiling", type=float, default=0.18)
    parser.add_argument("--audio-gate-soft-ceiling", type=float, default=0.10)
    parser.add_argument("--gate-penalty-weight", type=float, default=0.2)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-audio-frames", type=int, default=32)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    for name in (
        "epochs", "patience", "dialogue_batch_size", "encoder_batch_size",
        "gradient_accumulation", "max_audio_frames",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if not args.text_model.exists():
        raise FileNotFoundError(f"text checkpoint not found: {args.text_model}")
    for name in (
        "dropout", "dialogue_state_dropout", "speaker_state_dropout",
        "audio_dropout", "dialogue_reset_probability",
        "speaker_reset_probability",
    ):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be in [0, 1)")


def train(args, records, frame_dimension, device):
    sample = records["train"][0]
    model = FrameAttentionRecurrentModel(
        len(sample["text_embedding"]),
        frame_dimension,
        len(sample["speaker_acoustic"]),
        len(EMOTION_LABELS),
        args,
    ).to(device)
    loaders = {
        split: make_loader(records[split], args, split == "train")
        for split in records
    }
    dev_shuffled = [
        make_loader(
            records["dev"], args, False,
            base.different_label_audio_mapping(records["dev"], seed),
        )
        for seed in args.shuffle_seeds
    ]
    labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in records["train"]]
    )
    weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    baseline = base.evaluate(model, loaders["dev"], device)["text_metrics"]
    checkpoint = args.output_dir / "best_frame_attention.pt"
    print(
        f"Device: {device}; dialogues: {len(loaders['train'].dataset)}; "
        f"utterances: {len(records['train'])}; frames: {args.max_audio_frames}; "
        f"text dev weighted F1: {baseline['weighted_f1']:.4f}"
    )
    history = []
    best_score = (-1, -1.0, -1.0, -1.0)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, loaders["train"], optimizer, weights, args, device)
        matched = base.evaluate(model, loaders["dev"], device)
        reset = base.evaluate(model, loaders["dev"], device, reset_each_turn=True)
        zero = base.evaluate(model, loaders["dev"], device, zero_audio=True)
        shuffled = [base.evaluate(model, loader, device) for loader in dev_shuffled]
        state_margin = matched["metrics"]["weighted_f1"] - reset["metrics"]["weighted_f1"]
        zero_margin = matched["metrics"]["weighted_f1"] - zero["metrics"]["weighted_f1"]
        audio_margin = matched["metrics"]["weighted_f1"] - max(
            item["metrics"]["weighted_f1"] for item in shuffled
        )
        score = stabilized.checkpoint_score(
            matched["metrics"], baseline, state_margin,
            args.minimum_dev_state_margin,
        )
        history.append({
            "epoch": epoch,
            "losses": losses,
            "dev_matched": matched["metrics"],
            "dev_reset_state": reset["metrics"],
            "dev_zero_audio": zero["metrics"],
            "dev_shuffled": [item["metrics"] for item in shuffled],
            "state_margin": state_margin,
            "zero_audio_margin": zero_margin,
            "audio_margin": audio_margin,
            "checkpoint_eligible": bool(score[0]),
        })
        print(
            f"Epoch {epoch:02d}: loss={losses['total']:.4f} "
            f"macro={matched['metrics']['macro_f1']:.4f} "
            f"weighted={matched['metrics']['weighted_f1']:.4f} "
            f"state={state_margin:+.4f} zero_audio={zero_margin:+.4f} "
            f"shuffle={audio_margin:+.4f} eligible={bool(score[0])}"
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = base.evaluate(model, loaders["test"], device)
    reset = base.evaluate(model, loaders["test"], device, reset_each_turn=True)
    zero = base.evaluate(model, loaders["test"], device, zero_audio=True)
    shuffled = []
    for seed in args.shuffle_seeds:
        loader = make_loader(
            records["test"], args, False,
            base.different_label_audio_mapping(records["test"], seed),
        )
        shuffled.append(base.evaluate(model, loader, device))
    state_margin = test["metrics"]["weighted_f1"] - reset["metrics"]["weighted_f1"]
    zero_margin = test["metrics"]["weighted_f1"] - zero["metrics"]["weighted_f1"]
    audio_margin = test["metrics"]["weighted_f1"] - max(
        item["metrics"]["weighted_f1"] for item in shuffled
    )
    base.write_predictions(args.output_dir / "test_predictions.csv", records["test"], test)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "best_epoch": best_epoch,
        "best_checkpoint_eligible": bool(best_score[0]),
        "test_text": test["text_metrics"],
        "test_recurrent_matched": test["metrics"],
        "test_reset_state": reset["metrics"],
        "test_zero_audio": zero["metrics"],
        "test_shuffled_audio": [item["metrics"] for item in shuffled],
        "test_state_margin": state_margin,
        "test_zero_audio_margin": zero_margin,
        "test_audio_margin": audio_margin,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test['text_metrics']['weighted_f1']:.4f}")
    print(f"Frame-attention test weighted F1: {test['metrics']['weighted_f1']:.4f}")
    print(f"Frame-attention test macro F1: {test['metrics']['macro_f1']:.4f}")
    print(f"State margin over reset: {state_margin:+.4f}")
    print(f"Audio margin over zero audio: {zero_margin:+.4f}")
    print(f"Audio margin over worst shuffle: {audio_margin:+.4f}")
    print(f"Outputs: {args.output_dir}")


def main():
    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    records, normalization = stabilized.prepare_records(args, device)
    frame_dimension = attach_emotion_frames(records, args.max_audio_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0], std=normalization[1],
    )
    train(args, records, frame_dimension, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
