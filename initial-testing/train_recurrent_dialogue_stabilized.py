#!/usr/bin/env python3
"""Train one stabilized causal dialogue-state model on cached MELD features."""

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
from cache_embeddings import load_split_rows
from evaluate_meld import EMOTION_LABELS, select_device
from train_text import sqrt_class_weights
from train_text_audio_phase2 import (
    attach_speaker_relative_acoustics,
    combine_records,
)


def initialize_gru(cell, update_bias=1.0):
    torch.nn.init.xavier_uniform_(cell.weight_ih)
    for recurrent_gate in cell.weight_hh.chunk(3, dim=0):
        torch.nn.init.orthogonal_(recurrent_gate)
    torch.nn.init.zeros_(cell.bias_ih)
    torch.nn.init.zeros_(cell.bias_hh)
    start = cell.hidden_size
    cell.bias_ih.data[start : 2 * start].fill_(update_bias)


class StabilizedRecurrentDialogueModel(base.RecurrentDialogueModel):
    """Use only prior state for a context residual that vanishes at zero state."""

    def __init__(self, text_dimension, audio_dimension, number_of_classes, args):
        super().__init__(text_dimension, audio_dimension, number_of_classes, args)
        self.direct_audio_mix = getattr(args, "direct_audio_mix", False)
        self.disagreement_gate = getattr(args, "disagreement_gate", False)
        if self.direct_audio_mix:
            self.text_log_temperature = torch.nn.Parameter(torch.zeros(()))
            self.audio_log_temperature = torch.nn.Parameter(torch.zeros(()))
        self.context_hidden = torch.nn.Sequential(*list(self.context_hidden)[:-1])
        initialize_gru(self.dialogue_cell)
        initialize_gru(self.speaker_cell)

    def forward(self, batch, reset_each_turn=False, zero_audio=False):
        text = self.text_projection(batch["text_embeddings"])
        audio_input = batch["audio_features"]
        if self.training and self.audio_dropout > 0:
            keep_audio = torch.rand(
                *audio_input.shape[:2],
                1,
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
        dialogue_dropout = base.locked_dropout_mask(
            dialogue_state, self.dialogue_state_dropout, self.training
        )
        speaker_dropout = base.locked_dropout_mask(
            speaker_states[:, 0], self.speaker_state_dropout, self.training
        )
        zero_dialogue = torch.zeros_like(dialogue_state)
        zero_speaker = torch.zeros_like(speaker_states[:, 0])
        zero_dialogue_view = self.dialogue_norm(zero_dialogue)
        zero_speaker_view = self.speaker_norm(zero_speaker)

        logits, audio_logits = [], []
        context_gates, audio_gates = [], []
        context_corrections, audio_corrections, audio_influences = [], [], []
        for turn in range(turns):
            valid = batch["valid_mask"][:, turn]
            if reset_each_turn:
                dialogue_state = torch.zeros_like(dialogue_state)
                speaker_states = torch.zeros_like(speaker_states)
            elif self.training and turn > 0:
                reset_dialogue = torch.rand(batch_size, 1, device=text.device).lt(
                    self.dialogue_reset_probability
                )
                dialogue_state = torch.where(
                    reset_dialogue, torch.zeros_like(dialogue_state), dialogue_state
                )
            forced_reset = batch.get("forced_reset_mask")
            if forced_reset is not None:
                reset_rows = forced_reset[:, turn].unsqueeze(1)
                dialogue_state = torch.where(
                    reset_rows, torch.zeros_like(dialogue_state), dialogue_state
                )
                speaker_states = torch.where(
                    reset_rows.unsqueeze(2),
                    torch.zeros_like(speaker_states),
                    speaker_states,
                )

            speaker_index = batch["speaker_indices"][:, turn]
            gather_index = speaker_index[:, None, None].expand(
                -1, 1, self.speaker_dimension
            )
            previous_speaker = speaker_states.gather(1, gather_index).squeeze(1)
            if self.training and turn > 0:
                reset_speaker = torch.rand(batch_size, 1, device=text.device).lt(
                    self.speaker_reset_probability
                )
                previous_speaker = torch.where(
                    reset_speaker,
                    torch.zeros_like(previous_speaker),
                    previous_speaker,
                )

            dialogue_view = self.dialogue_norm(dialogue_state) * dialogue_dropout
            speaker_view = self.speaker_norm(previous_speaker) * speaker_dropout
            current_text = text[:, turn]
            context_with_state = self.context_hidden(
                torch.cat((current_text, dialogue_view, speaker_view), dim=-1)
            )
            context_without_state = self.context_hidden(
                torch.cat(
                    (current_text, zero_dialogue_view, zero_speaker_view), dim=-1
                )
            )
            context_correction = self.context_correction(context_with_state) - (
                self.context_correction(context_without_state)
            )

            base_logits = batch["text_logits"][:, turn]
            if self.direct_audio_mix:
                text_temperature = self.text_log_temperature.clamp(
                    min=-1.3863, max=1.3863
                ).exp()
                audio_temperature = self.audio_log_temperature.clamp(
                    min=-1.3863, max=1.3863
                ).exp()
            else:
                text_temperature = audio_temperature = 1.0
            probabilities = torch.softmax(
                base_logits.detach() / text_temperature, dim=-1
            )
            confidence = probabilities.max(dim=-1, keepdim=True).values
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True
            ) / math.log(self.number_of_classes)
            audio_prediction = self.audio_classifier(audio[:, turn])
            history = speaker_view.norm(dim=-1, keepdim=True)
            context_gate_input = torch.cat(
                (audio_prediction, confidence, entropy, history), dim=-1
            )
            if self.disagreement_gate:
                text_probabilities = probabilities.detach()
                audio_probabilities = torch.softmax(
                    audio_prediction.detach() / audio_temperature, dim=-1
                )
                audio_confidence = audio_probabilities.max(
                    dim=-1, keepdim=True
                ).values
                audio_entropy = -(
                    audio_probabilities
                    * audio_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1, keepdim=True) / math.log(self.number_of_classes)
                disagrees = text_probabilities.argmax(dim=-1).ne(
                    audio_probabilities.argmax(dim=-1)
                ).to(text.dtype).unsqueeze(1)
                midpoint = 0.5 * (text_probabilities + audio_probabilities)
                divergence = 0.5 * (
                    (
                        text_probabilities
                        * (
                            text_probabilities.clamp_min(1e-8).log()
                            - midpoint.clamp_min(1e-8).log()
                        )
                    ).sum(dim=-1, keepdim=True)
                    + (
                        audio_probabilities
                        * (
                            audio_probabilities.clamp_min(1e-8).log()
                            - midpoint.clamp_min(1e-8).log()
                        )
                    ).sum(dim=-1, keepdim=True)
                ) / math.log(2.0)
                acoustic_strength = audio_input[:, turn].norm(
                    dim=-1, keepdim=True
                ) / math.sqrt(audio_input.shape[-1])
                if "frame_mask" in batch:
                    frame_fraction = batch["frame_mask"][:, turn].sum(
                        dim=-1, keepdim=True
                    ).to(text.dtype) / 300.0
                else:
                    frame_fraction = torch.zeros_like(history)
                gate_input = torch.cat(
                    (
                        text_probabilities,
                        audio_probabilities,
                        audio_probabilities - text_probabilities,
                        confidence,
                        audio_confidence,
                        entropy,
                        audio_entropy,
                        disagrees,
                        divergence,
                        history,
                        acoustic_strength,
                        frame_fraction,
                    ),
                    dim=-1,
                )
            else:
                gate_input = context_gate_input
            context_gate = self.context_max_gate * torch.sigmoid(
                self.context_gate(context_gate_input)
            )
            audio_gate = self.audio_max_gate * torch.sigmoid(
                self.audio_gate(gate_input)
            )
            if zero_audio:
                audio_gate = torch.zeros_like(audio_gate)
            audio_correction = self.audio_correction(audio[:, turn])
            if self.direct_audio_mix:
                text_evidence = torch.log_softmax(
                    base_logits / text_temperature, dim=-1
                )
                audio_evidence = torch.log_softmax(
                    audio_prediction / audio_temperature, dim=-1
                )
                audio_influence = audio_gate * (audio_evidence - text_evidence)
                fused_logits = (
                    text_evidence
                    + audio_influence
                    + context_gate * context_correction
                )
            else:
                audio_influence = audio_gate * audio_correction
                fused_logits = (
                    base_logits
                    + context_gate * context_correction
                    + audio_influence
                )
            logits.append(fused_logits)
            audio_logits.append(audio_prediction)
            context_gates.append(context_gate)
            audio_gates.append(audio_gate)
            context_corrections.append(context_correction)
            audio_corrections.append(audio_correction)
            audio_influences.append(audio_influence)

            current = torch.cat((current_text, audio[:, turn]), dim=-1)
            new_dialogue = self.dialogue_cell(
                torch.cat((current, speaker_view), dim=-1), dialogue_view
            )
            new_speaker = self.speaker_cell(
                torch.cat((current, dialogue_view), dim=-1), speaker_view
            )
            valid_column = valid.unsqueeze(1)
            dialogue_state = torch.where(valid_column, new_dialogue, dialogue_state)
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
            "audio_influence": stack(audio_influences),
        }


def state_ranking_loss(matched_logits, reset_logits, labels, mask, margin):
    targets = labels[mask].unsqueeze(1)
    matched_support = torch.log_softmax(
        matched_logits[mask], dim=-1
    ).gather(1, targets)
    reset_support = torch.log_softmax(reset_logits[mask], dim=-1).gather(
        1, targets
    )
    return torch.relu(margin - matched_support + reset_support).mean()


def disagreement_gate_loss(output, batch):
    mask = batch["valid_mask"]
    labels = batch["labels"][mask]
    text_correct = output["text_logits"][mask].argmax(dim=-1).eq(labels)
    audio_correct = output["audio_logits"][mask].argmax(dim=-1).eq(labels)
    gate = output["audio_gate"][mask].squeeze(-1)
    audio_wins = audio_correct & ~text_correct
    text_wins = text_correct & ~audio_correct
    losses = []
    if audio_wins.any():
        losses.append(
            torch.nn.functional.binary_cross_entropy(
                gate[audio_wins], torch.ones_like(gate[audio_wins])
            )
        )
    if text_wins.any():
        losses.append(
            torch.nn.functional.binary_cross_entropy(
                gate[text_wins], torch.zeros_like(gate[text_wins])
            )
        )
    if not losses:
        return gate.sum() * 0.0
    return torch.stack(losses).mean()


def checkpoint_score(matched, baseline, state_margin, minimum_state_margin):
    eligible = int(
        matched["weighted_f1"] >= baseline["weighted_f1"]
        and state_margin >= minimum_state_margin
    )
    return (
        eligible,
        matched["weighted_f1"],
        state_margin,
        matched["macro_f1"],
    )


def summarize_runs(runs):
    fields = {
        "weighted_f1": [
            run["test_recurrent_matched"]["weighted_f1"] for run in runs
        ],
        "macro_f1": [
            run["test_recurrent_matched"]["macro_f1"] for run in runs
        ],
        "state_margin": [run["test_state_margin"] for run in runs],
        "audio_margin": [run["test_audio_margin"] for run in runs],
        "zero_audio_margin": [run["test_zero_audio_margin"] for run in runs],
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


def calculate_loss(matched, clean, reset, shuffled, batch, class_weights, args):
    total, components = base.calculate_loss(
        matched,
        shuffled,
        batch,
        class_weights,
        args,
        counterfactual_matched=clean,
    )
    state_ranking = state_ranking_loss(
        clean["logits"],
        reset["logits"],
        batch["labels"],
        batch["valid_mask"],
        args.state_counterfactual_margin,
    )
    total = total + args.state_counterfactual_weight * state_ranking
    components["state_counterfactual"] = float(state_ranking.detach().cpu())
    if getattr(args, "disagreement_gate", False):
        gate_supervision = disagreement_gate_loss(matched, batch)
        total = total + args.disagreement_gate_weight * gate_supervision
        components["disagreement_gate"] = float(
            gate_supervision.detach().cpu()
        )
    components["total"] = float(total.detach().cpu())
    return total, components


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
        shuffled = model(base.shuffle_valid_audio(batch))
        model.train()
        loss, components = calculate_loss(
            matched, clean, reset, shuffled, batch, class_weights, args
        )
        (loss / args.gradient_accumulation).backward()
        rows.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz"
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        default=root / "initial-testing/training-output-text/best-model",
    )
    parser.add_argument(
        "--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "initial-testing/training-output-recurrent-dialogue-stabilized",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dialogue-batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
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
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    return parser.parse_args()


def validate_arguments(args):
    for name in (
        "epochs",
        "dialogue_batch_size",
        "encoder_batch_size",
        "gradient_accumulation",
        "patience",
        "runs",
        "seed_step",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if not args.text_model.exists():
        raise FileNotFoundError(f"text checkpoint not found: {args.text_model}")
    for name in (
        "dropout",
        "dialogue_state_dropout",
        "speaker_state_dropout",
        "audio_dropout",
        "dialogue_reset_probability",
        "speaker_reset_probability",
    ):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be in [0, 1)")


def prepare_records(args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        base.cache_text_features(
            records[split],
            tokenizer,
            text_model,
            device,
            args.audio_cache_dir
            / split
            / f"recurrent-text-c{args.context_window}",
            args,
        )
    del text_model, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    attach_speaker_relative_acoustics(records)
    normalization = base.load_cached_features(records)
    return records, normalization


def train(args, records, device, run_number):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sample = records["train"][0]
    model = StabilizedRecurrentDialogueModel(
        len(sample["text_embedding"]),
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        args,
    ).to(device)
    loaders = {
        split: base.make_loader(records[split], args, split == "train")
        for split in records
    }
    dev_shuffled = [
        base.make_loader(
            records["dev"],
            args,
            False,
            base.different_label_audio_mapping(records["dev"], seed),
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
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_recurrent_dialogue_stabilized.pt"
    baseline = base.evaluate(model, loaders["dev"], device)["text_metrics"]
    print(
        f"Run {run_number:02d}/{args.runs}; seed={args.seed}; device={device}; "
        f"dialogues: {len(loaders['train'].dataset)}; "
        f"utterances: {len(records['train'])}; text dev weighted F1: "
        f"{baseline['weighted_f1']:.4f}"
    )

    history = []
    best_score = (-1, -1.0, -1.0, -1.0)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(
            model, loaders["train"], optimizer, class_weights, args, device
        )
        matched = base.evaluate(model, loaders["dev"], device)
        reset = base.evaluate(
            model, loaders["dev"], device, reset_each_turn=True
        )
        shuffled = [base.evaluate(model, loader, device) for loader in dev_shuffled]
        state_margin = (
            matched["metrics"]["weighted_f1"]
            - reset["metrics"]["weighted_f1"]
        )
        audio_margin = matched["metrics"]["weighted_f1"] - max(
            item["metrics"]["weighted_f1"] for item in shuffled
        )
        score = checkpoint_score(
            matched["metrics"],
            baseline,
            state_margin,
            args.minimum_dev_state_margin,
        )
        history.append(
            {
                "epoch": epoch,
                "losses": losses,
                "dev_matched": matched["metrics"],
                "dev_reset_state": reset["metrics"],
                "dev_shuffled": [item["metrics"] for item in shuffled],
                "state_margin": state_margin,
                "audio_margin": audio_margin,
                "checkpoint_eligible": bool(score[0]),
                "mean_context_gate": matched["mean_context_gate"],
                "mean_audio_gate": matched["mean_audio_gate"],
            }
        )
        print(
            f"Epoch {epoch:02d}: loss={losses['total']:.4f} "
            f"macro={matched['metrics']['macro_f1']:.4f} "
            f"weighted={matched['metrics']['weighted_f1']:.4f} "
            f"state_margin={state_margin:+.4f} "
            f"audio_margin={audio_margin:+.4f} eligible={bool(score[0])}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    test = base.evaluate(model, loaders["test"], device)
    test_reset = base.evaluate(
        model, loaders["test"], device, reset_each_turn=True
    )
    test_zero_audio = base.evaluate(
        model, loaders["test"], device, zero_audio=True
    )
    test_shuffled = []
    for seed in args.shuffle_seeds:
        loader = base.make_loader(
            records["test"],
            args,
            False,
            base.different_label_audio_mapping(records["test"], seed),
        )
        test_shuffled.append(base.evaluate(model, loader, device))
    state_margin = (
        test["metrics"]["weighted_f1"]
        - test_reset["metrics"]["weighted_f1"]
    )
    audio_margin = test["metrics"]["weighted_f1"] - max(
        item["metrics"]["weighted_f1"] for item in test_shuffled
    )
    zero_audio_margin = (
        test["metrics"]["weighted_f1"]
        - test_zero_audio["metrics"]["weighted_f1"]
    )
    base.write_predictions(
        args.output_dir / "test_predictions.csv", records["test"], test
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "run_number": run_number,
        "seed": args.seed,
        "samples": {split: len(values) for split, values in records.items()},
        "dialogues": {
            split: len(base.group_dialogues(values))
            for split, values in records.items()
        },
        "best_epoch": best_epoch,
        "best_checkpoint_eligible": bool(best_score[0]),
        "test_text": test["text_metrics"],
        "test_recurrent_matched": test["metrics"],
        "test_reset_state": test_reset["metrics"],
        "test_zero_audio": test_zero_audio["metrics"],
        "test_shuffled_audio": [item["metrics"] for item in test_shuffled],
        "test_state_margin": state_margin,
        "test_audio_margin": audio_margin,
        "test_zero_audio_margin": zero_audio_margin,
        "mean_context_gate": test["mean_context_gate"],
        "mean_audio_gate": test["mean_audio_gate"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test['text_metrics']['weighted_f1']:.4f}")
    print(f"Recurrent test weighted F1: {test['metrics']['weighted_f1']:.4f}")
    print(f"Recurrent test macro F1: {test['metrics']['macro_f1']:.4f}")
    print(f"State margin over reset: {state_margin:+.4f}")
    print(f"Audio margin over zero audio: {zero_audio_margin:+.4f}")
    print(f"Audio margin over worst shuffle: {audio_margin:+.4f}")
    print(f"Outputs: {args.output_dir}")
    return report


def main():
    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    records, normalization = prepare_records(args, device)
    aggregate_output_dir = args.output_dir
    aggregate_output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        aggregate_output_dir / "emotion_normalization.npz",
        mean=normalization[0],
        std=normalization[1],
    )
    reports = []
    started = time.perf_counter()
    seeds = base.make_run_seeds(args.seed, args.runs, args.seed_step)
    for run_number, seed in enumerate(seeds, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        run_args.output_dir = (
            aggregate_output_dir / f"run-{run_number:02d}-seed-{seed}"
        )
        reports.append(train(run_args, records, device, run_number))
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
            "runtime_seconds": time.perf_counter() - started,
        }
        (aggregate_output_dir / "aggregate_metrics.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    summary = summarize_runs(reports)
    print(
        f"Completed {len(reports)} runs; weighted F1 "
        f"{summary['weighted_f1']['mean']:.4f} ± "
        f"{summary['weighted_f1']['std']:.4f}; macro F1 "
        f"{summary['macro_f1']['mean']:.4f} ± "
        f"{summary['macro_f1']['std']:.4f}; zero-audio margin "
        f"{summary['zero_audio_margin']['mean']:+.4f}"
    )
    print(f"Aggregate outputs: {aggregate_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
