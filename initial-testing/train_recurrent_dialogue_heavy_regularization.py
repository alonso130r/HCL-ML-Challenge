#!/usr/bin/env python3
"""Train a moderately regularized context-two recurrent MELD model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import train_recurrent_dialogue as base
import train_recurrent_dialogue_stabilized as stabilized
from evaluate_meld import EMOTION_LABELS, select_device
from train_text import sqrt_class_weights


NORMAL = 0
ZERO_AUDIO = 1
RESET_STATE = 2
ZERO_AUDIO_AND_RESET = 3
SHUFFLED_AUDIO = 4


def sample_modality_conditions(batch_size, device):
    draws = torch.rand(batch_size, device=device)
    boundaries = torch.tensor([0.70, 0.80, 0.88, 0.90], device=device)
    return torch.bucketize(draws, boundaries)


def apply_modality_conditions(batch, conditions):
    result = dict(batch)
    audio = batch["audio_features"].clone()
    shuffled = base.shuffle_valid_audio(batch)["audio_features"]
    zero_rows = (conditions == ZERO_AUDIO) | (
        conditions == ZERO_AUDIO_AND_RESET
    )
    shuffled_rows = conditions == SHUFFLED_AUDIO
    audio[zero_rows] = 0
    audio[shuffled_rows] = shuffled[shuffled_rows]
    reset_rows = (conditions == RESET_STATE) | (
        conditions == ZERO_AUDIO_AND_RESET
    )
    result["audio_features"] = audio
    result["forced_reset_mask"] = (
        reset_rows.unsqueeze(1) & batch["valid_mask"]
    )
    return result


def corrupt_audio_features(
    audio,
    group_slices,
    feature_mask_probability,
    group_mask_probability,
    noise_std,
    gain_range,
    generator=None,
):
    random_options = {"device": audio.device, "generator": generator}
    gain = torch.empty(
        audio.shape[0], 1, 1, device=audio.device, dtype=audio.dtype
    ).uniform_(gain_range[0], gain_range[1], generator=generator)
    corrupted = audio * gain
    if noise_std > 0:
        noise = torch.randn(
            audio.shape,
            dtype=audio.dtype,
            **random_options,
        )
        corrupted = corrupted + noise_std * noise
    if feature_mask_probability > 0:
        feature_mask = torch.rand(
            audio.shape, dtype=audio.dtype, **random_options
        ).ge(feature_mask_probability)
        corrupted = corrupted * feature_mask
    if group_mask_probability > 0 and group_slices:
        apply_group = torch.rand(audio.shape[0], **random_options).lt(
            group_mask_probability
        )
        choices = torch.randint(
            len(group_slices), (audio.shape[0],), **random_options
        )
        for row in torch.nonzero(apply_group, as_tuple=False).flatten().tolist():
            start, end = group_slices[int(choices[row])]
            corrupted[row, :, start:end] = 0
    return corrupted


def random_history_reset_mask(valid_mask, probability, generator=None):
    mask = torch.zeros_like(valid_mask)
    for row in range(len(valid_mask)):
        length = int(valid_mask[row].sum().item())
        if length < 2:
            continue
        draw = torch.rand((), device=valid_mask.device, generator=generator)
        if draw < probability:
            turn = int(
                torch.randint(
                    1,
                    length,
                    (),
                    device=valid_mask.device,
                    generator=generator,
                ).item()
            )
            mask[row, turn] = True
    return mask


def symmetric_kl(first_logits, second_logits, mask):
    first_log = torch.log_softmax(first_logits[mask], dim=-1)
    second_log = torch.log_softmax(second_logits[mask], dim=-1)
    first = first_log.exp()
    second = second_log.exp()
    return 0.5 * (
        torch.nn.functional.kl_div(first_log, second, reduction="batchmean")
        + torch.nn.functional.kl_div(second_log, first, reduction="batchmean")
    )


class ZoneoutGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, zoneout_probability, state_noise=0.0):
        super().__init__()
        self.cell = torch.nn.GRUCell(input_size, hidden_size)
        stabilized.initialize_gru(self.cell)
        self.zoneout_probability = zoneout_probability
        self.state_noise = state_noise

    def forward(self, inputs, previous):
        recurrent_input = previous
        if self.training and self.state_noise > 0:
            recurrent_input = recurrent_input + self.state_noise * torch.randn_like(
                recurrent_input
            )
        candidate = self.cell(inputs, recurrent_input)
        if not self.training or self.zoneout_probability <= 0:
            return candidate
        if self.zoneout_probability >= 1:
            return previous
        preserve = torch.rand_like(previous).lt(self.zoneout_probability)
        return torch.where(preserve, previous, candidate)


class HeavilyRegularizedModel(stabilized.StabilizedRecurrentDialogueModel):
    def __init__(self, text_dimension, audio_dimension, number_of_classes, args):
        super().__init__(text_dimension, audio_dimension, number_of_classes, args)
        dialogue_input = self.dialogue_cell.input_size
        speaker_input = self.speaker_cell.input_size
        self.dialogue_cell = ZoneoutGRUCell(
            dialogue_input,
            self.dialogue_dimension,
            args.zoneout_probability,
            args.state_noise_std,
        )
        self.speaker_cell = ZoneoutGRUCell(
            speaker_input,
            self.speaker_dimension,
            args.zoneout_probability,
            args.state_noise_std,
        )


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = dict(model.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.lerp_(source[name], 1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(source_buffers[name])


def true_class_support(logits, labels, mask):
    targets = labels[mask].unsqueeze(1)
    return torch.log_softmax(logits[mask], dim=-1).gather(1, targets)


def label_smoothed_cross_entropy(logits, labels, mask, weights, smoothing):
    return torch.nn.functional.cross_entropy(
        logits[mask], labels[mask], weight=weights, label_smoothing=smoothing
    )


def calculate_loss(
    first,
    second,
    clean,
    reset,
    shuffled,
    zero_audio,
    batch,
    class_weights,
    args,
):
    mask, labels = batch["valid_mask"], batch["labels"]
    classification = 0.5 * (
        label_smoothed_cross_entropy(
            first["logits"], labels, mask, class_weights, args.label_smoothing
        )
        + label_smoothed_cross_entropy(
            second["logits"], labels, mask, class_weights, args.label_smoothing
        )
    )
    audio_classification = 0.5 * (
        label_smoothed_cross_entropy(
            first["audio_logits"],
            labels,
            mask,
            class_weights,
            args.label_smoothing,
        )
        + label_smoothed_cross_entropy(
            second["audio_logits"],
            labels,
            mask,
            class_weights,
            args.label_smoothing,
        )
    )
    clean_support = true_class_support(clean["logits"], labels, mask)
    shuffled_support = true_class_support(shuffled["logits"], labels, mask)
    reset_support = true_class_support(reset["logits"], labels, mask)
    zero_support = true_class_support(zero_audio["logits"], labels, mask)
    audio_ranking = torch.relu(
        args.counterfactual_margin - clean_support + shuffled_support
    ).mean()
    state_ranking = torch.relu(
        args.state_counterfactual_margin - clean_support + reset_support
    ).mean()
    zero_audio_preservation = torch.relu(zero_support - clean_support).mean()
    consistency = symmetric_kl(first["logits"], second["logits"], mask)
    negative_audio = (
        shuffled["audio_gate"][mask] * shuffled["audio_correction"][mask]
    ).square().mean()
    correction = (
        clean["context_correction"][mask].square().mean()
        + clean["audio_correction"][mask].square().mean()
    )
    context_gate = clean["context_gate"][mask]
    audio_gate = clean["audio_gate"][mask]
    gate_size = context_gate.abs().mean() + audio_gate.abs().mean()
    gate_variance = context_gate.var(unbiased=False) + audio_gate.var(unbiased=False)
    shuffled_gate = shuffled["audio_gate"][mask]
    gate_ranking = torch.relu(
        args.gate_ranking_margin - audio_gate + shuffled_gate
    ).mean()
    gate_ceiling = base.gate_ceiling_penalty(
        clean,
        mask,
        args.context_gate_soft_ceiling,
        args.audio_gate_soft_ceiling,
    )
    total = (
        classification
        + args.audio_loss_weight * audio_classification
        + args.counterfactual_weight * audio_ranking
        + args.state_counterfactual_weight * state_ranking
        + args.zero_audio_weight * zero_audio_preservation
        + args.consistency_weight * consistency
        + args.negative_residual_weight * negative_audio
        + args.correction_penalty_weight * correction
        + args.gate_l1_weight * gate_size
        - args.gate_variance_weight * gate_variance
        + args.gate_ranking_weight * gate_ranking
        + args.gate_penalty_weight * gate_ceiling
    )
    values = {
        "total": total,
        "classification": classification,
        "audio_classification": audio_classification,
        "audio_ranking": audio_ranking,
        "state_ranking": state_ranking,
        "zero_audio": zero_audio_preservation,
        "consistency": consistency,
        "negative_audio": negative_audio,
        "correction": correction,
        "gate_size": gate_size,
        "gate_variance": gate_variance,
        "gate_ranking": gate_ranking,
        "gate_ceiling": gate_ceiling,
    }
    return total, {
        name: float(value.detach().cpu()) for name, value in values.items()
    }


def feature_groups(record):
    with np.load(record["phase1_path"]) as cache:
        emotion = int(cache["emotion_summary"].size)
        egemaps = int(cache["egemaps"].size)
        prosody = int(cache["prosody_summary"].size)
    absolute_start = emotion
    relative_start = emotion + egemaps + prosody
    return [
        (0, emotion),
        (absolute_start, absolute_start + egemaps),
        (absolute_start + egemaps, relative_start),
        (relative_start, relative_start + egemaps),
        (relative_start + egemaps, relative_start + egemaps + prosody + 1),
    ]


def stochastic_batch(batch, conditions, groups, args):
    result = dict(batch)
    result["audio_features"] = corrupt_audio_features(
        batch["audio_features"],
        groups,
        args.feature_mask_probability,
        args.group_mask_probability,
        args.audio_noise_std,
        (args.audio_gain_min, args.audio_gain_max),
    )
    result = apply_modality_conditions(result, conditions)
    history_reset = random_history_reset_mask(
        batch["valid_mask"], args.history_truncation_probability
    )
    result["forced_reset_mask"] = (
        result["forced_reset_mask"] | history_reset
    )
    return result


def train_epoch(model, ema, loader, optimizer, class_weights, groups, args, device):
    model.train()
    rows = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = base.move_batch(batch, device)
        conditions = sample_modality_conditions(
            batch["valid_mask"].shape[0], device
        )
        first = model(stochastic_batch(batch, conditions, groups, args))
        second = model(stochastic_batch(batch, conditions, groups, args))
        model.eval()
        clean = model(batch)
        reset = model(batch, reset_each_turn=True)
        shuffled = model(base.shuffle_valid_audio(batch))
        zero_audio = model(batch, zero_audio=True)
        model.train()
        loss, components = calculate_loss(
            first,
            second,
            clean,
            reset,
            shuffled,
            zero_audio,
            batch,
            class_weights,
            args,
        )
        (loss / args.gradient_accumulation).backward()
        rows.append(components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--text-model", type=Path, default=root / "initial-testing/training-output-text/best-model")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/training-output-recurrent-dialogue-moderate-regularization")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dialogue-batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--text-projection-dimension", type=int, default=256)
    parser.add_argument("--audio-projection-dimension", type=int, default=128)
    parser.add_argument("--dialogue-state-dimension", type=int, default=128)
    parser.add_argument("--speaker-state-dimension", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dialogue-state-dropout", type=float, default=0.10)
    parser.add_argument("--speaker-state-dropout", type=float, default=0.15)
    parser.add_argument("--audio-dropout", type=float, default=0.10)
    parser.add_argument("--dialogue-reset-probability", type=float, default=0.02)
    parser.add_argument("--speaker-reset-probability", type=float, default=0.05)
    parser.add_argument("--zoneout-probability", type=float, default=0.05)
    parser.add_argument("--state-noise-std", type=float, default=0.0)
    parser.add_argument("--history-truncation-probability", type=float, default=0.15)
    parser.add_argument("--feature-mask-probability", type=float, default=0.0)
    parser.add_argument("--group-mask-probability", type=float, default=0.08)
    parser.add_argument("--audio-noise-std", type=float, default=0.02)
    parser.add_argument("--audio-gain-min", type=float, default=0.95)
    parser.add_argument("--audio-gain-max", type=float, default=1.05)
    parser.add_argument("--context-max-gate", type=float, default=0.22)
    parser.add_argument("--audio-max-gate", type=float, default=0.14)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--audio-loss-weight", type=float, default=0.3)
    parser.add_argument("--counterfactual-weight", type=float, default=0.55)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--state-counterfactual-weight", type=float, default=0.35)
    parser.add_argument("--state-counterfactual-margin", type=float, default=0.02)
    parser.add_argument("--zero-audio-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--negative-residual-weight", type=float, default=0.2)
    parser.add_argument("--correction-penalty-weight", type=float, default=0.01)
    parser.add_argument("--gate-l1-weight", type=float, default=0.005)
    parser.add_argument("--gate-variance-weight", type=float, default=0.0)
    parser.add_argument("--gate-ranking-weight", type=float, default=0.1)
    parser.add_argument("--gate-ranking-margin", type=float, default=0.005)
    parser.add_argument("--context-gate-soft-ceiling", type=float, default=0.15)
    parser.add_argument("--audio-gate-soft-ceiling", type=float, default=0.09)
    parser.add_argument("--gate-penalty-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--minimum-dev-state-margin", type=float, default=0.002)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    for name in (
        "epochs",
        "patience",
        "dialogue_batch_size",
        "encoder_batch_size",
        "gradient_accumulation",
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
    probabilities = (
        args.zoneout_probability,
        args.history_truncation_probability,
        args.feature_mask_probability,
        args.group_mask_probability,
        args.label_smoothing,
    )
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("regularization probabilities must be in [0, 1]")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if args.audio_gain_min <= 0 or args.audio_gain_max < args.audio_gain_min:
        raise ValueError("invalid audio gain range")


def train(args, records, device):
    sample = records["train"][0]
    groups = feature_groups(sample)
    model = HeavilyRegularizedModel(
        len(sample["text_embedding"]),
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        args,
    ).to(device)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    loaders = {
        split: base.make_loader(records[split], args, split == "train")
        for split in records
    }
    dev_shuffled = [
        base.make_loader(
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
    baseline = base.evaluate(ema.model, loaders["dev"], device)["text_metrics"]
    checkpoint = args.output_dir / "best_heavily_regularized.pt"
    print(
        f"Device: {device}; dialogues: {len(loaders['train'].dataset)}; "
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
            model, ema, loaders["train"], optimizer, weights, groups, args, device
        )
        matched = base.evaluate(ema.model, loaders["dev"], device)
        reset = base.evaluate(
            ema.model, loaders["dev"], device, reset_each_turn=True
        )
        zero = base.evaluate(ema.model, loaders["dev"], device, zero_audio=True)
        shuffled = [
            base.evaluate(ema.model, loader, device) for loader in dev_shuffled
        ]
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
            torch.save(ema.model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    ema.model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    test = base.evaluate(ema.model, loaders["test"], device)
    reset = base.evaluate(ema.model, loaders["test"], device, reset_each_turn=True)
    zero = base.evaluate(ema.model, loaders["test"], device, zero_audio=True)
    shuffled = []
    for seed in args.shuffle_seeds:
        loader = base.make_loader(
            records["test"], args, False,
            base.different_label_audio_mapping(records["test"], seed),
        )
        shuffled.append(base.evaluate(ema.model, loader, device))
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
    print(f"Regularized test weighted F1: {test['metrics']['weighted_f1']:.4f}")
    print(f"Regularized test macro F1: {test['metrics']['macro_f1']:.4f}")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0], std=normalization[1],
    )
    train(args, records, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
