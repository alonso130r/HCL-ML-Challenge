#!/usr/bin/env python3
"""Train text-anchored, frame-level audio fusion on MELD."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_meld import EMOTION_LABELS, compute_metrics, select_device
from train_audio import frame_padding_mask, pad_audio, prepare_audio_split, unfreeze_top_layers
from train_text import build_examples, sqrt_class_weights
from cache_embeddings import load_split_rows


AUDIO_MODEL = "microsoft/wavlm-base-plus"
ACOUSTIC_DIMENSION = 88


def current_token_mask(token_type_ids, attention_mask, special_tokens_mask=None):
    attended = attention_mask.bool()
    if token_type_ids is not None and (token_type_ids == 1).any():
        return attended & token_type_ids.eq(1)
    if special_tokens_mask is None:
        return attended
    return attended & ~special_tokens_mask.bool()


def masked_mean(sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
    return (sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def disable_layerdrop(model: torch.nn.Module) -> None:
    """Keep hidden-state layer indices stable while fine-tuning WavLM."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "layerdrop"):
            config.layerdrop = 0.0


def disable_pretraining_masking(model: torch.nn.Module) -> None:
    """Disable WavLM SpecAugment, which is unsafe for very short MELD clips."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        if hasattr(config, "mask_time_prob"):
            config.mask_time_prob = 0.0
        if hasattr(config, "mask_feature_prob"):
            config.mask_feature_prob = 0.0


class LearnedLayerMixture(torch.nn.Module):
    def __init__(self, layer_indices: tuple[int, ...]):
        super().__init__()
        if not layer_indices:
            raise ValueError("at least one hidden layer is required")
        self.layer_indices = layer_indices
        self.weights = torch.nn.Parameter(torch.zeros(len(layer_indices)))

    def forward(self, hidden_states):
        if max(self.layer_indices) >= len(hidden_states):
            raise ValueError(
                f"requested hidden states {self.layer_indices}, but only "
                f"{len(hidden_states)} are available"
            )
        selected = torch.stack([hidden_states[index] for index in self.layer_indices])
        shape = (len(self.layer_indices),) + (1,) * (selected.ndim - 1)
        return (selected * torch.softmax(self.weights, dim=0).view(shape)).sum(dim=0)


class GatedResidualFusion(torch.nn.Module):
    def __init__(
        self,
        text_dimension: int,
        audio_dimension: int,
        acoustic_dimension: int,
        fusion_dimension: int,
        number_of_heads: int,
        number_of_classes: int,
        dropout: float,
        initial_gate_bias: float,
    ):
        super().__init__()
        self.text_projection = torch.nn.Linear(text_dimension, fusion_dimension)
        self.audio_projection = torch.nn.Linear(audio_dimension, fusion_dimension)
        self.cross_attention = torch.nn.MultiheadAttention(
            fusion_dimension, number_of_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = torch.nn.LayerNorm(fusion_dimension)
        self.acoustic_projection = torch.nn.Sequential(
            torch.nn.Linear(acoustic_dimension, fusion_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        joint_dimension = fusion_dimension * 3 + number_of_classes
        self.correction = torch.nn.Sequential(
            torch.nn.Linear(joint_dimension, fusion_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fusion_dimension, number_of_classes),
        )
        self.gate = torch.nn.Linear(joint_dimension, number_of_classes)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, initial_gate_bias)
        self.audio_classifier = torch.nn.Sequential(
            torch.nn.Linear(fusion_dimension * 2, fusion_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fusion_dimension, number_of_classes),
        )

    def forward(
        self,
        text_hidden,
        text_padding_mask,
        audio_hidden,
        audio_padding_mask,
        acoustic_features,
        text_logits,
    ):
        text = self.text_projection(text_hidden)
        audio = self.audio_projection(audio_hidden)
        attended, _ = self.cross_attention(
            text, audio, audio, key_padding_mask=audio_padding_mask, need_weights=False
        )
        attended = self.attention_norm(text + attended)
        text_summary = masked_mean(attended, text_padding_mask)
        audio_summary = masked_mean(audio, audio_padding_mask)
        acoustic_summary = self.acoustic_projection(acoustic_features)
        joint = torch.cat(
            (text_summary, audio_summary, acoustic_summary, text_logits.detach()), dim=-1
        )
        correction = self.correction(joint)
        gate = torch.sigmoid(self.gate(joint))
        audio_logits = self.audio_classifier(
            torch.cat((audio_summary, acoustic_summary), dim=-1)
        )
        return {
            "logits": text_logits + gate * correction,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "gate": gate,
            "correction": correction,
        }


class TextAudioModel(torch.nn.Module):
    def __init__(self, text_model, audio_model, args):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.layer_mixture = LearnedLayerMixture(tuple(args.audio_layers))
        self.fusion = GatedResidualFusion(
            text_dimension=text_model.config.hidden_size,
            audio_dimension=audio_model.config.hidden_size,
            acoustic_dimension=ACOUSTIC_DIMENSION,
            fusion_dimension=args.fusion_dimension,
            number_of_heads=args.attention_heads,
            number_of_classes=len(EMOTION_LABELS),
            dropout=args.dropout,
            initial_gate_bias=args.initial_gate_bias,
        )
        self.text_frozen = True
        self.audio_frozen = True

    def forward(self, batch):
        text_inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }

        def encode_text():
            return self.text_model(
                **text_inputs, output_hidden_states=True, return_dict=True
            )

        def encode_audio():
            return self.audio_model(
                input_values=batch["waveforms"],
                attention_mask=batch["audio_attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        if self.text_frozen:
            with torch.no_grad():
                text_output = encode_text()
        else:
            text_output = encode_text()
        if self.audio_frozen:
            with torch.no_grad():
                audio_output = encode_audio()
        else:
            audio_output = encode_audio()

        audio_hidden = self.layer_mixture(audio_output.hidden_states)
        audio_padding = frame_padding_mask(
            batch["audio_lengths"], audio_hidden.shape[1]
        )
        token_mask = batch["current_token_mask"]
        return self.fusion(
            text_output.hidden_states[-1],
            ~token_mask,
            audio_hidden,
            audio_padding,
            batch["acoustic_features"],
            text_output.logits,
        )

    def fusion_parameters(self):
        return list(self.layer_mixture.parameters()) + list(self.fusion.parameters())


def sanitize_acoustic_features(values: np.ndarray) -> np.ndarray:
    writable = np.array(values, dtype=np.float32, copy=True)
    return np.nan_to_num(writable, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def extract_egemaps(waveform: np.ndarray) -> np.ndarray:
    import opensmile

    smile = getattr(extract_egemaps, "_smile", None)
    if smile is None:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        extract_egemaps._smile = smile
    values = smile.process_signal(waveform, 16000).to_numpy(dtype=np.float32)[0]
    if values.shape != (ACOUSTIC_DIMENSION,):
        raise ValueError(f"expected {ACOUSTIC_DIMENSION} eGeMAPS values, got {values.shape}")
    return sanitize_acoustic_features(values)


def attach_acoustic_features(records, split_dir: Path, rebuild: bool):
    descriptor_dir = split_dir / "egemaps"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, record in enumerate(records, start=1):
        target = descriptor_dir / (
            f"dia{record['dialogue_id']}_utt{record['utterance_id']}.npy"
        )
        if rebuild or not target.exists():
            np.save(target, extract_egemaps(np.load(record["audio_path"])))
        prepared.append(record | {"acoustic_path": str(target.resolve())})
        if index % 500 == 0:
            print(f"  cached {index}/{len(records)} eGeMAPS vectors", flush=True)
    return prepared


def combine_records(rows, audio_records, context_window):
    examples = build_examples(rows, context_window)
    by_key = {
        (item["dialogue_id"], item["utterance_id"]): item for item in examples
    }
    combined = []
    for audio in audio_records:
        key = (audio["dialogue_id"], audio["utterance_id"])
        if key in by_key:
            combined.append(by_key[key] | audio)
    return combined


def fit_acoustic_normalizer(records):
    matrix = np.stack([np.load(item["acoustic_path"]) for item in records])
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class TextAudioDataset(torch.utils.data.Dataset):
    def __init__(self, records, acoustic_mean, acoustic_std, audio_indices=None):
        self.records = records
        self.acoustic_mean = acoustic_mean
        self.acoustic_std = acoustic_std
        self.audio_indices = audio_indices or list(range(len(records)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        text_record = self.records[index]
        audio_record = self.records[self.audio_indices[index]]
        acoustic = np.load(audio_record["acoustic_path"]).astype(np.float32)
        return {
            "context": text_record["text"].split("Current:\n", 1)[0].removesuffix("\n"),
            "current": f"[{text_record['speaker']}] {text_record['utterance']}",
            "waveform": np.load(audio_record["audio_path"]).astype(np.float32),
            "acoustic": (acoustic - self.acoustic_mean) / self.acoustic_std,
            "label": EMOTION_LABELS.index(text_record["label"]),
            "index": index,
        }


def make_collator(tokenizer, max_length):
    def collate(items):
        prefixes = [
            item["context"] + ("\n" if item["context"] else "") + "Current:\n"
            for item in items
        ]
        texts = [prefix + item["current"] for prefix, item in zip(prefixes, items)]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        starts = torch.tensor([len(prefix) for prefix in prefixes]).unsqueeze(1)
        token_mask = (
            offsets[:, :, 0].ge(starts)
            & offsets[:, :, 1].gt(offsets[:, :, 0])
            & encoded["attention_mask"].bool()
            & ~encoded["special_tokens_mask"].bool()
        )
        encoded["current_token_mask"] = token_mask
        waveforms = []
        for item in items:
            waveform = item["waveform"]
            waveform = (waveform - waveform.mean()) / max(float(waveform.std()), 1e-5)
            waveforms.append(waveform.astype(np.float32, copy=False))
        padded, lengths = pad_audio(waveforms)
        sample_mask = ~frame_padding_mask(lengths, padded.shape[1])
        encoded.update(
            {
                "waveforms": padded,
                "audio_lengths": lengths,
                "audio_attention_mask": sample_mask.long(),
                "acoustic_features": torch.from_numpy(
                    np.stack([item["acoustic"] for item in items])
                ),
                "labels": torch.tensor([item["label"] for item in items]),
                "indices": torch.tensor([item["index"] for item in items]),
            }
        )
        return encoded

    return collate


def make_loader(records, tokenizer, acoustic_stats, args, training, audio_indices=None):
    return torch.utils.data.DataLoader(
        TextAudioDataset(records, *acoustic_stats, audio_indices=audio_indices),
        batch_size=args.batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collator(tokenizer, args.max_length),
    )


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def calculate_loss(output, labels, class_weights, args):
    fused_loss = torch.nn.functional.cross_entropy(
        output["logits"], labels, weight=class_weights
    )
    audio_loss = torch.nn.functional.cross_entropy(
        output["audio_logits"], labels, weight=class_weights
    )
    temperature = args.distillation_temperature
    distillation = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(output["audio_logits"] / temperature, dim=-1),
        torch.softmax(output["text_logits"].detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    gate_penalty = output["gate"].mean()
    total = (
        fused_loss
        + args.audio_loss_weight * audio_loss
        + args.distillation_weight * distillation
        + args.gate_penalty_weight * gate_penalty
    )
    return total, {
        "fused": float(fused_loss.detach().cpu()),
        "audio": float(audio_loss.detach().cpu()),
        "distillation": float(distillation.detach().cpu()),
        "gate": float(gate_penalty.detach().cpu()),
    }


def evaluate(model, loader, device):
    actual, fused, text, confidence, indices, gates = [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            probabilities = torch.softmax(output["logits"], dim=-1)
            batch_confidence, prediction = probabilities.max(dim=-1)
            actual.extend(batch["labels"].cpu().tolist())
            fused.extend(prediction.cpu().tolist())
            text.extend(output["text_logits"].argmax(dim=-1).cpu().tolist())
            confidence.extend(batch_confidence.cpu().tolist())
            indices.extend(batch["indices"].cpu().tolist())
            gates.append(output["gate"].float().cpu().numpy())
    names = lambda values: [EMOTION_LABELS[index] for index in values]
    return {
        "fused_metrics": compute_metrics(names(actual), names(fused)),
        "text_metrics": compute_metrics(names(actual), names(text)),
        "predicted": names(fused),
        "confidence": confidence,
        "indices": indices,
        "mean_gate_by_class": dict(
            zip(EMOTION_LABELS, np.concatenate(gates).mean(axis=0).tolist())
        ),
    }


def train_epoch(model, loader, optimizer, class_weights, args, device, train_audio):
    model.train()
    model.text_model.eval()
    if not train_audio:
        model.audio_model.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = []
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        output = model(batch)
        loss, components = calculate_loss(output, batch["labels"], class_weights, args)
        (loss / args.gradient_accumulation).backward()
        totals.append({"total": float(loss.detach().cpu())} | components)
        if step % args.gradient_accumulation == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return {
        key: sum(item[key] for item in totals) / len(totals) for key in totals[0]
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
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=root / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--text-model", type=Path, default=root / "models/text-emotion")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "research/experiments/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "research/experiments/training-output-text-audio")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-audio-layers", type=int, default=2)
    parser.add_argument("--audio-layers", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--audio-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fusion-dimension", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--initial-gate-bias", type=float, default=-4.0)
    parser.add_argument("--audio-loss-weight", type=float, default=0.15)
    parser.add_argument("--distillation-weight", type=float, default=0.1)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--gate-penalty-weight", type=float, default=0.01)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-audio-cache", action="store_true")
    parser.add_argument("--rebuild-acoustic-cache", action="store_true")
    return parser.parse_args()


def validate_arguments(args):
    if args.head_epochs < 0 or args.finetune_epochs < 0:
        raise ValueError("head and finetune epochs must be nonnegative")
    if args.head_epochs + args.finetune_epochs < 1:
        raise ValueError("at least one training epoch is required")
    if args.fusion_dimension % args.attention_heads:
        raise ValueError("fusion dimension must be divisible by attention heads")
    if min(args.batch_size, args.gradient_accumulation, args.patience) < 1:
        raise ValueError("batch size, gradient accumulation, and patience must be positive")
    if not args.text_model.exists():
        raise FileNotFoundError(f"text checkpoint not found: {args.text_model}")


def main():
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model)
    audio_model = AutoModel.from_pretrained(AUDIO_MODEL)
    disable_layerdrop(audio_model)
    disable_pretraining_masking(audio_model)
    for parameter in text_model.parameters():
        parameter.requires_grad = False
    unfreeze_top_layers(audio_model, 0)
    model = TextAudioModel(text_model, audio_model, args).to(device)

    records = {}
    for split in ("train", "dev", "test"):
        rows = load_split_rows(args.raw_archive, split)
        if args.max_samples_per_split is not None:
            rows = rows[:args.max_samples_per_split]
        audio = prepare_audio_split(
            args.raw_archive, split, args.audio_cache_dir,
            args.max_samples_per_split, args.rebuild_audio_cache,
        )
        audio = attach_acoustic_features(
            audio, args.audio_cache_dir / split, args.rebuild_acoustic_cache
        )
        records[split] = combine_records(rows, audio, args.context_window)

    acoustic_stats = fit_acoustic_normalizer(records["train"])
    loaders = {
        split: make_loader(
            records[split], tokenizer, acoustic_stats, args, split == "train"
        )
        for split in records
    }
    labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in records["train"]]
    )
    class_weights = torch.from_numpy(
        sqrt_class_weights(labels, len(EMOTION_LABELS))
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_text_audio.pt"
    if args.head_epochs == 0 and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"cannot skip head training without an existing checkpoint: {checkpoint_path}"
        )
    baseline = evaluate(model, loaders["dev"], device)["text_metrics"]
    print(
        f"Device: {device}; samples: {len(records['train'])}; "
        f"text dev weighted F1: {baseline['weighted_f1']:.4f}"
    )

    history = []
    best_score = (-1, -1.0)
    best_epoch = 0
    epoch = 0
    started = time.perf_counter()
    for phase, phase_epochs in (("head", args.head_epochs), ("finetune", args.finetune_epochs)):
        if phase_epochs == 0:
            continue
        if phase == "head":
            optimizer = torch.optim.AdamW(
                model.fusion_parameters(), lr=args.head_learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            unfreeze_top_layers(audio_model, args.unfreeze_audio_layers)
            model.audio_frozen = False
            audio_parameters = [p for p in audio_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.fusion_parameters(), "lr": args.head_learning_rate},
                    {"params": audio_parameters, "lr": args.audio_learning_rate},
                ],
                weight_decay=args.weight_decay,
            )
        stale = 0
        print(f"Starting {phase} phase for {phase_epochs} epochs")
        for _ in range(phase_epochs):
            epoch += 1
            losses = train_epoch(
                model, loaders["train"], optimizer, class_weights, args,
                device, phase == "finetune",
            )
            dev = evaluate(model, loaders["dev"], device)
            metrics = dev["fused_metrics"]
            eligible = int(metrics["weighted_f1"] >= baseline["weighted_f1"])
            score = (eligible, metrics["macro_f1"])
            row = {
                "epoch": epoch,
                "phase": phase,
                "losses": losses,
                "dev_fused": metrics,
                "dev_text": dev["text_metrics"],
                "mean_gate_by_class": dev["mean_gate_by_class"],
            }
            history.append(row)
            print(
                f"Epoch {epoch:02d} ({phase}): loss={losses['total']:.4f} "
                f"dev_macro_f1={metrics['macro_f1']:.4f} "
                f"dev_weighted_f1={metrics['weighted_f1']:.4f}"
            )
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping {phase} after epoch {epoch}")
                    break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test = evaluate(model, loaders["test"], device)
    shuffled = list(range(len(records["test"])))
    random.Random(args.seed + 1).shuffle(shuffled)
    shuffled_loader = make_loader(
        records["test"], tokenizer, acoustic_stats, args, False, shuffled
    )
    shuffled_result = evaluate(model, shuffled_loader, device)
    write_predictions(args.output_dir / "test_predictions.csv", records["test"], test)
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
        "best_epoch": best_epoch,
        "development_text_baseline": baseline,
        "test_text": test["text_metrics"],
        "test_fused": test["fused_metrics"],
        "test_shuffled_audio": shuffled_result["fused_metrics"],
        "mean_test_gate_by_class": test["mean_gate_by_class"],
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Text test weighted F1: {test['text_metrics']['weighted_f1']:.4f}")
    print(f"Fused test weighted F1: {test['fused_metrics']['weighted_f1']:.4f}")
    print(f"Fused test macro F1: {test['fused_metrics']['macro_f1']:.4f}")
    print(
        f"Shuffled-audio weighted F1: "
        f"{shuffled_result['fused_metrics']['weighted_f1']:.4f}"
    )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
