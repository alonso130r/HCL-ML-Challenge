#!/usr/bin/env python3
"""Jointly tune upper BERT layers with the stabilized context-two fusion model."""

from __future__ import annotations

import argparse
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


def load_text_model(checkpoint, loader):
    return loader.from_pretrained(checkpoint, attn_implementation="eager")


def set_trainable_text_layers(model, final_layers):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = getattr(getattr(model, "bert", None), "encoder", None)
    layers = getattr(layers, "layer", None)
    if layers is None:
        raise ValueError("text model must expose bert.encoder.layer")
    if final_layers < 0 or final_layers > len(layers):
        raise ValueError(f"final text layers must be between 0 and {len(layers)}")
    selected = []
    if final_layers:
        for layer in layers[-final_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
                selected.append(parameter)
    return selected


def pool_current_tokens(hidden, mask):
    count = mask.sum(dim=1, keepdim=True)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
    return torch.where(count.eq(0), hidden[:, 0], pooled)


def build_optimizer(fusion_parameters, text_parameters, args):
    fusion_parameters = [
        parameter for parameter in fusion_parameters if parameter.requires_grad
    ]
    text_parameters = [
        parameter for parameter in text_parameters if parameter.requires_grad
    ]
    groups = [{"params": fusion_parameters, "lr": args.learning_rate}]
    if text_parameters:
        groups.append(
            {"params": text_parameters, "lr": args.text_learning_rate}
        )
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


class JointDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, records, audio_mapping=None):
        self.records = records
        self.dialogues = base.group_dialogues(records)
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
            "texts": [record["text"] for record in dialogue],
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


class JointCollator:
    def __init__(self, tokenizer, maximum_length):
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length

    def __call__(self, items):
        batch_size = len(items)
        turns = max(len(item["labels"]) for item in items)
        class_count = items[0]["text_logits"].shape[1]
        audio_dimension = items[0]["audio_features"].shape[1]
        result = {
            "text_logits": torch.zeros(batch_size, turns, class_count),
            "audio_features": torch.zeros(batch_size, turns, audio_dimension),
            "speaker_indices": torch.zeros(batch_size, turns, dtype=torch.long),
            "labels": torch.zeros(batch_size, turns, dtype=torch.long),
            "record_indices": torch.zeros(batch_size, turns, dtype=torch.long),
            "valid_mask": torch.zeros(batch_size, turns, dtype=torch.bool),
        }
        texts, rows, columns = [], [], []
        for row, item in enumerate(items):
            size = len(item["labels"])
            for key in (
                "text_logits", "audio_features", "speaker_indices", "labels",
                "record_indices",
            ):
                result[key][row, :size] = torch.as_tensor(item[key])
            result["valid_mask"][row, :size] = True
            texts.extend(item["texts"])
            rows.extend([row] * size)
            columns.extend(range(size))
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.maximum_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        current_mask = base.current_token_mask(encoded, offsets, texts)
        encoded.pop("special_tokens_mask")
        result.update(
            {
                "text_input_ids": encoded["input_ids"],
                "text_attention_mask": encoded["attention_mask"],
                "current_token_mask": current_mask,
                "flat_batch_indices": torch.tensor(rows, dtype=torch.long),
                "flat_turn_indices": torch.tensor(columns, dtype=torch.long),
            }
        )
        if "token_type_ids" in encoded:
            result["text_token_type_ids"] = encoded["token_type_ids"]
        return result


def make_loader(records, args, tokenizer, training, audio_mapping=None):
    return torch.utils.data.DataLoader(
        JointDialogueDataset(records, audio_mapping),
        batch_size=args.dialogue_batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=JointCollator(tokenizer, args.max_length),
        num_workers=0,
    )


class JointTextRecurrentModel(torch.nn.Module):
    def __init__(self, text_model, fusion_model):
        super().__init__()
        self.text_model = text_model
        self.fusion_model = fusion_model

    def encode_text(self, batch):
        inputs = {
            "input_ids": batch["text_input_ids"],
            "attention_mask": batch["text_attention_mask"],
        }
        if "text_token_type_ids" in batch:
            inputs["token_type_ids"] = batch["text_token_type_ids"]
        output = self.text_model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        pooled = pool_current_tokens(
            output.hidden_states[-1], batch["current_token_mask"]
        )
        shape = (*batch["valid_mask"].shape, pooled.shape[-1])
        embeddings = pooled.new_zeros(shape)
        embeddings[
            batch["flat_batch_indices"], batch["flat_turn_indices"]
        ] = pooled
        return embeddings

    def fuse(self, batch, text_embeddings, reset_each_turn=False, zero_audio=False):
        fusion_batch = {
            "text_embeddings": text_embeddings,
            "text_logits": batch["text_logits"],
            "audio_features": batch["audio_features"],
            "speaker_indices": batch["speaker_indices"],
            "valid_mask": batch["valid_mask"],
        }
        return self.fusion_model(
            fusion_batch,
            reset_each_turn=reset_each_turn,
            zero_audio=zero_audio,
        )

    def forward(self, batch, reset_each_turn=False, zero_audio=False):
        return self.fuse(
            batch,
            self.encode_text(batch),
            reset_each_turn=reset_each_turn,
            zero_audio=zero_audio,
        )


def train_epoch(model, loader, optimizer, class_weights, args, device):
    model.train()
    if not any(parameter.requires_grad for parameter in model.text_model.parameters()):
        model.text_model.eval()
    rows = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = base.move_batch(batch, device)
        stochastic_embeddings = model.encode_text(batch)
        matched = model.fuse(batch, stochastic_embeddings)
        model.eval()
        clean_embeddings = model.encode_text(batch)
        clean = model.fuse(batch, clean_embeddings)
        reset = model.fuse(batch, clean_embeddings, reset_each_turn=True)
        shuffled_batch = base.shuffle_valid_audio(batch)
        shuffled = model.fuse(shuffled_batch, clean_embeddings)
        model.train()
        if not any(parameter.requires_grad for parameter in model.text_model.parameters()):
            model.text_model.eval()
        loss, components = stabilized.calculate_loss(
            matched, clean, reset, shuffled, batch, class_weights, args
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
    parser.add_argument("--fusion-checkpoint", type=Path, default=root / "initial-testing/training-output-recurrent-dialogue-stabilized-c2-five/run-02-seed-43/best_recurrent_dialogue_stabilized.pt")
    parser.add_argument("--audio-cache-dir", type=Path, default=root / "initial-testing/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=root / "initial-testing/training-output-joint-text-recurrent")
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-text-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dialogue-batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--text-learning-rate", type=float, default=1e-5)
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
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    for name in (
        "head_epochs", "finetune_epochs", "finetune_text_layers", "patience",
        "dialogue_batch_size", "encoder_batch_size", "gradient_accumulation",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    for path_name in ("text_model", "fusion_checkpoint"):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name.replace('_', ' ')} not found: {path}")


def train(args, records, tokenizer, text_model, device):
    sample = records["train"][0]
    hidden_dimension = text_model.config.hidden_size
    fusion = stabilized.StabilizedRecurrentDialogueModel(
        hidden_dimension,
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        args,
    )
    fusion.load_state_dict(
        torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=True)
    )
    model = JointTextRecurrentModel(text_model, fusion).to(device)
    loaders = {
        split: make_loader(records[split], args, tokenizer, split == "train")
        for split in records
    }
    dev_shuffled = [
        make_loader(
            records["dev"], args, tokenizer, False,
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
    initial = base.evaluate(model, loaders["dev"], device)
    baseline = initial["text_metrics"]
    checkpoint = args.output_dir / "best_joint_text_recurrent.pt"
    initial_reset = base.evaluate(
        model, loaders["dev"], device, reset_each_turn=True
    )
    initial_zero = base.evaluate(model, loaders["dev"], device, zero_audio=True)
    initial_shuffled = [
        base.evaluate(model, loader, device) for loader in dev_shuffled
    ]
    initial_state_margin = (
        initial["metrics"]["weighted_f1"]
        - initial_reset["metrics"]["weighted_f1"]
    )
    initial_zero_margin = (
        initial["metrics"]["weighted_f1"]
        - initial_zero["metrics"]["weighted_f1"]
    )
    initial_audio_margin = initial["metrics"]["weighted_f1"] - max(
        item["metrics"]["weighted_f1"] for item in initial_shuffled
    )
    best_score = stabilized.checkpoint_score(
        initial["metrics"],
        baseline,
        initial_state_margin,
        args.minimum_dev_state_margin,
    )
    torch.save(model.state_dict(), checkpoint)
    history = [{
        "epoch": 0,
        "phase": "initial",
        "losses": None,
        "dev_matched": initial["metrics"],
        "dev_reset_state": initial_reset["metrics"],
        "dev_zero_audio": initial_zero["metrics"],
        "dev_shuffled": [item["metrics"] for item in initial_shuffled],
        "state_margin": initial_state_margin,
        "zero_audio_margin": initial_zero_margin,
        "audio_margin": initial_audio_margin,
        "checkpoint_eligible": bool(best_score[0]),
    }]
    best_epoch = 0
    global_epoch = 0
    started = time.perf_counter()
    print(
        f"Device: {device}; dialogues: {len(loaders['train'].dataset)}; "
        f"utterances: {len(records['train'])}; text dev weighted F1: "
        f"{baseline['weighted_f1']:.4f}"
    )
    phases = (
        ("head", args.head_epochs, 0),
        ("finetune", args.finetune_epochs, args.finetune_text_layers),
    )
    for phase, epochs, text_layers in phases:
        text_parameters = set_trainable_text_layers(model.text_model, text_layers)
        optimizer = build_optimizer(
            model.fusion_model.parameters(), text_parameters, args
        )
        stale = 0
        print(f"Starting {phase} phase for {epochs} epochs")
        for _ in range(epochs):
            global_epoch += 1
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
                "epoch": global_epoch, "phase": phase, "losses": losses,
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
                f"Epoch {global_epoch:02d} ({phase}): loss={losses['total']:.4f} "
                f"macro={matched['metrics']['macro_f1']:.4f} "
                f"weighted={matched['metrics']['weighted_f1']:.4f} "
                f"state={state_margin:+.4f} zero_audio={zero_margin:+.4f} "
                f"shuffle={audio_margin:+.4f} eligible={bool(score[0])}"
            )
            if score > best_score:
                best_score, best_epoch, stale = score, global_epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if phase == "finetune" and stale >= args.patience:
                    print(f"Early stopping {phase} phase after epoch {global_epoch}")
                    break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = base.evaluate(model, loaders["test"], device)
    reset = base.evaluate(model, loaders["test"], device, reset_each_turn=True)
    zero = base.evaluate(model, loaders["test"], device, zero_audio=True)
    shuffled = []
    for seed in args.shuffle_seeds:
        loader = make_loader(
            records["test"], args, tokenizer, False,
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
    print(f"Joint test weighted F1: {test['metrics']['weighted_f1']:.4f}")
    print(f"Joint test macro F1: {test['metrics']['macro_f1']:.4f}")
    print(f"State margin over reset: {state_margin:+.4f}")
    print(f"Audio margin over zero audio: {zero_margin:+.4f}")
    print(f"Audio margin over worst shuffle: {audio_margin:+.4f}")
    print(f"Outputs: {args.output_dir}")


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_arguments()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    records, normalization = stabilized.prepare_records(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = load_text_model(args.text_model, AutoModelForSequenceClassification)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "emotion_normalization.npz",
        mean=normalization[0], std=normalization[1],
    )
    train(args, records, tokenizer, text_model, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
