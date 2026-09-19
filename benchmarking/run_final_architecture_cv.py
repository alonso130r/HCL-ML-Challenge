#!/usr/bin/env python3
"""Run leakage-safe dialogue-level CV for the final MELD architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
INITIAL_TESTING = ROOT / "initial-testing"
sys.path.insert(0, str(INITIAL_TESTING))

import train_recurrent_dialogue as recurrent  # noqa: E402
import train_recurrent_dialogue_stabilized as stabilized  # noqa: E402
from cache_embeddings import load_split_rows  # noqa: E402
from evaluate_meld import EMOTION_LABELS, select_device  # noqa: E402
from train_text import (  # noqa: E402
    MeldTextDataset,
    evaluate as evaluate_text,
    make_collator,
    sqrt_class_weights,
)
from train_text_audio_phase2 import (  # noqa: E402
    attach_speaker_relative_acoustics,
    combine_records,
)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def dialogue_summaries(records):
    summaries = {}
    for record in records:
        dialogue = str(record["dialogue_id"])
        if dialogue not in summaries:
            summaries[dialogue] = np.zeros(len(EMOTION_LABELS) + 1, dtype=np.float64)
        summaries[dialogue][EMOTION_LABELS.index(record["label"])] += 1
        summaries[dialogue][-1] += 1
    return summaries


def assign_dialogue_folds(records, folds, seed):
    if folds < 2:
        raise ValueError("at least two folds are required")
    summaries = dialogue_summaries(records)
    if len(summaries) < folds:
        raise ValueError("fewer dialogues than folds")
    rng = random.Random(seed)
    dialogues = list(summaries)
    rng.shuffle(dialogues)
    totals = np.stack(list(summaries.values())).sum(axis=0)
    scale = np.where(totals > 0, totals / folds, 1.0)
    dialogues.sort(
        key=lambda key: float(np.max(summaries[key] / scale)), reverse=True
    )
    fold_totals = np.zeros((folds, len(EMOTION_LABELS) + 1), dtype=np.float64)
    assignments = {}
    for position, dialogue in enumerate(dialogues):
        if position < folds:
            selected = position
        else:
            candidate = summaries[dialogue]
            scores = []
            for fold in range(folds):
                proposed = fold_totals.copy()
                proposed[fold] += candidate
                normalized = proposed / scale
                imbalance = normalized.var(axis=0).mean()
                size_penalty = normalized[:, -1].var()
                scores.append(float(imbalance + 0.5 * size_penalty))
            selected = min(range(folds), key=lambda fold: (scores[fold], fold))
        assignments[dialogue] = selected
        fold_totals[selected] += summaries[dialogue]
    return assignments


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    denominator = support + predicted
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = support.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total) if total else 0.0,
        "per_class_f1": {
            label: float(f1[index]) for index, label in enumerate(EMOTION_LABELS[: len(f1)])
        },
        "support": {
            label: int(support[index]) for index, label in enumerate(EMOTION_LABELS[: len(f1)])
        },
    }


def confusion_matrix(labels, predictions, classes=None):
    classes = classes or len(EMOTION_LABELS)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels), np.asarray(predictions)), 1)
    return matrix


def fit_temperature(logits, labels):
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(
            logits_tensor / temperature, labels_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=1, keepdims=True)


def calibration_metrics(probabilities, labels, bins=15):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    selected = probabilities[np.arange(len(labels)), labels].clip(1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1])[labels]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "negative_log_likelihood": float(-np.log(selected).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error": float(ece),
    }


def holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def load_all_records(raw_archive, audio_cache_dir, context_window):
    records = []
    dialogue_number = 0
    for source_split in ("train", "dev", "test"):
        rows = load_split_rows(raw_archive, source_split)
        split_records = combine_records(
            rows, context_window, audio_cache_dir, source_split
        )
        mapping = {}
        for record in split_records:
            original = str(record["dialogue_id"])
            if original not in mapping:
                mapping[original] = str(dialogue_number)
                dialogue_number += 1
            record = dict(record)
            record["source_split"] = source_split
            record["source_dialogue_id"] = original
            record["dialogue_id"] = mapping[original]
            records.append(record)
    return records


def select_records(records, dialogue_ids):
    selected = set(dialogue_ids)
    return [dict(record) for record in records if str(record["dialogue_id"]) in selected]


def text_loader(examples, tokenizer, args, training):
    return torch.utils.data.DataLoader(
        MeldTextDataset(examples),
        batch_size=args.text_batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collator(tokenizer, args.max_length),
        num_workers=0,
    )


def train_text_fold(train_records, dev_records, output_dir, args, device):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    checkpoint = output_dir / "best-model"
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and checkpoint.exists() and not args.retrain:
        return checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.text_model,
        num_labels=len(EMOTION_LABELS),
        id2label={index: label for index, label in enumerate(EMOTION_LABELS)},
        label2id={label: index for index, label in enumerate(EMOTION_LABELS)},
        attn_implementation="eager",
    ).to(device)
    labels = np.array(
        [EMOTION_LABELS.index(record["label"]) for record in train_records]
    )
    weights = torch.from_numpy(sqrt_class_weights(labels, len(EMOTION_LABELS))).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    train = text_loader(train_records, tokenizer, args, True)
    dev = text_loader(dev_records, tokenizer, args, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.text_learning_rate, weight_decay=args.text_weight_decay
    )
    updates = math.ceil(len(train) / args.text_gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(updates * args.text_epochs * args.text_warmup_ratio),
        num_training_steps=updates * args.text_epochs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], -1.0, 0
    for epoch in range(1, args.text_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch.pop("indices")
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss = loss_function(model(**inputs).logits, labels_batch)
            (loss / args.text_gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % args.text_gradient_accumulation == 0 or step == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, _, _, _ = evaluate_text(model, dev, device)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "dev": metrics,
        })
        print(
            f"    text epoch {epoch}: loss={np.mean(losses):.4f} "
            f"macro={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        else:
            stale += 1
            if stale >= args.text_patience:
                break
    metrics_path.write_text(
        json.dumps({"best_dev_macro_f1": best, "history": history}, indent=2),
        encoding="utf-8",
    )
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return checkpoint


def cache_fold_text(records_by_split, checkpoint, run_dir, args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    ).to(device)
    cache_args = SimpleNamespace(
        rebuild_text_cache=args.rebuild_text_cache or args.retrain,
        encoder_batch_size=args.encoder_batch_size,
        max_length=args.max_length,
    )
    for split, records in records_by_split.items():
        recurrent.cache_text_features(
            records,
            tokenizer,
            model,
            device,
            run_dir / "text-cache" / split,
            cache_args,
        )
    del model, tokenizer
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def fusion_arguments(args, seed):
    return SimpleNamespace(
        seed=seed,
        runs=len(args.seeds),
        epochs=args.fusion_epochs,
        dialogue_batch_size=args.dialogue_batch_size,
        gradient_accumulation=args.fusion_gradient_accumulation,
        learning_rate=args.fusion_learning_rate,
        weight_decay=args.fusion_weight_decay,
        text_projection_dimension=256,
        audio_projection_dimension=128,
        dialogue_state_dimension=128,
        speaker_state_dimension=64,
        dropout=0.2,
        dialogue_state_dropout=0.05,
        speaker_state_dropout=0.10,
        audio_dropout=0.10,
        dialogue_reset_probability=0.01,
        speaker_reset_probability=0.03,
        context_max_gate=0.25,
        audio_max_gate=0.15,
        initial_gate_bias=-2.0,
        audio_loss_weight=0.3,
        counterfactual_weight=0.5,
        counterfactual_margin=0.1,
        state_counterfactual_weight=0.3,
        state_counterfactual_margin=0.02,
        negative_residual_weight=0.2,
        correction_penalty_weight=0.01,
        context_gate_soft_ceiling=0.18,
        audio_gate_soft_ceiling=0.10,
        gate_penalty_weight=0.2,
        patience=args.fusion_patience,
        minimum_dev_state_margin=0.002,
        shuffle_seeds=[143, 144, 145, 146, 147],
        context_window=args.context_window,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )


def collect_logits(model, loader, device, reset=False, zero_audio=False):
    labels, indices, logits, text_logits = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = recurrent.move_batch(batch, device)
            output = model(batch, reset_each_turn=reset, zero_audio=zero_audio)
            mask = batch["valid_mask"]
            labels.append(batch["labels"][mask].cpu().numpy())
            indices.append(batch["record_indices"][mask].cpu().numpy())
            logits.append(output["logits"][mask].cpu().numpy())
            text_logits.append(output["text_logits"][mask].cpu().numpy())
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    order = np.argsort(indices)
    return {
        "labels": labels[order],
        "indices": indices[order],
        "logits": np.concatenate(logits)[order],
        "text_logits": np.concatenate(text_logits)[order],
    }


def evaluate_run(records, fusion_args, checkpoint, temperature, device):
    sample = records["train"][0]
    model = stabilized.StabilizedRecurrentDialogueModel(
        len(sample["text_embedding"]),
        len(sample["audio_features"]),
        len(EMOTION_LABELS),
        fusion_args,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loader = recurrent.make_loader(records["test"], fusion_args, False)
    matched = collect_logits(model, loader, device)
    reset = collect_logits(model, loader, device, reset=True)
    zero = collect_logits(model, loader, device, zero_audio=True)
    shuffled_candidates = []
    for shuffle_seed in fusion_args.shuffle_seeds:
        shuffled_loader = recurrent.make_loader(
            records["test"],
            fusion_args,
            False,
            recurrent.different_label_audio_mapping(records["test"], shuffle_seed),
        )
        shuffled_candidates.append(collect_logits(model, shuffled_loader, device))
    shuffled = max(
        shuffled_candidates,
        key=lambda item: metrics_from_confusion(
            confusion_matrix(item["labels"], item["logits"].argmax(axis=1))
        )["weighted_f1"],
    )
    probabilities = softmax(matched["logits"], temperature)
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return matched | {
        "probabilities": probabilities,
        "temperature": temperature,
        "reset_logits": reset["logits"],
        "zero_audio_logits": zero["logits"],
        "shuffled_logits": shuffled["logits"],
    }


def train_one_outer_run(records, outer_assignments, outer_fold, seed, args, device):
    run_dir = args.output_dir / f"fold-{outer_fold + 1:02d}" / f"seed-{seed}"
    prediction_path = run_dir / "outer_predictions.npz"
    if prediction_path.exists() and not args.retrain:
        print(f"Reusing fold {outer_fold + 1}, seed {seed}", flush=True)
        return prediction_path
    outer_test_dialogues = {
        dialogue for dialogue, fold in outer_assignments.items() if fold == outer_fold
    }
    outer_train_dialogues = set(outer_assignments) - outer_test_dialogues
    outer_train_records = select_records(records, outer_train_dialogues)
    inner_assignments = assign_dialogue_folds(
        outer_train_records, folds=8, seed=args.fold_seed + outer_fold
    )
    dev_dialogues = {
        dialogue for dialogue, fold in inner_assignments.items() if fold == 0
    }
    train_dialogues = outer_train_dialogues - dev_dialogues
    split_records = {
        "train": select_records(records, train_dialogues),
        "dev": select_records(records, dev_dialogues),
        "test": select_records(records, outer_test_dialogues),
    }
    run_args = SimpleNamespace(**vars(args))
    run_args.seed = seed
    text_checkpoint = train_text_fold(
        split_records["train"], split_records["dev"], run_dir / "text", run_args, device
    )
    cache_fold_text(split_records, text_checkpoint, run_dir, run_args, device)
    attach_speaker_relative_acoustics(split_records)
    recurrent.load_cached_features(split_records)
    fusion_args = fusion_arguments(run_args, seed)
    fusion_dir = run_dir / "fusion"
    fusion_args.output_dir = fusion_dir
    checkpoint = fusion_dir / "best_recurrent_dialogue_stabilized.pt"
    if checkpoint.exists() and (fusion_dir / "metrics.json").exists() and not args.retrain:
        print(f"    reusing fusion checkpoint for fold {outer_fold + 1}, seed {seed}")
    else:
        stabilized.train(
            fusion_args,
            split_records,
            device,
            list(args.seeds).index(seed) + 1,
        )
    model = stabilized.StabilizedRecurrentDialogueModel(
        len(split_records["train"][0]["text_embedding"]),
        len(split_records["train"][0]["audio_features"]),
        len(EMOTION_LABELS),
        fusion_args,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    dev_loader = recurrent.make_loader(split_records["dev"], fusion_args, False)
    dev_result = collect_logits(model, dev_loader, device)
    temperature = fit_temperature(dev_result["logits"], dev_result["labels"])
    del model
    result = evaluate_run(
        split_records, fusion_args, checkpoint, temperature, device
    )
    ordered = sorted(
        split_records["test"], key=lambda record: record["record_index"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        labels=result["labels"],
        logits=result["logits"],
        probabilities=result["probabilities"],
        text_logits=result["text_logits"],
        reset_logits=result["reset_logits"],
        zero_audio_logits=result["zero_audio_logits"],
        shuffled_logits=result["shuffled_logits"],
        temperature=np.array(result["temperature"]),
        dialogue_ids=np.array([record["dialogue_id"] for record in ordered]),
        source_splits=np.array([record["source_split"] for record in ordered]),
        source_dialogue_ids=np.array([record["source_dialogue_id"] for record in ordered]),
        utterance_ids=np.array([record["utterance_id"] for record in ordered]),
        seed=np.array(seed),
        outer_fold=np.array(outer_fold),
    )
    return prediction_path


CONDITIONS = {
    "matched": "logits",
    "text": "text_logits",
    "zero_audio": "zero_audio_logits",
    "shuffled_audio": "shuffled_logits",
    "reset_state": "reset_logits",
}


def load_prediction_files(paths):
    runs = []
    for path in paths:
        with np.load(path) as data:
            runs.append({key: data[key] for key in data.files})
    return runs


def pooled_by_seed(runs):
    result = {}
    for seed in sorted({int(run["seed"]) for run in runs}):
        selected = sorted(
            [run for run in runs if int(run["seed"]) == seed],
            key=lambda run: int(run["outer_fold"]),
        )
        result[seed] = {
            key: np.concatenate([run[key] for run in selected])
            for key in (
                "labels", "logits", "probabilities", "text_logits", "reset_logits",
                "zero_audio_logits", "shuffled_logits", "dialogue_ids",
                "source_splits", "source_dialogue_ids", "utterance_ids",
            )
        }
    return result


def run_metrics(run):
    labels = run["labels"].astype(np.int64)
    report = {}
    for name, key in CONDITIONS.items():
        predictions = run[key].argmax(axis=1)
        report[name] = metrics_from_confusion(confusion_matrix(labels, predictions))
    report["calibration"] = calibration_metrics(run["probabilities"], labels)
    report["margins"] = {
        "over_text": report["matched"]["weighted_f1"] - report["text"]["weighted_f1"],
        "zero_audio": report["matched"]["weighted_f1"] - report["zero_audio"]["weighted_f1"],
        "shuffled_audio": report["matched"]["weighted_f1"] - report["shuffled_audio"]["weighted_f1"],
        "state": report["matched"]["weighted_f1"] - report["reset_state"]["weighted_f1"],
    }
    return report


def dialogue_confusions(run, condition):
    labels = run["labels"].astype(np.int64)
    predictions = run[CONDITIONS[condition]].argmax(axis=1)
    dialogues = run["dialogue_ids"].astype(str)
    keys = sorted(set(dialogues), key=int)
    matrices = []
    for key in keys:
        mask = dialogues == key
        matrices.append(confusion_matrix(labels[mask], predictions[mask]))
    return keys, np.stack(matrices)


def batch_weighted_f1(confusions):
    support = confusions.sum(axis=2)
    predicted = confusions.sum(axis=1)
    true_positive = np.diagonal(confusions, axis1=1, axis2=2)
    denominator = support + predicted
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )
    total = support.sum(axis=1)
    return np.divide(
        (f1 * support).sum(axis=1),
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


def bootstrap_margins(seed_runs, iterations, seed):
    rng = np.random.default_rng(seed)
    seed_values = list(seed_runs.values())
    group_keys, _ = dialogue_confusions(seed_values[0], "matched")
    groups = len(group_keys)
    condition_pairs = {
        "over_text": "text",
        "zero_audio": "zero_audio",
        "shuffled_audio": "shuffled_audio",
        "state": "reset_state",
    }
    samples = {name: [] for name in condition_pairs}
    samples["matched_weighted_f1"] = []
    chunk_size = 100
    for start in range(0, iterations, chunk_size):
        size = min(chunk_size, iterations - start)
        counts = rng.multinomial(groups, np.full(groups, 1.0 / groups), size=size)
        matched_by_seed = []
        controls_by_name = {name: [] for name in condition_pairs}
        for run in seed_values:
            keys, matched = dialogue_confusions(run, "matched")
            if keys != group_keys:
                raise ValueError("dialogue ordering differs across seeds")
            matched_confusions = (counts @ matched.reshape(groups, -1)).reshape(
                size, len(EMOTION_LABELS), len(EMOTION_LABELS)
            )
            matched_scores = batch_weighted_f1(matched_confusions)
            matched_by_seed.append(matched_scores)
            for name, condition in condition_pairs.items():
                _, control = dialogue_confusions(run, condition)
                control_scores = batch_weighted_f1(
                    (counts @ control.reshape(groups, -1)).reshape(
                        size, len(EMOTION_LABELS), len(EMOTION_LABELS)
                    )
                )
                controls_by_name[name].append(control_scores)
        matched_mean = np.mean(matched_by_seed, axis=0)
        samples["matched_weighted_f1"].extend(matched_mean.tolist())
        for name in condition_pairs:
            control_mean = np.mean(controls_by_name[name], axis=0)
            samples[name].extend((matched_mean - control_mean).tolist())
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        }
        for name, values in samples.items()
    }


def randomization_p_value(seed_runs, control, iterations, seed):
    rng = np.random.default_rng(seed)
    seed_values = list(seed_runs.values())
    keys, first_matched = dialogue_confusions(seed_values[0], "matched")
    groups = len(keys)
    observed = []
    pairs = []
    for run in seed_values:
        run_keys, matched = dialogue_confusions(run, "matched")
        _, counterfactual = dialogue_confusions(run, control)
        if run_keys != keys:
            raise ValueError("dialogue ordering differs across seeds")
        matched_score = metrics_from_confusion(matched.sum(axis=0))["weighted_f1"]
        control_score = metrics_from_confusion(counterfactual.sum(axis=0))["weighted_f1"]
        observed.append(matched_score - control_score)
        pairs.append((matched, counterfactual))
    observed_mean = float(np.mean(observed))
    exceed = 0
    for _ in range(iterations):
        differences = []
        swap = rng.integers(0, 2, size=groups, dtype=np.int8).astype(bool)
        for matched, counterfactual in pairs:
            first = np.where(swap[:, None, None], counterfactual, matched).sum(axis=0)
            second = np.where(swap[:, None, None], matched, counterfactual).sum(axis=0)
            differences.append(
                metrics_from_confusion(first)["weighted_f1"]
                - metrics_from_confusion(second)["weighted_f1"]
            )
        if np.mean(differences) >= observed_mean:
            exceed += 1
    return (exceed + 1) / (iterations + 1)


def aggregate(paths, args):
    runs = load_prediction_files(paths)
    seed_runs = pooled_by_seed(runs)
    seed_reports = {str(seed): run_metrics(run) for seed, run in seed_runs.items()}
    fields = {
        "weighted_f1": [report["matched"]["weighted_f1"] for report in seed_reports.values()],
        "macro_f1": [report["matched"]["macro_f1"] for report in seed_reports.values()],
        "over_text": [report["margins"]["over_text"] for report in seed_reports.values()],
        "zero_audio": [report["margins"]["zero_audio"] for report in seed_reports.values()],
        "shuffled_audio": [report["margins"]["shuffled_audio"] for report in seed_reports.values()],
        "state": [report["margins"]["state"] for report in seed_reports.values()],
    }
    summary = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in fields.items()
    }
    bootstrap = bootstrap_margins(
        seed_runs, args.bootstrap_iterations, args.analysis_seed
    )
    controls = ["text", "zero_audio", "shuffled_audio", "reset_state"]
    raw_p = [
        randomization_p_value(
            seed_runs, control, args.randomization_iterations,
            args.analysis_seed + index,
        )
        for index, control in enumerate(controls)
    ]
    adjusted = holm_adjust(raw_p)
    tests = {
        control: {"raw_p": raw_p[index], "holm_p": adjusted[index]}
        for index, control in enumerate(controls)
    }
    report = {
        "configuration": json_ready(vars(args)),
        "completed_runs": len(runs),
        "seed_reports": seed_reports,
        "summary": summary,
        "dialogue_bootstrap": bootstrap,
        "paired_randomization_tests": tests,
    }
    (args.output_dir / "aggregate_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown_report(report, args.output_dir / "report.md")
    return report


def write_markdown_report(report, path):
    summary = report["summary"]
    bootstrap = report["dialogue_bootstrap"]
    lines = [
        "# Final Architecture Cross-Validation Benchmark",
        "",
        f"Completed runs: {report['completed_runs']}",
        "",
        "## Primary results",
        "",
        "| Metric | Mean | Seed SD | 95% dialogue-bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    mapping = {
        "weighted_f1": ("Weighted F1", "matched_weighted_f1"),
        "over_text": ("Gain over text", "over_text"),
        "zero_audio": ("Zero-audio margin", "zero_audio"),
        "shuffled_audio": ("Shuffled-audio margin", "shuffled_audio"),
        "state": ("Reset-state margin", "state"),
    }
    for key, (label, bootstrap_key) in mapping.items():
        interval = bootstrap[bootstrap_key]["ci95"]
        lines.append(
            f"| {label} | {summary[key]['mean']:.4f} | "
            f"{summary[key]['std']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    lines.extend([
        "",
        "## Paired dialogue randomization tests",
        "",
        "| Control | Raw p | Holm-adjusted p |",
        "|---|---:|---:|",
    ])
    for control, result in report["paired_randomization_tests"].items():
        lines.append(
            f"| {control} | {result['raw_p']:.6f} | {result['holm_p']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "Every prediction is out-of-fold at the dialogue level, and BERT was retrained inside each outer fold. The architecture was nevertheless developed on MELD, so these results still estimate performance conditional on that prior architecture selection.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fold_file(assignments, path):
    path.write_text(json.dumps(assignments, indent=2, sort_keys=True), encoding="utf-8")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, default=ROOT / "data/MELD/MELD.Raw.tar.gz")
    parser.add_argument("--audio-cache-dir", type=Path, default=ROOT / "initial-testing/audio-cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarking/results/final-architecture-cv")
    parser.add_argument("--text-model", default="bert-base-uncased")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--only-folds", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--fold-seed", type=int, default=20260919)
    parser.add_argument("--analysis-seed", type=int, default=1701)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-epochs", type=int, default=5)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--text-gradient-accumulation", type=int, default=4)
    parser.add_argument("--text-learning-rate", type=float, default=2e-5)
    parser.add_argument("--text-weight-decay", type=float, default=0.01)
    parser.add_argument("--text-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--text-patience", type=int, default=2)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--fusion-epochs", type=int, default=30)
    parser.add_argument("--dialogue-batch-size", type=int, default=16)
    parser.add_argument("--fusion-gradient-accumulation", type=int, default=1)
    parser.add_argument("--fusion-learning-rate", type=float, default=2e-4)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.01)
    parser.add_argument("--fusion-patience", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--randomization-iterations", type=int, default=10000)
    parser.add_argument("--rebuild-text-cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(args):
    for name in (
        "outer_folds", "text_epochs", "text_batch_size",
        "text_gradient_accumulation", "encoder_batch_size", "fusion_epochs",
        "dialogue_batch_size", "fusion_gradient_accumulation",
        "bootstrap_iterations", "randomization_iterations",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if args.outer_folds < 2:
        raise ValueError("outer folds must be at least two")
    if not args.raw_archive.exists():
        raise FileNotFoundError(args.raw_archive)
    if not args.audio_cache_dir.exists():
        raise FileNotFoundError(args.audio_cache_dir)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be distinct")


def main():
    args = parse_arguments()
    validate_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_all_records(
        args.raw_archive, args.audio_cache_dir, args.context_window
    )
    assignments = assign_dialogue_folds(records, args.outer_folds, args.fold_seed)
    fold_path = args.output_dir / "dialogue_folds.json"
    if fold_path.exists():
        existing = json.loads(fold_path.read_text(encoding="utf-8"))
        if existing != assignments:
            raise ValueError("saved fold assignment differs from current assignment")
    else:
        write_fold_file(assignments, fold_path)
    fold_counts = {
        str(fold + 1): sum(value == fold for value in assignments.values())
        for fold in range(args.outer_folds)
    }
    print(
        f"Loaded {len(records)} utterances in {len(assignments)} dialogues; "
        f"fold dialogue counts: {fold_counts}",
        flush=True,
    )
    if args.dry_run:
        return 0
    device = select_device(torch)
    selected_folds = (
        [fold - 1 for fold in args.only_folds]
        if args.only_folds
        else list(range(args.outer_folds))
    )
    if any(fold < 0 or fold >= args.outer_folds for fold in selected_folds):
        raise ValueError("only-folds uses one-based fold numbers")
    prediction_paths = []
    started = time.perf_counter()
    for fold in selected_folds:
        for seed in args.seeds:
            print(
                f"Starting outer fold {fold + 1}/{args.outer_folds}, seed {seed}",
                flush=True,
            )
            prediction_paths.append(
                train_one_outer_run(records, assignments, fold, seed, args, device)
            )
    expected_runs = args.outer_folds * len(args.seeds)
    all_paths = sorted(args.output_dir.glob("fold-*/seed-*/outer_predictions.npz"))
    if len(all_paths) == expected_runs:
        report = aggregate(all_paths, args)
        print(
            f"Completed {expected_runs} runs; weighted F1 "
            f"{report['summary']['weighted_f1']['mean']:.4f} ± "
            f"{report['summary']['weighted_f1']['std']:.4f}",
            flush=True,
        )
        print(f"Report: {args.output_dir / 'report.md'}", flush=True)
    else:
        print(
            f"Completed subset: {len(all_paths)}/{expected_runs} runs available; "
            "aggregate analysis will run after every fold and seed is complete.",
            flush=True,
        )
    print(f"Runtime this invocation: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
