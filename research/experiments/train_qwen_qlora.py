#!/usr/bin/env python3
"""Train text-only Qwen3.5-2B emotion adapters with an 8-bit MLX base.

Install in an Apple Silicon Python environment:
  uv pip install --python .venv/bin/python 'mlx-lm[train]==0.31.3'
Run:
  .venv/bin/python research/experiments/train_qwen_qlora.py
Pilot:
  .venv/bin/python research/experiments/train_qwen_qlora.py --iters 8 --limit-train 16 --limit-valid 4 --output-dir research/experiments/qwen-qlora-pilot

Writes train/valid JSONL, run metadata, training.log and adapters/. Converts the
base once to models/qwen3.5-2b-mlx-8bit. No test split is read or used for tuning.
Early stopping uses development weighted F1 and also reports macro F1.
The best adapter is restored at completion. Adapters use MLX format and cannot
be passed directly to the Transformers-based compare_qwen.py.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import shutil
import math
import time
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

from compare_qwen import ROOT, contexts, key, make_prompt, json_constraint
from evaluate_meld import EMOTION_LABELS, compute_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='Qwen/Qwen3.5-2B')
    parser.add_argument('--quantized-model', type=Path, default=ROOT / 'models/qwen3.5-2b-mlx-8bit')
    parser.add_argument('--raw-archive', type=Path, default=ROOT / 'data/MELD/MELD.Raw.tar.gz')
    parser.add_argument('--csv-dir', type=Path, help='Optional directory containing train_sent_emo.csv and dev_sent_emo.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/experiments/training-output-qwen-qlora-8bit')
    parser.add_argument('--context-window', type=int, default=2)
    parser.add_argument('--iters', type=int, default=10000, help='MLX microbatch iterations; 10000 is roughly one MELD pass at batch size 1')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accumulation-steps', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=512)
    parser.add_argument('--patience', type=int, default=3, help='F1 checks without a significant gain before stopping')
    parser.add_argument('--min-delta', type=float, default=0.001, help='Absolute weighted-F1 improvement to reset patience')
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--limit-valid', type=int)
    parser.add_argument('--prepare-only', action='store_true', help='Prepare data/config without quantization or training')
    args = parser.parse_args(argv)
    for name in ('iters', 'batch_size', 'grad_accumulation_steps', 'max_seq_length', 'num_layers', 'rank', 'learning_rate', 'eval_every', 'patience'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.num_layers > 24 or args.context_window < 0:
        parser.error('Qwen3.5-2B has 24 layers; context window must be nonnegative')
    if any(value is not None and value < 1 for value in (args.limit_train, args.limit_valid)):
        parser.error('dataset limits must be positive')
    if args.iters % args.grad_accumulation_steps:
        parser.error('--iters must be divisible by --grad-accumulation-steps')
    if args.eval_every % args.grad_accumulation_steps or not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error('--eval-every must align with gradient accumulation; --min-delta must be finite and nonnegative')
    args.output_dir = args.output_dir.resolve()
    args.quantized_model = args.quantized_model.resolve()
    return args


def build_messages(rows, context_window):
    normalized = [dict(dialogue_id=row['Dialogue_ID'], utterance_id=row['Utterance_ID'],
                       speaker=row['Speaker'], utterance=row['Utterance'], label=row['Emotion'].lower())
                  for row in rows]
    if len({key(row) for row in normalized}) != len(normalized):
        raise ValueError('duplicate utterance IDs in split')
    normalized.sort(key=lambda row: (int(row['dialogue_id']), int(row['utterance_id'])))
    history = contexts(normalized, context_window)
    examples = []
    for row in normalized:
        if row['label'] not in EMOTION_LABELS:
            raise ValueError(f"unknown emotion: {row['label']}")
        examples.append({'messages': [
            {'role': 'user', 'content': make_prompt(row, history[key(row)], has_media=False)},
            {'role': 'assistant', 'content': json.dumps({'emotion': row['label']}, separators=(',', ':'))},
        ]})
    return examples


def training_config(args):
    return dict(model=str(args.quantized_model), data=str(args.output_dir / 'data'),
                train=True, test=False, fine_tune_type='lora', mask_prompt=True,
                adapter_path=str(args.output_dir / 'adapters'), seed=args.seed,
                batch_size=args.batch_size, grad_accumulation_steps=args.grad_accumulation_steps,
                iters=args.iters, num_layers=args.num_layers, max_seq_length=args.max_seq_length,
                grad_checkpoint=True, learning_rate=args.learning_rate,
                steps_per_report=math.gcd(args.grad_accumulation_steps, args.eval_every),
                steps_per_eval=min(args.eval_every, args.iters),
                save_every=min(args.eval_every, args.iters), val_batches=-1,
                early_stopping=dict(eval_every=args.eval_every, patience=args.patience, min_delta=args.min_delta),
                lora_parameters=dict(rank=args.rank, scale=16.0, dropout=0.05,
                                     keys=['self_attn.q_proj', 'self_attn.v_proj',
                                           'linear_attn.in_proj_qkv', 'linear_attn.out_proj']))


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best = self.reference = float('-inf')
        self.bad_checks = 0

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError('non-finite development score')
        improved = score > self.best
        self.best = max(self.best, score)
        if score > self.reference + self.min_delta:
            self.reference, self.bad_checks = score, 0
        else:
            self.bad_checks += 1
        return improved, self.bad_checks >= self.patience


def development_metrics(model, tokenizer, examples):
    import mlx.core as mx
    from mlx_lm import generate

    actual, predictions = [], []
    started = time.perf_counter()
    print(f'Development F1: generating predictions for {len(examples)} examples...', flush=True)
    was_training = model.training
    model.eval()
    try:
        for index, example in enumerate(examples, 1):
            messages = example['messages']
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                                   add_generation_prompt=True, return_dict=False)
            allowed = None
            _, max_tokens = json_constraint(tokenizer, 0)

            def constrain(tokens, logits):
                nonlocal allowed
                # MLX prefill consumes most prompt tokens before invoking the
                # processor; derive its remaining prefix length on first call.
                if allowed is None:
                    allowed, _ = json_constraint(tokenizer, len(tokens))
                ids = allowed(0, tokens)
                mask = mx.full(logits.shape, float('-inf'))
                mask[..., mx.array(ids)] = 0
                return logits + mask

            output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                              logits_processors=[constrain], verbose=False)
            prediction = json.loads(output)
            if set(prediction) != {'emotion'} or prediction['emotion'] not in EMOTION_LABELS:
                raise ValueError(f'invalid development prediction: {output}')
            predictions.append(prediction['emotion'])
            actual.append(json.loads(messages[-1]['content'])['emotion'])
            if index == 1 or index % 25 == 0 or index == len(examples):
                elapsed = time.perf_counter() - started
                remaining = elapsed / index * (len(examples) - index)
                print(f'Development F1: {index}/{len(examples)}, '
                      f'elapsed {elapsed:.0f}s, ETA {remaining:.0f}s', flush=True)
    finally:
        model.train(was_training)
    return compute_metrics(actual, predictions)


def train_worker(config_path):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset

    config = json.loads(Path(config_path).read_text())
    settings = config.pop('early_stopping')
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config))
    np.random.seed(args.seed)
    model, tokenizer = load(args.model)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    examples = [json.loads(line) for line in (Path(args.data) / 'valid.jsonl').read_text().splitlines()]
    adapter_dir = Path(args.adapter_path)
    run_dir = adapter_dir.parent
    stop = EarlyStopping(settings['patience'], settings['min_delta'])
    state = dict(best_iteration=None, last_iteration=0, stopped_early=False)

    class StopTraining(Exception):
        pass

    def check(iteration):
        metrics = development_metrics(model, tokenizer, examples)
        improved, should_stop = stop.update(metrics['weighted_f1'])
        if improved:
            mx.save_safetensors(str(adapter_dir / 'best.safetensors'), dict(tree_flatten(model.trainable_parameters())))
            state['best_iteration'] = iteration
            state['best_metrics'] = metrics
        event = dict(iteration=iteration, **metrics, best=improved, bad_checks=stop.bad_checks)
        with (run_dir / 'validation-metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(f"Iter {iteration}: Dev weighted F1 {metrics['weighted_f1']:.4f}, "
              f"macro F1 {metrics['macro_f1']:.4f}, patience {stop.bad_checks}/{stop.patience}", flush=True)
        (run_dir / 'early-stopping.json').write_text(json.dumps(state | {'bad_checks': stop.bad_checks}, indent=2) + '\n')
        if should_stop:
            state['stopped_early'] = True
            raise StopTraining

    class Callback:
        def on_val_loss_report(self, info):
            if info['iteration'] == 0:
                check(0)  # Retain the initial adapter if training only degrades F1.

        def on_train_loss_report(self, info):
            state['last_iteration'] = info['iteration']
            if info['iteration'] % settings['eval_every'] == 0 or info['iteration'] == args.iters:
                check(info['iteration'])

    try:
        train_model(args, model, train_set, valid_set, training_callback=Callback())
    except StopTraining:
        print('Early stopping: development weighted F1 has plateaued.', flush=True)
    # Also restores on iteration-limit completion, when MLX saves its last weights.
    shutil.copy2(adapter_dir / 'best.safetensors', adapter_dir / 'adapters.safetensors')
    (run_dir / 'early-stopping.json').write_text(json.dumps(state | {'bad_checks': stop.bad_checks}, indent=2) + '\n')
    print(f"Restored best adapter from iteration {state['best_iteration']}", flush=True)


def run_logged(command, log_path):
    with log_path.open('a') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def prepare_data(args, tokenizer):
    from cache_embeddings import load_split_rows

    destination = args.output_dir / 'data'
    destination.mkdir()
    report = {}
    for split, filename, limit in [('train', 'train', args.limit_train), ('dev', 'valid', args.limit_valid)]:
        if args.csv_dir:
            with (args.csv_dir / f'{split}_sent_emo.csv').open(encoding='cp1252', newline='') as stream:
                rows = list(csv.DictReader(stream))
        else:
            print(f'Reading MELD {split} metadata...', flush=True)
            rows = load_split_rows(args.raw_archive, split)
        examples = build_messages(rows, args.context_window)
        if limit:
            examples = examples[:limit]
        kept, dropped, counts = [], 0, Counter()
        for example in examples:
            messages = example['messages']
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
            if tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
                raise ValueError('chat template does not align the masked prompt with the completion')
            if len(tokens) > args.max_seq_length:
                dropped += 1
                continue  # Never silently truncate away the supervised response.
            kept.append(example)
            counts[json.loads(messages[-1]['content'])['emotion']] += 1
        if len(kept) < args.batch_size:
            raise ValueError(f'{split}: too few examples after length filtering')
        with (destination / f'{filename}.jsonl').open('w') as stream:
            for example in kept:
                stream.write(json.dumps(example, ensure_ascii=False) + '\n')
        report[split] = dict(source_count=len(rows), considered=len(examples), kept=len(kept),
                             dropped_overlength=dropped, label_counts=dict(counts))
    return report


def main():
    args = parse_args()
    if not args.prepare_only and importlib.util.find_spec('mlx_lm') is None:
        raise SystemExit("Install training dependencies first: uv pip install --python .venv/bin/python 'mlx-lm[train]==0.31.3'")
    if args.output_dir.exists():
        raise SystemExit('Output directory already exists; choose a new --output-dir.')
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # MLX ChatDataset does not forward enable_thinking. Persist the override in
    # the converted tokenizer so both prompt masking and inference use it.
    tokenizer.chat_template = '{% set enable_thinking = false %}\n' + tokenizer.chat_template
    args.output_dir.mkdir(parents=True)
    report = prepare_data(args, tokenizer)
    config = training_config(args)
    config_path = args.output_dir / 'training-config.json'  # JSON is valid YAML.
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    splits=report, quantization_bits=8, quantization_group_size=64,
                    notes=['Only train and dev splits are used.', 'Early stopping uses development weighted F1; macro F1 is also reported.',
                           'Final adapters are restored from the highest development weighted-F1 checkpoint.'])
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only:
        return
    if not args.quantized_model.exists():
        from huggingface_hub import snapshot_download

        # MLX's save path copies model-card metadata from a complete snapshot.
        # A Transformers inference cache may contain weights but omit those files.
        source = str(Path(args.model).resolve()) if Path(args.model).exists() else snapshot_download(args.model)
        args.quantized_model.parent.mkdir(parents=True, exist_ok=True)
        run_logged([sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', source,
                    '--mlx-path', str(args.quantized_model), '-q', '--q-bits', '8',
                    '--q-group-size', '64', '--dtype', 'bfloat16'], args.output_dir / 'conversion.log')
        (args.quantized_model / 'training-source.json').write_text(json.dumps({'model': args.model}))
    source_path = args.quantized_model / 'training-source.json'
    if not source_path.is_file() or json.loads(source_path.read_text())['model'] != args.model:
        raise ValueError('Existing quantized model has no matching source metadata; use a new --quantized-model path')
    quant_config = json.loads((args.quantized_model / 'config.json').read_text())
    if quant_config.get('quantization', {}).get('bits') != 8:
        raise ValueError('Converted model is not an 8-bit quantized base')
    tokenizer.save_pretrained(args.quantized_model)
    run_logged([sys.executable, str(Path(__file__).resolve()), '--worker-config', str(config_path)], args.output_dir / 'training.log')
    print(f'Adapters saved to {args.output_dir / "adapters"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker-config':
        train_worker(sys.argv[2])
    else:
        main()
