#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/third_party/llama.cpp/build-clang/bin/llama-server" ]]; then
  echo "Missing llama-server. Build third_party/llama.cpp first." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/models/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf" ]]; then
  echo "Missing chat model: models/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf" >&2
  exit 1
fi

exec python3 -m src.interface "$@"
