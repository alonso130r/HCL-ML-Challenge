#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/third_party/llama.cpp/CMakeLists.txt" ]]; then
  echo "Initializing llama.cpp submodule"
  git submodule update --init --recursive
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Git LFS is required to download the fine-tuned demo weights." >&2
  exit 1
fi
git lfs install --local >/dev/null
echo "Pulling Git LFS assets"
git lfs pull

VENV_DIR="${HCL_VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/build-clang/bin/llama-server"
CHAT_MODEL="$ROOT_DIR/models/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf"
TEXT_MODEL="$ROOT_DIR/models/text-emotion/model.safetensors"
EMOTION_MODEL="$ROOT_DIR/models/frame-attention/best_frame_attention.pt"

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
  export HCL_DEVICE="cuda"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  export HCL_DEVICE="metal"
else
  export HCL_DEVICE="cpu"
fi
echo "Selected device: $HCL_DEVICE"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [[ ! -f "$TEXT_MODEL" || ! -f "$EMOTION_MODEL" ]]; then
  echo "Fine-tuned demo weights are missing. Pull them with Git LFS, then rerun:" >&2
  echo "  git lfs pull" >&2
  exit 1
fi

if grep -q "git-lfs.github.com/spec/v1" "$TEXT_MODEL" "$EMOTION_MODEL" 2>/dev/null; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS is required to download the fine-tuned demo weights." >&2
    exit 1
  fi
  echo "Downloading Git LFS demo weights"
  git lfs pull --include="models/text-emotion/**,models/frame-attention/**"
fi

echo "Installing Python dependencies"
"$PIP_BIN" install -q -r research/experiments/requirements.txt

if [[ ! -f "$CHAT_MODEL" ]]; then
  echo "Downloading Qwen3 chat model"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path("models/qwen3-1.7b-gguf")
target.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id="Qwen/Qwen3-1.7B-GGUF",
    filename="Qwen3-1.7B-Q8_0.gguf",
    local_dir=target,
)
PY
fi

if [[ ! -x "$LLAMA_BIN" || ! -f "$LLAMA_DIR/build-clang/.hcl-device" || "$(cat "$LLAMA_DIR/build-clang/.hcl-device" 2>/dev/null)" != "$HCL_DEVICE" ]]; then
  echo "Building llama-server"
  if [[ "$HCL_DEVICE" == "cuda" ]]; then
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-clang" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
  elif [[ "$HCL_DEVICE" == "metal" ]]; then
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-clang" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
  else
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-clang" \
      -DCMAKE_BUILD_TYPE=Release
  fi
  cmake --build "$LLAMA_DIR/build-clang" --target llama-server --config Release --parallel
  printf '%s\n' "$HCL_DEVICE" > "$LLAMA_DIR/build-clang/.hcl-device"
fi

echo "Caching local emotion model"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("emotion2vec/emotion2vec_plus_base")
PY

exec "$PYTHON_BIN" -m src.interface "$@"
