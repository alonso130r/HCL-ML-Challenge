# Human-Computer Lab Challenge

### Primary track

This demo will be **text + audio**. Why? Low-latency video processing is
 extremely difficult to accomplish under the time and hardware constraints
 (2-3 days of work, <6B params, local inference). Staying with the
philosophy that a simpler, more useful product is better than a complex,
half-working product also points towards audio as extracting emotion from
video is a lot more complex than audio and more likely to be less effective.

The interface records audio in the browser, transcribes it locally with a
small FunASR model, and places the transcript in the composer for editing
before sending. The original audio is retained for local emotion analysis.

## Local transcription model

Place a small FunASR-compatible model in `models/funasr-small`, or pass a
different directory with `--transcription-model`. If that directory does not
exist, startup automatically downloads and caches FunASR's `paraformer-zh`
model. No browser speech service is used.
The `ct-punc` FunASR model is also loaded automatically so transcripts include
basic punctuation.

## Run the demo

Build `third_party/llama.cpp` so `build-clang/bin/llama-server` exists, then run:

```bash
git lfs install
git lfs pull
./run_app.sh
```

The script creates `.venv`, installs Python dependencies, downloads the base
chat and emotion models, builds `llama-server`, and starts the app. It accepts
the same options as the Python app, for example `./run_app.sh --port 8001`.
Fine-tuned demo weights are stored with Git LFS; experiment outputs remain
local and ignored.

### My definition of "real-time"

A system like this is useless unless the client can hold an actual conversation
with it. My definition of real-time is **the user receives a response in N seconds**,
so that live conversation feels usable. Additionally, LLM responses are streamed
to the client (the full response isn't needed to begin reciting it), further
reducing the perceived latency.

### Model architecture
