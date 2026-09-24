# End-to-end inference benchmark

`inference_benchmark.py` measures the running HCL demo from its public HTTP
interface. It covers service startup and readiness, first-byte and first
generated-token latency, emotion metadata latency, full response latency,
transcription latency, response size, process-tree resident and virtual memory,
host memory, optional NVIDIA GPU memory/utilization, llama.cpp Prometheus
counters, and llama.cpp slot state.

## Run

Use the same Python environment and model assets as the demo. From the
repository root:

```sh
python benchmarking/inference_benchmark.py --runs 5 --audio-dir path/to/audio
```

With no audio directory, the script still measures text-only chat; acoustic
chat and transcription are marked skipped. For representative results, pass
short local `.wav`, `.webm`, or `.ogg` recordings. The first file in sorted
order is reused for consistent repetitions. No model downloads are initiated
by the harness.

To benchmark an already running app, use `--app-url http://127.0.0.1:8000` and
optionally `--llama-url http://127.0.0.1:<port>`. llama.cpp metrics are
available only if that server was launched with `--metrics`. A self-started
run enables this endpoint automatically through the benchmark-only
`--llama-metrics` option.

Results are saved under `benchmarking/results/` as timestamped JSON and CSV.
The JSON contains host and run configuration, raw per-request observations,
latency distribution summaries, process and host memory snapshots, cache
counter samples/deltas, and response events. CSV contains one row per measured
chat request for quick plotting or comparison. Use `--label` to name a run,
`--warmup` to set unreported warmups, and `--output-dir` to change the output
location.

## Workloads and interpretation

- `chat_text_only`: browser-facing `/api/chat`, then streamed local Qwen output.
- `chat_text_plus_audio`: full audio emotion feature extraction and
  classification, metadata event, then Qwen stream.
- `transcription`: `/api/transcribe` with the local FunASR model.
- Startup is measured from process launch until the interface root returns
  HTTP 200. Model loading and llama.cpp readiness are therefore included.
- `first_token_ms` means time to first non-empty streamed text delta, not the
  model's internal tokenization or decode time. `metadata_ms` ends when the
  emotion metadata event reaches the client.
- The first measured request is process-warm but prompt-cache-cold for its
  unique prompt; later sequential requests exercise llama.cpp's
  `cache_prompt: true` behavior. Compare identical text and run counts across
  builds. There is no application-level response cache.
- llama.cpp metrics include raw Prometheus output, parsed counters, and
  before/after deltas. Depending on the bundled llama.cpp version, useful
  counters include prompt-cache reuse and token throughput. `/slots` provides
  per-slot prompt/decode token and timing detail when exposed by that build.
- RSS/VMS are snapshots, not guaranteed high-water values. NVIDIA details are
  gathered with `nvidia-smi` when available. On Apple Silicon, GPU memory is
  shared with system memory and cannot be isolated by this portable harness.

## Cache and reproducibility limits

The harness does not clear operating-system page caches because doing so is
privileged, platform-specific, and disruptive to other processes. Its
"process-cold" startup means a fresh app and llama.cpp process, while model
files and OS caches may already be warm. Record host load, power mode, device,
driver/runtime versions, model artifact revisions, and whether model files
were already cached when comparing runs. For cold storage measurements, reboot
or use a dedicated machine and document the procedure externally.

Prompt cache details are captured via llama.cpp metrics when supported. Metrics
are cumulative process counters, so the report records snapshots and deltas;
they do not reveal a universal cache hit ratio for every server version. Use
the raw metric text and `/slots` output alongside latency. The benchmark uses
sequential requests because this app configures a single llama.cpp slot; it
does not claim concurrent capacity or queueing performance.

For load and concurrency studies, run separate labeled trials with an external
load generator against `--app-url`, and retain the server's own logs/metrics.
Do not compare those runs directly with the default serial latency summary.
