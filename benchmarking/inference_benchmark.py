#!/usr/bin/env python3
"""End-to-end latency, memory, cache, and resource benchmark for the HCL demo.

Run from the repository root with the project's Python environment:
  python benchmarking/inference_benchmark.py --runs 5 --audio-dir data/benchmark-audio

The harness can start the complete app (default), or target an already-running
app with --app-url. It never downloads assets or modifies model/cache state.
Results are written as JSON and CSV under benchmarking/results/ by default.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values: list[float], unit: str = "ms") -> dict[str, Any]:
    if not values:
        return {"unit": unit, "count": 0}
    return {
        "unit": unit,
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, .90),
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def http_get(url: str, timeout: float = 3) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "hcl-inference-benchmark/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read(), dict(response.headers.items())


def available_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_ready(url: str, process: subprocess.Popen[bytes] | None, timeout: float) -> float:
    start = time.perf_counter()
    deadline = start + timeout
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"inference process exited during startup ({process.returncode})")
        try:
            status, _, _ = http_get(url.rstrip("/") + "/", timeout=1)
            if status == 200:
                return (time.perf_counter() - start) * 1000
        except (OSError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(.1)
    raise TimeoutError(f"application did not become ready within {timeout:g}s: {url}")


def process_tree(root_pid: int) -> list[int]:
    """Return root and descendants using ps; empty on unsupported systems."""
    try:
        rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
        parents: dict[int, list[int]] = {}
        for row in rows:
            fields = row.split()
            if len(fields) == 2:
                parents.setdefault(int(fields[1]), []).append(int(fields[0]))
        found, stack = {root_pid}, [root_pid]
        while stack:
            for child in parents.get(stack.pop(), []):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        return sorted(found)
    except (OSError, ValueError, subprocess.SubprocessError):
        return [root_pid]


def process_memory(pids: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"processes": [], "rss_bytes": None, "vms_bytes": None}
    if not pids:
        return result
    try:
        # macOS and Linux ps expose RSS in KiB; VSZ is also KiB.
        rows = subprocess.check_output(["ps", "-o", "pid=,rss=,vsz=,comm=", "-p", ",".join(map(str, pids))], text=True)
        for row in rows.splitlines():
            fields = row.strip().split(None, 3)
            if len(fields) >= 3:
                result["processes"].append({"pid": int(fields[0]), "rss_bytes": int(fields[1]) * 1024,
                                            "vms_bytes": int(fields[2]) * 1024,
                                            "command": fields[3] if len(fields) > 3 else ""})
        result["rss_bytes"] = sum(x["rss_bytes"] for x in result["processes"])
        result["vms_bytes"] = sum(x["vms_bytes"] for x in result["processes"])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def system_memory() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info = {"total_bytes": vm.total, "available_bytes": vm.available,
                "used_bytes": vm.used, "percent_used": vm.percent}
    except ImportError:
        if platform.system() == "Darwin":
            try:
                info["available_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    return info


def gpu_snapshot() -> dict[str, Any]:
    if shutil.which("nvidia-smi"):
        try:
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=3)
            return {"backend": "nvidia-smi", "devices": [line.strip().split(", ") for line in raw.splitlines()]}
        except (OSError, subprocess.SubprocessError):
            pass
    if platform.system() == "Darwin":
        try:
            return {"backend": "macOS", "metal": "not separately observable; included in unified memory"}
        except Exception:
            pass
    return {"backend": None, "devices": []}


def read_audio(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    # Valid, short PCM WAV silence provides a deterministic no-download fallback.
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    return buffer.getvalue()


def post_chat(app_url: str, text: str, audio: bytes | None, timeout: float) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}
    if audio is not None:
        payload["audio"] = base64.b64encode(audio).decode("ascii")
    request = urllib.request.Request(app_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first_byte = first_delta = metadata_at = done_at = None
    output = bytearray(); events: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            line = response.readline()
            if not line:
                break
            now = time.perf_counter()
            if first_byte is None:
                first_byte = now
            output.extend(line)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("type") == "delta" and first_delta is None:
                first_delta = now
            if event.get("type") == "metadata":
                metadata_at = now
            if event.get("type") in {"done", "error"}:
                done_at = now
    end = time.perf_counter()
    deltas = [e.get("text", "") for e in events if e.get("type") == "delta"]
    return {
        "elapsed_ms": (end - start) * 1000,
        "first_byte_ms": (first_byte - start) * 1000 if first_byte else None,
        "first_token_ms": (first_delta - start) * 1000 if first_delta else None,
        "metadata_ms": (metadata_at - start) * 1000 if metadata_at else None,
        "completion_ms": (done_at - start) * 1000 if done_at else (end - start) * 1000,
        "response_bytes": len(output), "output_chars": sum(map(len, deltas)),
        "delta_count": len(deltas), "events": [e.get("type") for e in events],
        "error": next((e.get("message") for e in events if e.get("type") == "error"), None),
    }


def metric_snapshot(llama_url: str | None) -> dict[str, Any]:
    if not llama_url:
        return {"available": False}
    result: dict[str, Any] = {"available": False, "metrics": {}, "slots": None}
    try:
        _, body, _ = http_get(llama_url.rstrip("/") + "/metrics")
        parsed: dict[str, float] = {}
        for line in body.decode("utf-8", "replace").splitlines():
            match = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
            if match:
                parsed[match.group(1)] = float(match.group(2))
        result.update(available=True, metrics=parsed, raw=body.decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError, TimeoutError):
        pass
    try:
        _, body, _ = http_get(llama_url.rstrip("/") + "/slots")
        result["slots"] = json.loads(body)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass
    return result


def delta_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    a, b = before.get("metrics", {}), after.get("metrics", {})
    return {key: b[key] - a[key] for key in a.keys() & b.keys() if b[key] >= a[key]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-url", help="target existing app; otherwise launch it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--llama-url", help="llama.cpp URL when using --app-url; discovers child server otherwise")
    parser.add_argument("--runs", type=int, default=5, help="measured repetitions per workload (minimum 1)")
    parser.add_argument("--warmup", type=int, default=1, help="unreported warm-up repetitions per workload")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--audio-dir", type=Path, help="directory of .wav/.webm/.ogg inputs; first file is reused")
    parser.add_argument("--text", default="I had a difficult morning, but I am trying to stay hopeful.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarking/results")
    parser.add_argument("--label", default="default")
    parser.add_argument("--no-transcription", action="store_true", help="skip /api/transcribe workload")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup must be >= 0")

    app_process = None
    started_at = time.perf_counter()
    cold_start = args.app_url is None
    app_url = args.app_url
    llama_url = args.llama_url
    logs = tempfile.TemporaryFile() if cold_start else None
    try:
        if cold_start:
            port = args.port or available_port(args.host)
            app_url = f"http://{args.host}:{port}"
            command = [PYTHON, "-m", "src.inference", "--host", args.host, "--port", str(port), "--llama-metrics"]
            app_process = subprocess.Popen(command, cwd=ROOT, stdout=logs, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            startup_ready_ms = wait_ready(app_url, app_process, args.startup_timeout)
            llama_url = llama_url or f"http://127.0.0.1:{discover_llama_port(app_process.pid, logs)}"
        else:
            startup_ready_ms = wait_ready(app_url, None, min(args.startup_timeout, 15))
        assert app_url
        process_ids = process_tree(app_process.pid) if app_process else []
        initial_memory = process_memory(process_ids)
        baseline_system = system_memory()
        baseline_gpu = gpu_snapshot()
        before_metrics = metric_snapshot(llama_url)

        audio_paths = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in {".wav", ".webm", ".ogg"}) if args.audio_dir and args.audio_dir.is_dir() else []
        audio_path = audio_paths[0] if audio_paths else None
        audio = read_audio(audio_path) if audio_path else None
        workloads: dict[str, Any] = {}
        for name, with_audio in (("chat_text_only", False), ("chat_text_plus_audio", True)):
            if with_audio and audio is None:
                workloads[name] = {"skipped": "no audio supplied; pass --audio-dir for acoustic inference"}
                continue
            samples = []
            for idx in range(args.warmup + args.runs):
                sample = post_chat(app_url, args.text, audio if with_audio else None, args.request_timeout)
                sample["iteration"] = idx
                sample["warmup"] = idx < args.warmup
                if sample["error"]:
                    raise RuntimeError(f"{name} returned an inference error: {sample['error']}")
                samples.append(sample)
            measured = [s for s in samples if not s["warmup"]]
            workloads[name] = {
                "runs": measured,
                "latency": {key: summarize([s[field] for s in measured if s[field] is not None])
                            for key, field in (("end_to_end", "elapsed_ms"), ("first_byte", "first_byte_ms"),
                                               ("first_token", "first_token_ms"), ("emotion_metadata", "metadata_ms"),
                                               ("completion", "completion_ms"))},
                "output_chars": summarize([float(s["output_chars"]) for s in measured], "characters"),
            }

        transcription = {"skipped": "disabled" if args.no_transcription else "no audio supplied; pass --audio-dir"}
        if audio is not None and not args.no_transcription:
            payload = json.dumps({"audio": base64.b64encode(audio).decode("ascii")}).encode()
            request = urllib.request.Request(app_url.rstrip("/") + "/api/transcribe", data=payload,
                                             headers={"Content-Type": "application/json"}, method="POST")
            samples = []
            for idx in range(args.warmup + args.runs):
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                    body = json.loads(response.read())
                elapsed = (time.perf_counter() - start) * 1000
                if "error" in body:
                    raise RuntimeError(f"transcription failed: {body['error']}")
                samples.append({"elapsed_ms": elapsed, "text_chars": len(body.get("text", "")), "warmup": idx < args.warmup})
            transcription = {"runs": [x for x in samples if not x["warmup"]],
                             "latency": summarize([x["elapsed_ms"] for x in samples if not x["warmup"]])}

        after_metrics = metric_snapshot(llama_url)
        peak_memory = process_memory(process_ids)
        result = {
            "schema_version": 1, "label": args.label, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "system": {"platform": platform.platform(), "python": sys.version, "processor": platform.processor(),
                       "logical_cpus": os.cpu_count(), "memory_before": baseline_system, "gpu_before": baseline_gpu},
            "configuration": {"cold_start": cold_start, "app_url": app_url, "llama_url": llama_url,
                              "runs": args.runs, "warmup": args.warmup, "cache_prompt": True,
                              "prompt_cache_state": "process-warm sequential requests; app has no explicit response cache",
                              "filesystem_cache_state": "OS state uncontrolled; cold-start is process-cold only unless host cache was cleared externally",
                              "audio_file": str(audio_path) if audio_path else None,
                              "audio_bytes": len(audio) if audio else None},
            "startup": {"ready_ms": startup_ready_ms, "total_from_harness_start_ms": (time.perf_counter() - started_at) * 1000 if cold_start else None},
            "memory": {"initial_process_tree": initial_memory, "end_process_tree": peak_memory,
                       "system_after": system_memory(), "gpu_after": gpu_snapshot(),
                       "note": "RSS/VMS are sampled snapshots, not peak high-water marks; GPU snapshots depend on driver tooling."},
            "cache": {"llama_metrics_before": before_metrics, "llama_metrics_after": after_metrics,
                      "llama_metric_deltas": delta_metrics(before_metrics, after_metrics),
                      "interpretation": "llama prompt-cache counters are exposed when server is started with --metrics; prompt reuse benefits are best compared across identical repeated prompts."},
            "workloads": workloads, "transcription": transcription,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = args.output_dir / f"inference-{args.label}-{stamp}"
        base.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with base.with_suffix(".csv").open("w", newline="") as fp:
            fields = ["workload", "iteration", "warmup", "elapsed_ms", "first_byte_ms", "first_token_ms", "metadata_ms", "completion_ms", "response_bytes", "output_chars", "delta_count"]
            writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader()
            for workload, details in workloads.items():
                for sample in details.get("runs", []):
                    writer.writerow({"workload": workload, **sample})
        print(json.dumps({"json": str(base.with_suffix('.json')), "csv": str(base.with_suffix('.csv')),
                          "startup_ready_ms": startup_ready_ms, "workloads": {k: v.get("latency") for k, v in workloads.items()}}, indent=2))
        return 0
    finally:
        if app_process is not None:
            try:
                os.killpg(app_process.pid, signal.SIGTERM)
                app_process.wait(timeout=12)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(app_process.pid, signal.SIGKILL)
                except OSError:
                    pass


def discover_llama_port(pid: int, log_file: Any) -> int:
    """Find the child llama-server process's bound port, if observable."""
    # The app's random port is selected internally; inspect localhost listening
    # sockets and retain the one exposing the server's public model properties.
    deadline = time.monotonic() + 15
    checked: set[int] = set()
    while time.monotonic() < deadline:
        for child in process_tree(pid):
            if child == pid:
                continue
            try:
                text = subprocess.check_output(["lsof", "-Pan", "-p", str(child), "-iTCP", "-sTCP:LISTEN"], text=True, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                continue
            for match in re.finditer(r"(?:127\.0\.0\.1|\*|localhost):(\d+)\s+\(LISTEN\)", text):
                port = int(match.group(1))
                if port in checked:
                    continue
                checked.add(port)
                try:
                    http_get(f"http://127.0.0.1:{port}/props", timeout=.3)
                    return port
                except (OSError, urllib.error.URLError, TimeoutError):
                    pass
        time.sleep(.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
