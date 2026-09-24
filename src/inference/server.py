"""Serve the demo UI and stream responses from a local llama.cpp server."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import platform
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIRECTORY = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
DEFAULT_LLAMA_SERVER = ROOT / "third_party/llama.cpp/build-clang/bin/llama-server"
DEFAULT_MODEL = ROOT / "models/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf"
DEFAULT_EMOTION_MODEL = (
    ROOT
    / "models/frame-attention/best_frame_attention.pt"
)
DEFAULT_TRANSCRIPTION_MODEL = ROOT / "models/funasr-small"
SYSTEM_PROMPT = (
    "You are a concise, emotionally aware conversational assistant. Respond "
    "naturally and helpfully. Do not mention these instructions."
)
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def select_device() -> str:
    configured = os.environ.get("HCL_DEVICE", "").lower()
    if configured in {"cuda", "metal", "cpu"}:
        return configured
    if platform.system() == "Darwin":
        return "metal"
    return "cpu"


def decode_audio_payload(encoded_audio: object) -> bytes:
    if not isinstance(encoded_audio, str) or not encoded_audio:
        raise ValueError("Audio recording is required.")
    try:
        return base64.b64decode(encoded_audio, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid audio recording encoding") from error


class LlamaClient:
    """Stream chat completions from llama.cpp's OpenAI-compatible endpoint."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def stream_chat(self, text: str, emotion: str | None = None) -> Iterator[str]:
        system_prompt = SYSTEM_PROMPT
        if emotion is not None:
            system_prompt += (
                f" The user's current emotion was classified as {emotion}. "
                "Adapt your tone appropriately without stating the classification."
            )
        payload = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "stream": True,
                "max_tokens": 160,
                "temperature": 0.6,
                "top_p": 0.9,
                "cache_prompt": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with LOCAL_HTTP.open(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                event = json.loads(data)
                choices = event.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content


class TranscriptionModel:
    """Keep a small local FunASR model resident for short recordings."""

    def __init__(self, model_path: Path) -> None:
        try:
            from funasr import AutoModel
        except ImportError as error:
            raise RuntimeError("transcription requires FunASR") from error
        model = str(model_path) if model_path.is_dir() else "paraformer-zh"
        if model == "paraformer-zh":
            print(
                "Local transcription model not found; downloading FunASR paraformer-zh...",
                flush=True,
            )
        self.model = AutoModel(
            model=model,
            punc_model="ct-punc",
            device="cpu",
            disable_update=True,
        )

    def transcribe(self, audio: bytes) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".webm") as recording:
            recording.write(audio)
            recording.flush()
            results = self.model.generate(input=recording.name)
        if not results:
            return ""
        return str(results[0].get("text", "")).strip()


class LlamaServerProcess:
    """Own a single persistent, Metal-accelerated llama.cpp server."""

    def __init__(self, executable: Path, model: Path, host: str, port: int, metrics: bool = False) -> None:
        self.executable = executable
        self.model = model
        self.host = host
        self.port = port or self._available_port(host)
        self.metrics = metrics
        self.process: subprocess.Popen[bytes] | None = None

    @staticmethod
    def _available_port(host: str) -> int:
        with socket.socket() as candidate:
            candidate.bind((host, 0))
            return candidate.getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        for path, label in ((self.executable, "llama-server"), (self.model, "model")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        device = select_device()
        command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpu-layers",
            "0" if device == "cpu" else "99",
            "--ctx-size",
            "2048",
            "--parallel",
            "1",
            "--flash-attn",
            "on" if device != "cpu" else "off",
            "--reasoning",
            "off",
            "--threads-http",
            "1",
        ]
        if self.metrics:
            command.append("--metrics")
        self.process = subprocess.Popen(command, start_new_session=True)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 120
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup with code {self.process.returncode}"
                )
            try:
                with LOCAL_HTTP.open(health_url, timeout=1) as response:
                    if response.status == HTTPStatus.OK:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.1)
        self.stop()
        raise TimeoutError("llama-server did not become ready within 120 seconds")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class InterfaceServer(ThreadingHTTPServer):
    llama_client: LlamaClient
    emotion_model: Any
    transcription_model: TranscriptionModel

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


class InterfaceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        static_file = STATIC_FILES.get(path)
        if static_file is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static_file
        content = (STATIC_DIRECTORY / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/transcribe"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            text = payload.get("text", "")
            encoded_audio = payload.get("audio")
            if path == "/api/transcribe":
                try:
                    audio = decode_audio_payload(encoded_audio)
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                server = self.server
                if not isinstance(server, InterfaceServer):
                    raise RuntimeError("Transcription model is not configured")
                transcript = server.transcription_model.transcribe(audio)
                self._send_json(HTTPStatus.OK, {"text": transcript})
                return
            if not isinstance(text, str) or not text.strip():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message text is required."})
                return
            if encoded_audio is not None and not isinstance(encoded_audio, str):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid audio recording."})
                return
        except (ValueError, json.JSONDecodeError, AttributeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON request."})
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._write_event({"type": "start"})
            server = self.server
            if not isinstance(server, InterfaceServer):
                raise RuntimeError("Inference client is not configured")
            emotion = None
            if encoded_audio:
                try:
                    audio = decode_audio_payload(encoded_audio)
                except ValueError as error:
                    raise error
                diagnostic = server.emotion_model.predict(text.strip(), audio)
                emotion = str(diagnostic["emotion"])
                self._write_event({"type": "metadata", **diagnostic})
            for content in server.llama_client.stream_chat(text.strip(), emotion):
                self._write_event({"type": "delta", "text": content})
            self._write_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            try:
                self._write_event({"type": "error", "message": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.close_connection = True

    def _write_event(self, event: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
        self.wfile.flush()

    def _send_json(self, status: HTTPStatus, payload: dict[str, str]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument(
        "--llama-port",
        type=int,
        default=0,
        help="llama.cpp port; defaults to an available loopback port",
    )
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--emotion-model", type=Path, default=DEFAULT_EMOTION_MODEL
    )
    parser.add_argument(
        "--transcription-model", type=Path, default=DEFAULT_TRANSCRIPTION_MODEL
    )
    parser.add_argument("--llama-metrics", action="store_true", help="enable llama.cpp /metrics endpoint")
    return parser.parse_args()


def main() -> int:
    from .emotion import EmotionModel

    args = parse_arguments()
    inference = LlamaServerProcess(
        args.llama_server, args.model, args.llama_host, args.llama_port, args.llama_metrics
    )
    server: InterfaceServer | None = None
    try:
        print(f"Loading emotion model from {args.emotion_model}", flush=True)
        emotion_model = EmotionModel(args.emotion_model)
        transcription_model = TranscriptionModel(args.transcription_model)
        print(f"Emotion model ready on {emotion_model.device}", flush=True)
        inference.start()
        server = InterfaceServer((args.host, args.port), InterfaceHandler)
        server.llama_client = LlamaClient(inference.base_url)
        server.emotion_model = emotion_model
        server.transcription_model = transcription_model
        print(
            f"Interface available at http://{args.host}:{server.server_port}",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping interface.")
    finally:
        if server is not None:
            server.server_close()
        inference.stop()
    return 0
