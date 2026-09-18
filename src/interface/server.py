"""Serve the local demo UI and a mock streaming chat response."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


STATIC_DIRECTORY = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def mock_chat_events(text: str) -> Iterator[dict[str, object]]:
    """Yield the future inference boundary as deterministic mock events."""
    yield {"type": "start"}
    response = (
        "I’m with you. It sounds like there’s something meaningful behind what "
        f"you shared: “{text.strip()}” What feels most important right now?"
    )
    words = response.split()
    for index in range(0, len(words), 3):
        piece = " ".join(words[index : index + 3])
        if index + 3 < len(words):
            piece += " "
        yield {"type": "delta", "text": piece}
    yield {"type": "metadata", "emotion": "neutral", "confidence": 0.0}
    yield {"type": "done"}


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
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message text is required."})
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
            for event in mock_chat_events(text):
                self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                self.wfile.flush()
                if event["type"] == "delta":
                    time.sleep(0.06)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

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
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    server = ThreadingHTTPServer((args.host, args.port), InterfaceHandler)
    print(f"Interface available at http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping interface.")
    finally:
        server.server_close()
    return 0
