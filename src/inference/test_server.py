import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from src.inference.server import InterfaceHandler, mock_chat_events


class InterfaceServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), InterfaceHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_home_page_contains_demo_controls(self):
        with urllib.request.urlopen(f"{self.base_url}/") as response:
            page = response.read().decode()

        self.assertEqual(response.status, 200)
        self.assertIn("Multimodal inference", page)
        self.assertNotIn("Eunoia", page)
        self.assertIn('id="camera-preview"', page)
        self.assertIn('id="record-button"', page)
        self.assertIn('id="message-input"', page)

    def test_mock_chat_events_have_stable_stream_contract(self):
        events = list(mock_chat_events("I had a difficult morning."))

        self.assertEqual(events[0], {"type": "start"})
        self.assertGreaterEqual(
            len([event for event in events if event["type"] == "delta"]), 2
        )
        self.assertEqual(
            events[-2],
            {"type": "metadata", "emotion": "neutral", "confidence": 0.0},
        )
        self.assertEqual(events[-1], {"type": "done"})

    def test_chat_endpoint_streams_ndjson(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps({"text": "Hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request) as response:
            events = [json.loads(line) for line in response if line.strip()]

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "application/x-ndjson")
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "done")

    def test_chat_endpoint_rejects_empty_text(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=b'{"text": "  "}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)

        self.assertEqual(raised.exception.code, 400)
        body = json.loads(raised.exception.read())
        self.assertEqual(body, {"error": "Message text is required."})


if __name__ == "__main__":
    unittest.main()
