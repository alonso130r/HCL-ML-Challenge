import base64
import unittest

from src.interface.server import decode_audio_payload


class FakeTranscriptionModel:
    def transcribe(self, audio: bytes) -> str:
        assert audio == b"audio"
        return "hello from the microphone"


class TranscriptionEndpointTests(unittest.TestCase):
    def test_transcribe_returns_editable_text(self):
        payload = {"audio": base64.b64encode(b"audio").decode()}
        model = FakeTranscriptionModel()
        result = model.transcribe(decode_audio_payload(payload["audio"]))
        self.assertEqual(result, "hello from the microphone")

    def test_transcribe_rejects_missing_audio(self):
        with self.assertRaisesRegex(ValueError, "Audio recording is required"):
            decode_audio_payload(None)


if __name__ == "__main__":
    unittest.main()
