import unittest

import compare_qwen as comparison


class ComparisonTests(unittest.TestCase):
    def test_context_is_causal_and_does_not_contain_labels(self):
        rows = [dict(dialogue_id="0", utterance_id=str(i), speaker="A",
                     utterance=f"turn {i}", expected="fear", predicted="joy")
                for i in range(4)]
        prompt = comparison.make_prompt(rows[2], comparison.contexts(rows, 2)[("0", "2")])
        self.assertIn("turn 0", prompt)
        self.assertIn("turn 2", prompt)
        self.assertNotIn("turn 3", prompt)
        self.assertNotIn('"expected"', prompt)
        self.assertNotIn('"predicted"', prompt)

    def test_structured_decoder_only_allows_complete_label_json(self):
        class Tokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        allowed, limit = comparison.json_constraint(Tokenizer(), 2)
        class IDs(list):
            def tolist(self):
                return list(self)
        prefix = '{"emotion":"'
        self.assertEqual(set(allowed(0, IDs([7, 8] + list(map(ord, prefix))))),
                         {ord(label[0]) for label in comparison.EMOTION_LABELS})
        complete = '{"emotion":"joy"}'
        self.assertEqual(allowed(0, IDs([7, 8] + list(map(ord, complete)))), [0])
        self.assertGreater(limit, len(complete))
        # GPU generation may evaluate constraints again after EOS, including
        # padding for a sequence which has already finished.
        for padding in ([0], [0, 0]):
            self.assertEqual(allowed(0, IDs([7, 8] + list(map(ord, complete)) + padding)), [0])
        with self.assertRaises(ValueError):
            allowed(0, IDs([7, 8, ord('{'), 0]))
        with self.assertRaises(ValueError):
            allowed(0, IDs([7, 8] + list(map(ord, complete)) + [0, ord('x')]))

    def test_video_sampling_handles_short_and_single_frame_clips(self):
        import tempfile
        from pathlib import Path
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            for count in (1, 6, 7, 12):
                path = Path(directory) / f"clip-{count}.mp4"
                with av.open(str(path), mode="w") as container:
                    stream = container.add_stream("mpeg4", rate=10)
                    stream.width = stream.height = 64
                    stream.pix_fmt = "yuv420p"
                    for index in range(count):
                        frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 10, dtype=np.uint8), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                frames, metadata = comparison.load_video_frames(path, 8)
                self.assertEqual(len(frames), max(2, min(8, count)))
                self.assertEqual(len(metadata.frames_indices), len(frames))
                self.assertTrue(all(0 <= i < count for i in metadata.frames_indices))

    def test_paired_scores_exclude_failed_rows_for_every_model(self):
        records = [
            dict(model=model, dialogue_id="0", utterance_id=str(i), expected="joy",
                 predicted="joy" if not (model == "omni" and i == 1) else None,
                 status="error" if model == "omni" and i == 1 else "ok", latency_ms=None)
            for model in ("baseline", "omni", "vision") for i in range(2)
        ]
        summary = comparison.summarize(records, ["baseline", "omni", "vision"])
        self.assertEqual(summary["paired_count"], 1)
        self.assertEqual(summary["models"]["omni"]["errors"], 1)
        for result in summary["models"].values():
            self.assertEqual(result["paired_metrics"]["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
