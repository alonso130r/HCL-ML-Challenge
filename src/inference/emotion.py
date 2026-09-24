"""Live inference for the latest frame-attention MELD model."""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INITIAL_TESTING = ROOT / "research/experiments"
if str(INITIAL_TESTING) not in sys.path:
    sys.path.insert(0, str(INITIAL_TESTING))

from audio_phase1 import (  # noqa: E402
    Emotion2VecExtractor,
    extract_prosody_contours,
    normalize,
    summarize_sequence,
)
from evaluate_meld import EMOTION_LABELS  # noqa: E402
from train_text_audio import extract_egemaps, sanitize_acoustic_features  # noqa: E402
from train_recurrent_dialogue_frame_attention import (  # noqa: E402
    FrameAttentionRecurrentModel,
)


TEXT_MODEL = ROOT / "models/text-emotion"
CHECKPOINT = (
    ROOT
    / "models/frame-attention/best_frame_attention.pt"
)
ACOUSTIC_NORMALIZATION_FILENAME = "speaker_acoustic_normalization.npz"


def _model_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        text_projection_dimension=256,
        audio_projection_dimension=128,
        dialogue_state_dimension=128,
        speaker_state_dimension=64,
        dropout=0.2,
        dialogue_state_dropout=0.05,
        speaker_state_dropout=0.10,
        audio_dropout=0.10,
        dialogue_reset_probability=0.01,
        speaker_reset_probability=0.03,
        context_max_gate=0.25,
        audio_max_gate=1.0,
        initial_gate_bias=-2.0,
        direct_audio_mix=True,
        disagreement_gate=True,
    )


class EmotionModel:
    """Keep modern text, frame-audio, and recurrent fusion models resident."""

    def __init__(self, checkpoint: Path | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "emotion inference requires the packages in research/experiments/requirements.txt"
            ) from error

        checkpoint = checkpoint or CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frame-attention checkpoint not found: {checkpoint}")
        if not TEXT_MODEL.is_dir():
            raise FileNotFoundError(f"text checkpoint not found: {TEXT_MODEL}")
        acoustic_normalization = checkpoint.parent.parent / ACOUSTIC_NORMALIZATION_FILENAME
        if not acoustic_normalization.is_file():
            raise FileNotFoundError(
                f"speaker acoustic normalization not found: {acoustic_normalization}"
            )

        with np.load(acoustic_normalization) as values:
            self.absolute_acoustic_stats = (
                values["absolute_mean"].astype(np.float32),
                values["absolute_std"].astype(np.float32),
            )
            self.final_acoustic_stats = (
                values["final_mean"].astype(np.float32),
                values["final_std"].astype(np.float32),
            )
        if self.absolute_acoustic_stats[0].shape != (98,) or self.absolute_acoustic_stats[1].shape != (98,):
            raise ValueError("speaker absolute acoustic normalization must be 98-dimensional")
        if self.final_acoustic_stats[0].shape != (197,) or self.final_acoustic_stats[1].shape != (197,):
            raise ValueError("speaker final acoustic normalization must be 197-dimensional")

        self.torch = torch
        self.device = self._select_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL, local_files_only=True
        )
        self.text_model = (
            AutoModelForSequenceClassification.from_pretrained(
                TEXT_MODEL, local_files_only=True
            )
            .to(self.device)
            .eval()
        )
        from huggingface_hub import snapshot_download

        emotion2vec_path = snapshot_download(
            "emotion2vec/emotion2vec_plus_base", local_files_only=True
        )
        self.frame_extractor = Emotion2VecExtractor(
            emotion2vec_path, "cpu"
        )
        self.model = FrameAttentionRecurrentModel(
            text_dimension=768,
            frame_dimension=768,
            acoustic_dimension=197,
            number_of_classes=len(EMOTION_LABELS),
            args=_model_arguments(),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.model.eval()
        self.turns: list[dict[str, np.ndarray]] = []
        self.transcripts: list[str] = []
        self.acoustic_history: list[np.ndarray] = []
        self.lock = threading.Lock()

    @staticmethod
    def _select_device(torch):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def predict(self, text: str, encoded_audio: bytes) -> dict[str, object]:
        waveform = self._decode_audio(encoded_audio)
        frames = self.frame_extractor(waveform).astype(np.float32, copy=False)
        with self.lock:
            acoustic_features = self._extract_speaker_acoustics(waveform)
            embedding, text_logits = self._encode_text(
                text, self.transcripts[-2:]
            )
            turn = {
                "text_embedding": embedding,
                "text_logits": text_logits,
                "emotion_frames": frames,
                "acoustic_features": acoustic_features,
            }
            self.turns.append(turn)
            self.transcripts.append(text)
            batch = self._make_batch(self.turns)
            with self.torch.inference_mode():
                output = self.model(batch)
                fused = self.torch.softmax(output["logits"][0, -1], dim=-1)
                text_only = self.torch.softmax(
                    output["text_logits"][0, -1], dim=-1
                )
                audio_only = self.torch.softmax(
                    output["audio_logits"][0, -1], dim=-1
                )
                audio_mix = float(output["audio_gate"][0, -1, 0].item())
                audio_effects = output["audio_influence"][0, -1]
        result = self._top_prediction(fused)
        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "text": self._top_prediction(text_only),
            "audio": self._top_prediction(audio_only),
            "fused": result,
            "audio_gate": audio_mix,
            "audio_mix": audio_mix,
            "audio_logit_change": {
                label: float(audio_effects[index].item())
                for index, label in enumerate(EMOTION_LABELS)
            },
        }

    def _extract_speaker_acoustics(self, waveform: np.ndarray) -> np.ndarray:
        absolute = sanitize_acoustic_features(
            np.concatenate(
                (
                    extract_egemaps(waveform),
                    summarize_sequence(extract_prosody_contours(waveform)),
                )
            )
        )
        absolute_mean, absolute_std = self.absolute_acoustic_stats
        if self.acoustic_history:
            prior = np.stack(self.acoustic_history)
            center = prior.mean(axis=0)
            scale = prior.std(axis=0) if len(prior) > 1 else absolute_std
            scale = np.where(scale < 1e-6, absolute_std, scale)
        else:
            center, scale = absolute_mean, absolute_std
        relative = sanitize_acoustic_features((absolute - center) / scale)
        combined = np.concatenate(
            (
                normalize(absolute, self.absolute_acoustic_stats),
                relative,
                np.asarray([np.log1p(len(self.acoustic_history))], dtype=np.float32),
            )
        )
        self.acoustic_history.append(absolute)
        return normalize(combined, self.final_acoustic_stats)

    @staticmethod
    def _top_prediction(probabilities) -> dict[str, object]:
        index = int(probabilities.argmax().item())
        return {
            "emotion": EMOTION_LABELS[index],
            "confidence": float(probabilities[index].item()),
        }

    def _encode_text(
        self, text: str, context: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        parts = []
        if context:
            parts.append(
                "Context:\n" + "\n".join(f"[User] {turn}" for turn in context)
            )
        parts.append(f"Current:\n[User] {text}")
        formatted = "\n".join(parts)
        encoded = self.tokenizer(
            formatted,
            truncation=True,
            max_length=256,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        special = encoded.pop("special_tokens_mask").bool()
        current_start = formatted.rfind("Current:\n") + len("Current:\n")
        mask = (
            offsets[:, :, 0].ge(current_start)
            & offsets[:, :, 1].gt(offsets[:, :, 0])
            & encoded["attention_mask"].bool()
            & ~special
        ).to(self.device)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.text_model(
                **inputs, output_hidden_states=True, return_dict=True
            )
            hidden = output.hidden_states[-1]
            count = mask.sum(dim=1, keepdim=True)
            embedding = (hidden * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
            embedding = self.torch.where(count.eq(0), hidden[:, 0], embedding)
        return (
            embedding[0].float().cpu().numpy(),
            output.logits[0].float().cpu().numpy(),
        )

    def _make_batch(self, turns: list[dict[str, np.ndarray]]):
        torch = self.torch
        count = len(turns)
        maximum_frames = max(len(turn["emotion_frames"]) for turn in turns)
        batch = {
            "text_embeddings": torch.zeros(1, count, 768),
            "text_logits": torch.zeros(1, count, len(EMOTION_LABELS)),
            "acoustic_features": torch.zeros(1, count, 197),
            "emotion_frames": torch.zeros(1, count, maximum_frames, 768),
            "frame_mask": torch.zeros(1, count, maximum_frames, dtype=torch.bool),
            "speaker_indices": torch.zeros(1, count, dtype=torch.long),
            "valid_mask": torch.ones(1, count, dtype=torch.bool),
        }
        for index, turn in enumerate(turns):
            frame_count = len(turn["emotion_frames"])
            batch["text_embeddings"][0, index] = torch.from_numpy(
                turn["text_embedding"]
            )
            batch["text_logits"][0, index] = torch.from_numpy(turn["text_logits"])
            batch["acoustic_features"][0, index] = torch.from_numpy(
                turn["acoustic_features"]
            )
            batch["emotion_frames"][0, index, :frame_count] = torch.from_numpy(
                turn["emotion_frames"]
            )
            batch["frame_mask"][0, index, :frame_count] = True
        return {key: value.to(self.device) for key, value in batch.items()}

    def _decode_audio(self, encoded_audio: bytes) -> np.ndarray:
        try:
            import av
        except ImportError as error:
            raise RuntimeError("audio decoding requires av") from error

        samples = []
        with av.open(io.BytesIO(encoded_audio)) as container:
            if not container.streams.audio:
                raise ValueError("recording has no audio stream")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    samples.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                samples.append(converted.to_ndarray().reshape(-1))
        if not samples:
            raise ValueError("decoded recording was empty")
        waveform = np.concatenate(samples).astype(np.float32, copy=False)
        return waveform[: 16000 * 6]
