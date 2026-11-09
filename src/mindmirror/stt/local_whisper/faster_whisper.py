"""Faster-Whisper STT implementation using CTranslate2 and int8 quantization."""

import io
import logging
import tempfile
import numpy as np
import soundfile as sf
import torch

from mindmirror.stt.interface import STTInterface

logger = logging.getLogger(__name__)


class FasterWhisperSTT(STTInterface):
    """High-performance local STT engine using faster-whisper (CTranslate2).

    Supports int8 quantization on CPU and float16 on GPU (CUDA).
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str | None = None,
        compute_type: str | None = None,
        log_queue=None,
    ) -> None:
        self.model_size = model_size
        self.log_queue = log_queue
        self.model = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

    def load_model(self) -> None:
        """Loads the faster-whisper model into memory with configured quantization."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading FasterWhisper model '%s' on %s (%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        if self.log_queue:
            self.log_queue.put({
                "type": "info",
                "text": f"Loading FasterWhisper ({self.model_size}) on {self.device} ({self.compute_type})...",
            })

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info("FasterWhisper model loaded successfully.")
        if self.log_queue:
            self.log_queue.put({"type": "success", "text": "✅ FasterWhisper Model Loaded."})

    def transcribe(self, audio_data: np.ndarray | bytes, sample_rate: int = 16000) -> str:
        """Transcribes single-channel audio using faster-whisper."""
        if not self.model:
            self.load_model()

        try:
            if isinstance(audio_data, bytes):
                # 16-bit PCM bytes to float32
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype in (np.float32, np.float64):
                    samples = audio_data.astype(np.float32)
                else:
                    samples = audio_data.astype(np.float32) / 32768.0
            else:
                return ""

            if len(samples) == 0:
                return ""

            # faster_whisper accepts float32 numpy arrays directly
            segments, info = self.model.transcribe(
                samples,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )

            transcript = " ".join([segment.text for segment in segments]).strip()
            return transcript

        except Exception as e:
            logger.error("FasterWhisper transcription failed: %s", e)
            if self.log_queue:
                self.log_queue.put({"type": "error", "text": f"FasterWhisper error: {e}"})
            return ""
