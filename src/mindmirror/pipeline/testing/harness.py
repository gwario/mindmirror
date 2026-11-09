"""Testing and headless simulation harness for MindMirror pipeline."""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
import logging
from typing import Any
import numpy as np

from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Abstract audio input contract — allows swapping live microphone with synthetic audio."""

    @abstractmethod
    async def stream_frames(self) -> AsyncIterator[bytes]:
        """Yields sequential raw PCM audio byte chunks."""
        ...


class AudioSink(ABC):
    """Abstract audio output contract — allows capturing playback without physical soundcard."""

    @abstractmethod
    async def play_frame(self, frame: bytes) -> None:
        """Plays or buffers an incoming audio byte chunk."""
        ...

    @abstractmethod
    def get_buffered_audio(self) -> bytes:
        """Returns all aggregated audio bytes captured so far."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clears buffered audio."""
        ...

    @abstractmethod
    def drain(self) -> None:
        """Flushes/drains currently buffered playback."""
        ...


class SyntheticAudioSource(AudioSource):
    """Feeds deterministic PCM chunks from golden WAV files or synthetic waveforms.

    Allows fast-forward headless simulation (e.g. 50-100x real-time).
    """

    def __init__(
        self,
        pcm_bytes: bytes,
        chunk_size: int = 1024,
        sample_rate: int = 16000,
        frame_delay: float = 0.001,
    ) -> None:
        self.pcm_bytes = pcm_bytes
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.frame_delay = frame_delay
        self._is_stopped = False

    def stop(self) -> None:
        self._is_stopped = True

    async def stream_frames(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.pcm_bytes), self.chunk_size):
            if self._is_stopped:
                break
            chunk = self.pcm_bytes[i : i + self.chunk_size]
            yield chunk
            if self.frame_delay > 0:
                await asyncio.sleep(self.frame_delay)

    @classmethod
    def generate_sine_pcm(
        cls,
        frequency_hz: float = 440.0,
        duration_s: float = 1.0,
        sample_rate: int = 16000,
        amplitude: float = 0.5,
        chunk_size: int = 1024,
    ) -> "SyntheticAudioSource":
        """Generates a synthetic 16-bit PCM mono sine wave for deterministic testing."""
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        waveform = (np.sin(2 * np.pi * frequency_hz * t) * amplitude * 32767).astype(np.int16)
        pcm_bytes = waveform.tobytes()
        return cls(pcm_bytes=pcm_bytes, chunk_size=chunk_size, sample_rate=sample_rate)

    @classmethod
    def generate_silence_pcm(
        cls,
        duration_s: float = 1.0,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ) -> "SyntheticAudioSource":
        """Generates synthetic silence (zero bytes) for deterministic testing."""
        total_samples = int(sample_rate * duration_s)
        pcm_bytes = np.zeros(total_samples, dtype=np.int16).tobytes()
        return cls(pcm_bytes=pcm_bytes, chunk_size=chunk_size, sample_rate=sample_rate)


class VirtualAudioSink(AudioSink):
    """In-memory audio sink for headless verification of synthesized voice chunks."""

    def __init__(self, frame_delay: float = 0.0) -> None:
        self.frames: list[bytes] = []
        self.playback_interrupted: bool = False
        self.frame_delay = frame_delay
        self._drain_event = asyncio.Event()

    async def play_frame(self, frame: bytes) -> None:
        if not self.playback_interrupted:
            self.frames.append(frame)
            if self.frame_delay > 0:
                await asyncio.sleep(self.frame_delay)

    def get_buffered_audio(self) -> bytes:
        return b"".join(self.frames)

    def total_samples(self) -> int:
        return len(self.get_buffered_audio()) // 2  # 16-bit mono PCM

    def clear(self) -> None:
        self.frames.clear()
        self.playback_interrupted = False

    def drain(self) -> None:
        """Drains audio buffer immediately upon barge-in or completion."""
        self.frames.clear()
        self.playback_interrupted = True
        self._drain_event.set()


class DeterministicReasoningEngine:
    """Mock reasoning engine emitting predictable thinking tokens, tool calls, and style tags."""

    def __init__(
        self,
        canned_response: str = "[NEUTRAL] Test response text.",
        canned_steps: list[dict[str, Any]] | None = None,
        canned_thinking: str = "Deliberating over synthetic input...",
    ) -> None:
        self.canned_response = canned_response
        self.canned_steps = canned_steps or []
        self.canned_thinking = canned_thinking

    async def generate_turn(self, prompt: str, bus: MessageBus, trace_id: str | None = None) -> None:
        # 1. Emit intermediate reasoning steps
        for step in self.canned_steps:
            await bus.publish(
                "reasoning.trace",
                PipelineMessage(
                    channel="reasoning.trace",
                    phase=StreamPhase.REASONING_STEP,
                    payload=step,
                    trace_id=trace_id or "",
                ),
            )

        # 2. Emit internal thinking deltas
        await bus.publish(
            "ttt.thinking",
            PipelineMessage(
                channel="ttt.thinking",
                phase=StreamPhase.THINKING,
                payload=self.canned_thinking,
                trace_id=trace_id or "",
            ),
        )

        # 3. Emit final vocalised response with style tag
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload=self.canned_response,
                trace_id=trace_id or "",
            ),
        )


class MockGS1STTEngine:
    """Fallback/mock STT engine generating deterministic transcripts for synthetic speech."""

    def load_model(self) -> None:
        pass

    def transcribe(self, audio_data: np.ndarray | bytes, sample_rate: int = 16000) -> str:
        return "Can you explain the MindMirror pipeline architecture?"


class MockGS1TTSEngine:
    """Fallback/mock TTS engine generating deterministic PCM audio chunks."""

    def __init__(self, chunk_count: int = 4) -> None:
        self.chunk_count = chunk_count

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        chunks = []
        for _ in range(self.chunk_count):
            pcm = (np.sin(np.linspace(0, 0.05, 500)) * 12000).astype(np.int16).tobytes()
            chunks.append(pcm)
        return chunks

