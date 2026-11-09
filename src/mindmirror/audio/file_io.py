"""File-based AudioSource and AudioSink for hardware-free testing, simulations, and recordings."""

import asyncio
import logging
from pathlib import Path
import time
from typing import Optional
import numpy as np

from mindmirror.pipeline.testing.harness import AudioSink, AudioSource

logger = logging.getLogger(__name__)


class FileAudioSource(AudioSource):
    """Feeds audio frames directly from a WAV or audio file into the pipeline.

    Enables testing without PortAudio or physical soundcard hardware.
    Supports real-time pacing or fast-forward simulation.
    """

    def __init__(
        self,
        file_path: str | Path,
        target_sample_rate: int = 16000,
        chunk_size: int = 1024,
        realtime: bool = True,
        loop: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size
        self.realtime = realtime
        self.loop = loop
        self._pcm_bytes: bytes = b""
        self._load_and_resample()

    def _load_and_resample(self) -> None:
        """Reads audio file and normalizes to target mono 16-bit PCM."""
        import soundfile as sf

        data, original_sr = sf.read(str(self.file_path))

        # Downmix to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Resample if sample rates differ
        if original_sr != self.target_sample_rate:
            from scipy import signal
            num_target_samples = int(len(data) * self.target_sample_rate / original_sr)
            data = signal.resample(data, num_target_samples)

        # Convert float to int16 PCM bytes
        if data.dtype in (np.float32, np.float64):
            int16_samples = (data * 32767.0).clip(-32768, 32767).astype(np.int16)
        else:
            int16_samples = data.astype(np.int16)

        self._pcm_bytes = int16_samples.tobytes()
        logger.info(
            "Loaded FileAudioSource from '%s' (%d bytes, %.2fs @ %dHz)",
            self.file_path.name,
            len(self._pcm_bytes),
            len(int16_samples) / self.target_sample_rate,
            self.target_sample_rate,
        )

    async def stream_frames(self):
        """Asynchronously streams PCM frames chunk by chunk."""
        chunk_byte_len = self.chunk_size * 2  # 2 bytes per 16-bit sample
        delay_per_frame = (self.chunk_size / self.target_sample_rate) if self.realtime else 0.001

        while True:
            offset = 0
            while offset < len(self._pcm_bytes):
                chunk = self._pcm_bytes[offset : offset + chunk_byte_len]
                offset += chunk_byte_len
                yield chunk
                if delay_per_frame > 0:
                    await asyncio.sleep(delay_per_frame)

            if not self.loop:
                break


class FileAudioSink(AudioSink):
    """AudioSink capturing pipeline playback frames and writing them to a WAV file."""

    def __init__(
        self,
        output_path: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[bytes] = []

    async def play_frame(self, frame: bytes) -> None:
        """Buffers playback frame asynchronously."""
        self._frames.append(frame)

    def play_chunk(self, chunk: bytes) -> None:
        """Synchronously buffers playback frame."""
        self._frames.append(chunk)

    def get_buffered_audio(self) -> bytes:
        """Returns aggregated audio bytes captured so far."""
        return b"".join(self._frames)

    def drain(self, timeout: float = 1.0) -> None:
        """Drains and flushes buffered audio to WAV file."""
        self.save()

    def save(self) -> Path:
        """Writes accumulated PCM frames to a standard WAV file."""
        import soundfile as sf

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        all_pcm = b"".join(self._frames)
        samples = np.frombuffer(all_pcm, dtype=np.int16)

        sf.write(str(self.output_path), samples, self.sample_rate)
        logger.info(
            "Saved FileAudioSink output to '%s' (%d samples, %.2fs)",
            self.output_path,
            len(samples),
            len(samples) / self.sample_rate,
        )
        return self.output_path

    def clear(self) -> None:
        """Discards buffered frames without writing."""
        self._frames.clear()
