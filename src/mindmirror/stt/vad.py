"""Voice Activity Detection (VAD) engine and pipeline component for speech segmentation and barge-in."""

import asyncio
from collections import deque
import logging
import time
from typing import Any
import numpy as np

from mindmirror.audio.playback import PlaybackState
from mindmirror.config import (
    INITIAL_NOISE_FLOOR,
    MIN_NOISE_FLOOR,
    SILENCE_THRESHOLD_MULTIPLIER,
    SPEECH_THRESHOLD_MULTIPLIER,
)
from mindmirror.pipeline.component import PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class VADEngine:
    """Classifies audio chunks into speech vs silence based on adaptive background noise estimation."""

    def __init__(self) -> None:
        self.noise_window = deque(maxlen=100)
        self.noise_window.append(INITIAL_NOISE_FLOOR)

    def process_chunk(self, audio_chunk: np.ndarray, adapt: bool = True):
        """Processes an audio chunk.

        Returns: (is_speech, is_silence, rms, noise_floor)
        adapt: If False, threshold is calculated but noise floor is NOT updated.
        """
        # 1. Calculate Energy
        rms = float(np.std(audio_chunk)) if len(audio_chunk) > 0 else 0.0
        if rms < 1e-7:
            return False, True, 0.0, self.get_noise_floor()

        # 2. Update Background Noise Model
        if adapt:
            self.noise_window.append(rms)

        noise_floor = self.get_noise_floor()

        # 3. Decision
        speech_thresh = max(noise_floor * SPEECH_THRESHOLD_MULTIPLIER, MIN_NOISE_FLOOR * 2)
        silence_thresh = max(noise_floor * SILENCE_THRESHOLD_MULTIPLIER, MIN_NOISE_FLOOR)

        is_speech = rms > speech_thresh
        is_silence = rms < silence_thresh

        return is_speech, is_silence, rms, noise_floor

    def get_noise_floor(self) -> float:
        return float(max(np.percentile(list(self.noise_window), 10), MIN_NOISE_FLOOR))


class VADComponent(PipelineComponent):
    """Supervised pipeline component that consumes audio.raw, detects speech segments,

    publishes segmented audio to vad.speech_audio, and triggers control.interrupt on barge-in.
    """

    def __init__(
        self,
        bus: MessageBus,
        vad_engine: VADEngine | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        silence_duration: float = 0.5,
        preroll_duration: float = 0.2,
        min_speech_duration: float = 0.2,
    ) -> None:
        super().__init__(name="vad_engine", bus=bus)
        self.vad_engine = vad_engine or VADEngine()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.preroll_duration = preroll_duration
        self.min_speech_duration = min_speech_duration

        chunk_duration = chunk_size / sample_rate
        self.required_silence_chunks = max(1, int(silence_duration / chunk_duration))
        preroll_chunk_count = max(1, int(preroll_duration / chunk_duration))

        self._preroll_buffer: deque[bytes] = deque(maxlen=preroll_chunk_count)
        self._speech_buffer: list[bytes] = []
        self._is_speaking = False
        self._silence_chunks = 0
        self._subscription_id: str | None = None

    def set_silence_duration(self, duration_s: float) -> None:
        """Dynamically updates the required silence pause threshold for speech segmentation."""
        self.silence_duration = max(0.1, float(duration_s))
        chunk_duration = self.chunk_size / self.sample_rate
        self.required_silence_chunks = max(1, int(self.silence_duration / chunk_duration))
        logger.info("VAD silence_duration updated to %.2fs (%d chunks).", self.silence_duration, self.required_silence_chunks)

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("audio.raw", self._handle_raw_audio)
        self.create_subscription_task("speaker.identified", self._handle_speaker_identified)
        self.create_subscription_task("control.speaker_override", self._handle_speaker_override)
        await asyncio.sleep(0.01)
        logger.info("VADComponent started (listening to audio.raw and speaker.identified).")

    async def stop(self) -> None:
        self._speech_buffer.clear()
        self._preroll_buffer.clear()
        await super().stop()
        logger.info("VADComponent stopped.")

    async def _handle_speaker_identified(self, message: PipelineMessage) -> None:
        """Adapts VAD pause threshold when speaker profile provides vad_silence_duration."""
        payload = message.payload
        if isinstance(payload, dict):
            profile_dict = payload.get("profile")
            if isinstance(profile_dict, dict) and "vad_silence_duration" in profile_dict:
                self.set_silence_duration(profile_dict["vad_silence_duration"])
            elif "vad_silence_duration" in payload:
                self.set_silence_duration(payload["vad_silence_duration"])

    async def _handle_speaker_override(self, message: PipelineMessage) -> None:
        """Handles manual/runtime speaker parameter overrides."""
        payload = message.payload
        if isinstance(payload, dict) and "vad_silence_duration" in payload:
            self.set_silence_duration(payload["vad_silence_duration"])

    async def _handle_raw_audio(self, message: PipelineMessage) -> None:
        payload = message.payload
        if isinstance(payload, bytes):
            pcm_bytes = payload
            # Convert 16-bit PCM to float normalized [-1, 1]
            audio_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(payload, np.ndarray):
            audio_chunk = payload.astype(np.float32)
            if audio_chunk.max() > 1.0 or audio_chunk.min() < -1.0:
                audio_chunk = audio_chunk / 32768.0
            pcm_bytes = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        else:
            return

        is_speech, is_silence, rms, noise_floor = self.vad_engine.process_chunk(
            audio_chunk, adapt=not self._is_speaking
        )

        playback_state = PlaybackState.get_instance()

        # Acoustic Barge-In detection:
        # If user speaks while playback is active, trigger atomic interruption
        if playback_state.is_playback_active and is_speech:
            logger.info("Acoustic barge-in detected! Emitting control.interrupt (rms=%.4f).", rms)
            await self.bus.publish(
                "control.interrupt",
                PipelineMessage(
                    channel="control.interrupt",
                    phase=StreamPhase.CONTROL,
                    payload={"reason": "barge_in", "rms": rms},
                    trace_id=message.trace_id,
                ),
            )
            await playback_state.set_playback_active(False)
            await playback_state.set_mic_muted(False)

        # Standard Speech Segmentation State Machine
        if is_speech:
            self._silence_chunks = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_buffer.extend(list(self._preroll_buffer))
            self._speech_buffer.append(pcm_bytes)
        elif self._is_speaking:
            self._speech_buffer.append(pcm_bytes)
            if is_silence:
                self._silence_chunks += 1
            else:
                self._silence_chunks = 0

            if self._silence_chunks >= self.required_silence_chunks:
                # Speech ended!
                full_audio_bytes = b"".join(self._speech_buffer)
                duration_s = (len(full_audio_bytes) // 2) / self.sample_rate

                if duration_s >= self.min_speech_duration:
                    logger.debug("Emitting segmented speech audio (%.2fs) to vad.speech_audio.", duration_s)
                    await self.bus.publish(
                        "vad.speech_audio",
                        PipelineMessage(
                            channel="vad.speech_audio",
                            phase=StreamPhase.AUDIO,
                            payload=full_audio_bytes,
                            trace_id=message.trace_id,
                            metadata={
                                "sample_rate": self.sample_rate,
                                "duration_s": duration_s,
                                "rms": rms,
                            },
                        ),
                    )

                self._speech_buffer.clear()
                self._is_speaking = False
                self._silence_chunks = 0
        else:
            self._preroll_buffer.append(pcm_bytes)