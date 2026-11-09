"""Audio capture pipeline component streaming raw PCM frames to the message bus."""

import asyncio
import logging
from typing import Any
import numpy as np

from mindmirror.audio.devices import safe_open_stream
from mindmirror.audio.playback import PlaybackState
from mindmirror.pipeline.component import ComponentHealth, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import AudioSource
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class LiveAudioSource(AudioSource):
    """Streams live audio frames from PortAudio/sounddevice."""

    def __init__(self, device_id: int | None = None, sample_rate: int = 16000, chunk_size: int = 1024) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None
        self._is_running = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio input warning: %s", status)
        # Convert float32 or int16 to int16 bytes
        if indata.dtype == np.float32 or indata.dtype == np.float64:
            pcm = (indata[:, 0] * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        else:
            pcm = indata[:, 0].astype(np.int16).tobytes()

        if self._loop and self._is_running:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm)

    async def stream_frames(self):
        import sounddevice as sd
        self._loop = asyncio.get_running_loop()
        self._is_running = True
        try:
            self._stream = safe_open_stream(
                self.device_id,
                self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            while self._is_running:
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    yield frame
                except asyncio.TimeoutError:
                    continue
        finally:
            self._is_running = False
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None


class LiveAudioCapture(PipelineComponent):
    """Supervised audio capture component streaming PCM audio chunks to audio.raw."""

    def __init__(
        self,
        bus: MessageBus,
        source: AudioSource | None = None,
        device_id: int | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        headphones_mode: bool = False,
    ) -> None:
        super().__init__(name="audio_capture", bus=bus)
        self.source = source or LiveAudioSource(
            device_id=device_id,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        )
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.headphones_mode = headphones_mode
        self._capture_task: asyncio.Task | None = None
        self._control_sub_id: str | None = None
        self._muted: bool = False

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("control.mute_mic", self._handle_control_message)
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info("LiveAudioCapture component started (headphones_mode=%s).", self.headphones_mode)

    async def stop(self) -> None:
        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        await super().stop()
        logger.info("LiveAudioCapture component stopped.")

    async def _handle_control_message(self, message: PipelineMessage) -> None:
        if message.channel == "control.mute_mic" and isinstance(message.payload, dict):
            self._muted = bool(message.payload.get("muted", False))
            logger.debug("LiveAudioCapture mute state updated to: %s", self._muted)

    async def _capture_loop(self) -> None:
        """Reads audio frames from the source and publishes to audio.raw."""
        try:
            async for frame in self.source.stream_frames():
                if not self._is_running:
                    break

                # If mic is muted and not in headphones mode, drop or skip frame to prevent echo
                playback_state = PlaybackState.get_instance()
                is_muted = self._muted or playback_state.is_mic_muted

                # In barge-in mode (headphones or interruption enabled), we still pass frames
                # to audio.raw so VAD can detect user speech, but we flag in metadata
                meta = {
                    "sample_rate": self.sample_rate,
                    "is_mic_muted": is_muted,
                    "playback_active": playback_state.is_playback_active,
                }

                if is_muted and not self.headphones_mode and playback_state.is_playback_active:
                    # Echo prevention: discard frame while speaker is blaring if not headphones
                    continue

                await self.bus.publish(
                    "audio.raw",
                    PipelineMessage(
                        channel="audio.raw",
                        phase=StreamPhase.AUDIO,
                        payload=frame,
                        metadata=meta,
                    ),
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.record_error(e)
            self._status = ComponentHealth.FAILED
            logger.error("Error in audio capture loop: %s", e)
