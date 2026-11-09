"""Audio playback state, acoustic muting coordination, and supervised PlaybackLoop."""

import asyncio
import logging
import time
from typing import Any
import numpy as np

from mindmirror.pipeline.component import PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import AudioSink
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class PlaybackState:
    """Coordinates playback activity and microphone muting without file-based locks.

    Uses in-memory asyncio events and bus control signals (control.mute_mic, control.playback_active).
    """

    _instance: "PlaybackState | None" = None

    def __init__(self, bus: MessageBus | None = None) -> None:
        self.bus = bus
        self._playback_active_event = asyncio.Event()
        self._mic_muted_event = asyncio.Event()
        self._playback_active: bool = False
        self._mic_muted: bool = False
        self._lock = asyncio.Lock()
        PlaybackState._instance = self

    @classmethod
    def get_instance(cls) -> "PlaybackState":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = PlaybackState()
        return cls._instance

    @property
    def is_playback_active(self) -> bool:
        """True if the system is currently playing synthesised speech through the audio sink."""
        return self._playback_active

    @property
    def is_mic_muted(self) -> bool:
        """True if the microphone should be muted to prevent acoustic echo feedback."""
        return self._mic_muted

    async def set_playback_active(self, active: bool) -> None:
        """Update playback state and optionally broadcast to the message bus."""
        async with self._lock:
            self._playback_active = active
            if active:
                self._playback_active_event.set()
            else:
                self._playback_active_event.clear()

        if self.bus:
            await self.bus.publish(
                "control.playback_active",
                PipelineMessage(
                    channel="control.playback_active",
                    phase=StreamPhase.CONTROL,
                    payload={"active": active},
                ),
            )

    async def set_mic_muted(self, muted: bool) -> None:
        """Update microphone mute state and optionally broadcast to the message bus."""
        async with self._lock:
            self._mic_muted = muted
            if muted:
                self._mic_muted_event.set()
            else:
                self._mic_muted_event.clear()

        if self.bus:
            await self.bus.publish(
                "control.mute_mic",
                PipelineMessage(
                    channel="control.mute_mic",
                    phase=StreamPhase.CONTROL,
                    payload={"muted": muted},
                ),
            )

    async def wait_for_playback_end(self, timeout: float | None = None) -> None:
        """Wait until playback has completed."""
        if not self._playback_active:
            return
        while self._playback_active:
            await asyncio.sleep(0.01)


class LiveAudioSink(AudioSink):
    """Outputs PCM audio chunks to hardware soundcard using sounddevice."""

    def __init__(self, device_id: int | None = None, sample_rate: int = 16000) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self._stream = None
        self._buffered: list[bytes] = []

    def _ensure_stream(self) -> None:
        import sounddevice as sd
        if self._stream is None:
            self._stream = sd.OutputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            self._stream.start()

    async def play_frame(self, frame: bytes) -> None:
        self._buffered.append(frame)
        self._ensure_stream()
        audio_array = np.frombuffer(frame, dtype=np.int16)
        await asyncio.to_thread(self._stream.write, audio_array)

    def get_buffered_audio(self) -> bytes:
        return b"".join(self._buffered)

    def clear(self) -> None:
        self._buffered.clear()

    def drain(self) -> None:
        self._buffered.clear()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.start()
            except Exception:
                pass


class PlaybackLoop(PipelineComponent):
    """Supervised component consuming audio.playback, managing audio sinks and handling acoustic barge-in."""

    def __init__(
        self,
        bus: MessageBus,
        sink: AudioSink | None = None,
        sample_rate: int = 16000,
        device_id: int | None = None,
        name: str = "audio_playback",
    ) -> None:
        super().__init__(name=name, bus=bus)
        self.sink = sink or LiveAudioSink(device_id=device_id, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._playback_sub_id: str | None = None
        self._interrupt_sub_id: str | None = None
        self._loop_task: asyncio.Task | None = None
        self._interrupted = False
        self._last_drain_duration_ms: float = 0.0

    @property
    def last_drain_duration_ms(self) -> float:
        return self._last_drain_duration_ms

    async def start(self) -> None:
        await super().start()
        # Initialize PlaybackState singleton with bus
        state = PlaybackState.get_instance()
        state.bus = self.bus

        self.create_subscription_task("audio.playback", self._handle_playback_frame)
        self.create_subscription_task("control.interrupt", self._handle_interrupt)
        self._loop_task = asyncio.create_task(self._playback_worker())
        await asyncio.sleep(0.01)
        logger.info("PlaybackLoop component '%s' started.", self.name)

    async def stop(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        await super().stop()
        logger.info("PlaybackLoop component '%s' stopped.", self.name)

    async def _handle_playback_frame(self, message: PipelineMessage) -> None:
        if self._interrupted:
            return

        frame = message.payload
        if isinstance(frame, bytes):
            pcm = frame
        elif isinstance(frame, np.ndarray):
            pcm = frame.astype(np.int16).tobytes()
        else:
            return

        state = PlaybackState.get_instance()
        if not state.is_playback_active:
            await state.set_playback_active(True)
            await state.set_mic_muted(True)

        await self._queue.put(pcm)

    async def _handle_interrupt(self, message: PipelineMessage) -> None:
        """Handles acoustic barge-in interruption: drains audio queue in < 30ms and clears mute lock."""
        start_time = time.perf_counter()
        self._interrupted = True

        # 1. Drain pending playback queue immediately
        drained_chunks = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                drained_chunks += 1
            except asyncio.QueueEmpty:
                break

        # 2. Drain hardware/virtual sink buffer
        self.sink.drain()

        # 3. Clear active playback state and microphone mute lock
        state = PlaybackState.get_instance()
        await state.set_playback_active(False)
        await state.set_mic_muted(False)

        drain_duration_ms = (time.perf_counter() - start_time) * 1000
        self._last_drain_duration_ms = drain_duration_ms
        logger.info(
            "Barge-in handled: %d chunks discarded, sink drained in %.2f ms (SLA < 30ms)",
            drained_chunks,
            drain_duration_ms,
        )
        self._interrupted = False

    async def _playback_worker(self) -> None:
        """Processes and plays queued audio frames."""
        state = PlaybackState.get_instance()
        while self._is_running:
            try:
                try:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if self._queue.empty() and state.is_playback_active and not self._interrupted:
                        # Playback finished naturally
                        await state.set_playback_active(False)
                        await state.set_mic_muted(False)
                    continue

                if self._interrupted:
                    continue

                res = self.sink.play_frame(frame)
                if asyncio.iscoroutine(res):
                    await res
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.record_error(e)
                logger.error("Error in playback worker: %s", e)
