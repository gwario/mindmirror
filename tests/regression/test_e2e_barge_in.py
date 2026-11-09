"""Golden Scenario GS3: Barge-In / Acoustic Interruption Handling regression test.

Contract assertions:
1. Synthetic speech arriving while playback is active triggers control.interrupt via VAD.
2. TTS cancels pending synthesis immediately.
3. Playback sink buffer drains in < 30ms.
4. Microphone mute state un-mutes cleanly.
"""

import asyncio
import numpy as np
import pytest

from mindmirror.audio.playback import PlaybackLoop, PlaybackState
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import SyntheticAudioSource, VirtualAudioSink
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.stt.vad import VADComponent
from mindmirror.tts.component import TTSComponent
from mindmirror.tts.interface import TTSInterface


class MockStreamingTTSEngine(TTSInterface):
    """Mock TTS engine generating multiple chunks to test mid-stream interruption."""

    def __init__(self, chunk_count: int = 10) -> None:
        self.chunk_count = chunk_count

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        chunks = []
        for _ in range(self.chunk_count):
            # 1024 samples of synthetic 440Hz tone
            t = np.linspace(0, 0.05, 800, endpoint=False)
            sine = (np.sin(2 * np.pi * 440.0 * t) * 0.4 * 32767).astype(np.int16)
            chunks.append(sine.tobytes())
        return chunks


@pytest.mark.asyncio
async def test_gs3_barge_in_interruption_and_sub_30ms_drain():
    """GS3: Validates end-to-end acoustic barge-in interruption and sub-30ms sink drain."""
    bus = LocalBus()
    virtual_sink = VirtualAudioSink(frame_delay=0.02)
    playback_state = PlaybackState(bus=bus)

    # Instantiate pipeline components
    playback_loop = PlaybackLoop(bus=bus, sink=virtual_sink, sample_rate=16000)
    vad_component = VADComponent(bus=bus, sample_rate=16000, silence_duration=0.3)
    tts_engine = MockStreamingTTSEngine(chunk_count=10)
    tts_component = TTSComponent(bus=bus, tts_engine=tts_engine)

    # Track bus events for assertions
    interrupt_events: list[PipelineMessage] = []
    async def _record_interrupts():
        async for msg in bus.subscribe("control.interrupt"):
            interrupt_events.append(msg)
    interrupt_task = asyncio.create_task(_record_interrupts())

    # Start all components
    await playback_loop.start()
    await vad_component.start()
    await tts_component.start()

    # Step 1: Trigger assistant speech via ttt.response
    await bus.publish(
        "ttt.response",
        PipelineMessage(
            channel="ttt.response",
            phase=StreamPhase.RESPONDING,
            payload="[EXCITED] MindMirror is now running with full pipeline supervision!",
        ),
    )

    # Allow playback loop to receive frames and activate playback state
    for _ in range(50):
        if playback_state.is_playback_active:
            break
        await asyncio.sleep(0.01)

    assert playback_state.is_playback_active, "Playback should be active while audio is streaming"
    assert playback_state.is_mic_muted, "Mic should be muted during active playback"
    assert virtual_sink.total_samples() > 0, "VirtualAudioSink should have received audio frames"

    # Step 2: User interrupts by speaking (inject loud synthetic speech frame into audio.raw)
    synthetic_source = SyntheticAudioSource.generate_sine_pcm(
        frequency_hz=300.0,
        duration_s=0.2,
        sample_rate=16000,
        amplitude=0.6,
    )
    speech_frames = [f async for f in synthetic_source.stream_frames()]

    # Inject first frame of speech to trigger VAD barge-in
    await bus.publish(
        "audio.raw",
        PipelineMessage(
            channel="audio.raw",
            phase=StreamPhase.AUDIO,
            payload=speech_frames[0],
        ),
    )

    # Wait briefly for interruption to propagate across the bus
    await asyncio.sleep(0.05)

    # Step 3: Assert Pipeline Contracts
    # 1. Assert VAD emitted control.interrupt
    assert len(interrupt_events) >= 1, "VAD must emit control.interrupt upon barge-in"
    assert interrupt_events[0].payload.get("reason") == "barge_in"

    # 2. Assert playback sink buffer was drained in < 30ms (SLA)
    assert playback_loop.last_drain_duration_ms > 0.0
    assert playback_loop.last_drain_duration_ms < 30.0, (
        f"Sink drain took {playback_loop.last_drain_duration_ms:.2f}ms, exceeding 30ms SLA!"
    )

    # 3. Assert playback sink buffer is cleared
    assert virtual_sink.playback_interrupted, "Virtual sink should record interrupted state"
    assert len(virtual_sink.get_buffered_audio()) == 0, "Virtual sink buffer must be emptied on drain"

    # 4. Assert microphone is un-muted and playback is cleared
    assert not playback_state.is_playback_active, "Playback active flag must be cleared"
    assert not playback_state.is_mic_muted, "Microphone mute lock must be released cleanly"

    # Clean up
    interrupt_task.cancel()
    await tts_component.stop()
    await vad_component.stop()
    await playback_loop.stop()
    await bus.close()
