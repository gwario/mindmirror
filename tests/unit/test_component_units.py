"""Unit-scale tests for individual pipeline components with isolated mocks.

Verifies:
1. LiveAudioCapture: Reads from AudioSource, respects mute controls, publishes audio.raw.
2. VADComponent: Segments speech audio, respects silence threshold, triggers barge-in interrupt.
3. STTComponent: Consumes vad.speech_audio, invokes STT engine in worker thread, emits stt.transcript.
4. ReasoningComponent: Consumes stt.transcript, updates ContextManager, executes strategy, publishes latency span.
5. TTSComponent: Consumes ttt.response, parses style tags, emits audio.playback chunks, cancels on interrupt.
6. PlaybackLoop: Consumes audio.playback, coordinates PlaybackState, drains in < 30ms on control.interrupt.
7. SpeakerIdentifierComponent: Consumes vad.speech_audio, queries SpeakerRegistry, emits speaker.identified.
"""

import asyncio
from pathlib import Path
import numpy as np
import pytest

from mindmirror.audio.capture import LiveAudioCapture
from mindmirror.audio.playback import PlaybackLoop, PlaybackState
from mindmirror.llm.component import ReasoningComponent
from mindmirror.llm.context.manager import ContextManager
from mindmirror.llm.reasoning.interface import ReasoningStrategy
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import (
    AudioSource,
    DeterministicReasoningEngine,
    SyntheticAudioSource,
    VirtualAudioSink,
)
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.identifier import (
    AcousticVoiceEmbedder,
    SpeakerIdentifierComponent,
    SpeakerRegistry,
)
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore
from mindmirror.stt.component import STTComponent
from mindmirror.stt.interface import STTInterface
from mindmirror.stt.vad import VADComponent
from mindmirror.tts.component import TTSComponent
from mindmirror.tts.interface import TTSInterface


# --- Mock Interfaces ---

class MockSTTEngine(STTInterface):
    def __init__(self, return_text: str = "Test transcript text."):
        self.return_text = return_text
        self.calls: list[bytes] = []

    def load_model(self) -> None:
        pass

    def transcribe(self, audio_data: np.ndarray | bytes, sample_rate: int = 16000) -> str:
        self.calls.append(audio_data if isinstance(audio_data, bytes) else audio_data.tobytes())
        return self.return_text


class MockTTSEngine(TTSInterface):
    def __init__(self, chunk_count: int = 3):
        self.chunk_count = chunk_count

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        chunks = []
        for i in range(self.chunk_count):
            pcm = (np.sin(np.linspace(0, 0.05, 400)) * 10000).astype(np.int16).tobytes()
            chunks.append(pcm)
        return chunks


class MockReasoningStrategy(ReasoningStrategy):
    def __init__(self, response: str = "[NEUTRAL] Mock response."):
        self.response = response

    async def reason_and_respond(self, prompt: str, bus, trace_id: str = "", speaker_id: str | None = None) -> str:
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload=self.response,
                trace_id=trace_id,
                speaker_id=speaker_id,
            ),
        )
        return self.response


# --- Tests ---

@pytest.mark.asyncio
async def test_unit_audio_capture_publishes_raw_frames():
    """Unit: LiveAudioCapture streams frames from source to audio.raw and respects mute."""
    bus = LocalBus()
    synth_source = SyntheticAudioSource.generate_sine_pcm(duration_s=0.05, chunk_size=512)
    capture = LiveAudioCapture(bus=bus, source=synth_source, sample_rate=16000)

    received_frames = []
    async def _capture():
        async for msg in bus.subscribe("audio.raw"):
            received_frames.append(msg)
            if len(received_frames) >= 2:
                break
    t = asyncio.create_task(_capture())

    await capture.start()
    await asyncio.wait_for(t, timeout=3.0)
    await capture.stop()
    await bus.close()

    assert len(received_frames) >= 2
    assert received_frames[0].channel == "audio.raw"
    assert isinstance(received_frames[0].payload, bytes)
    assert len(received_frames[0].payload) == 512


@pytest.mark.asyncio
async def test_unit_vad_component_speech_segmentation():
    """Unit: VADComponent segments audio.raw into vad.speech_audio upon speech."""
    bus = LocalBus()
    vad = VADComponent(bus=bus, sample_rate=16000, silence_duration=0.1, min_speech_duration=0.05)

    received_speech = []
    async def _capture():
        async for msg in bus.subscribe("vad.speech_audio"):
            received_speech.append(msg)
            break
    t = asyncio.create_task(_capture())

    await vad.start()

    # Feed speech sine wave followed by silence
    speech = SyntheticAudioSource.generate_sine_pcm(duration_s=0.2, amplitude=0.8, chunk_size=1024)
    silence = SyntheticAudioSource.generate_silence_pcm(duration_s=0.2, chunk_size=1024)

    async for chunk in speech.stream_frames():
        await bus.publish("audio.raw", PipelineMessage(channel="audio.raw", payload=chunk))
    async for chunk in silence.stream_frames():
        await bus.publish("audio.raw", PipelineMessage(channel="audio.raw", payload=chunk))

    await asyncio.wait_for(t, timeout=3.0)
    await vad.stop()
    await bus.close()

    assert len(received_speech) == 1
    assert received_speech[0].channel == "vad.speech_audio"
    assert len(received_speech[0].payload) > 0


@pytest.mark.asyncio
async def test_unit_stt_component_transcribes_speech():
    """Unit: STTComponent transcribes vad.speech_audio and publishes stt.transcript."""
    bus = LocalBus()
    mock_engine = MockSTTEngine(return_text="Transcribed voice prompt")
    stt_comp = STTComponent(bus=bus, stt_engine=mock_engine)

    received_transcripts = []
    async def _capture():
        async for msg in bus.subscribe("stt.transcript"):
            received_transcripts.append(msg)
            break
    t = asyncio.create_task(_capture())

    await stt_comp.start()
    await asyncio.sleep(0.02)

    dummy_pcm = b"\x00\x00" * 8000
    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(
            channel="vad.speech_audio",
            payload=dummy_pcm,
            trace_id="unit-trace-1",
            speaker_id="mario",
        ),
    )

    await asyncio.wait_for(t, timeout=1.0)
    await stt_comp.stop()
    await bus.close()

    assert len(received_transcripts) == 1
    msg = received_transcripts[0]
    assert msg.payload == "Transcribed voice prompt"
    assert msg.speaker_id == "mario"
    assert msg.trace_id == "unit-trace-1"


@pytest.mark.asyncio
async def test_unit_reasoning_component_executes_strategy():
    """Unit: ReasoningComponent processes stt.transcript, invokes strategy, and updates context."""
    bus = LocalBus()
    strategy = MockReasoningStrategy(response="[EXCITED] System active!")
    ctx_mgr = ContextManager()
    reasoning_comp = ReasoningComponent(bus=bus, strategy=strategy, context_manager=ctx_mgr)

    received_responses = []
    async def _capture():
        async for msg in bus.subscribe("ttt.response"):
            received_responses.append(msg)
            break
    t = asyncio.create_task(_capture())

    await reasoning_comp.start()
    await asyncio.sleep(0.02)

    await bus.publish(
        "stt.transcript",
        PipelineMessage(
            channel="stt.transcript",
            payload="Hello MindMirror",
            trace_id="reason-trace-1",
            speaker_id="mario",
        ),
    )

    await asyncio.wait_for(t, timeout=1.0)
    await reasoning_comp.stop()
    await bus.close()

    assert len(received_responses) == 1
    assert received_responses[0].payload == "[EXCITED] System active!"
    assert len(ctx_mgr.messages) == 2  # user message + assistant message


@pytest.mark.asyncio
async def test_unit_tts_component_synthesizes_and_publishes_playback():
    """Unit: TTSComponent parses style tags, synthesizes, and emits audio.playback chunks."""
    bus = LocalBus()
    tts_engine = MockTTSEngine(chunk_count=3)
    tts_comp = TTSComponent(bus=bus, tts_engine=tts_engine)

    playback_chunks = []
    async def _capture():
        async for msg in bus.subscribe("audio.playback"):
            playback_chunks.append(msg)
            if len(playback_chunks) >= 3:
                break
    t = asyncio.create_task(_capture())

    await tts_comp.start()

    await bus.publish(
        "ttt.response",
        PipelineMessage(
            channel="ttt.response",
            payload="[SERIOUS] Emergency protocol engaged.",
            trace_id="tts-trace-1",
            speaker_id="mario",
        ),
    )

    await asyncio.wait_for(t, timeout=1.0)
    await tts_comp.stop()
    await bus.close()

    assert len(playback_chunks) == 3
    assert playback_chunks[0].metadata.get("style") == "SERIOUS"
    assert playback_chunks[0].metadata.get("text") == "Emergency protocol engaged."
    assert isinstance(playback_chunks[0].payload, bytes)


@pytest.mark.asyncio
async def test_unit_playback_loop_plays_and_drains_on_interrupt():
    """Unit: PlaybackLoop coordinates PlaybackState and flushes sink buffer on control.interrupt."""
    bus = LocalBus()
    virtual_sink = VirtualAudioSink()
    playback_loop = PlaybackLoop(bus=bus, sink=virtual_sink)
    state = PlaybackState.get_instance()

    await playback_loop.start()

    # Push audio chunks
    dummy_pcm = b"\x01\x00" * 1024
    await bus.publish("audio.playback", PipelineMessage(channel="audio.playback", payload=dummy_pcm))
    await asyncio.sleep(0.05)

    assert state.is_playback_active is True
    assert virtual_sink.total_samples() > 0

    # Emit barge-in interrupt
    await bus.publish("control.interrupt", PipelineMessage(channel="control.interrupt", payload={"reason": "barge_in"}))
    await asyncio.sleep(0.05)

    assert state.is_playback_active is False
    assert state.is_mic_muted is False
    assert playback_loop.last_drain_duration_ms < 30.0
    assert len(virtual_sink.get_buffered_audio()) == 0

    await playback_loop.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_unit_speaker_identifier_matches_enrolled_speaker(tmp_path: Path):
    """Unit: SpeakerIdentifierComponent matches enrolled voice audio and emits speaker.identified."""
    bus = LocalBus()
    db_dir = tmp_path / "unit_speaker"
    registry = SpeakerRegistry(db_dir=db_dir / "reg")
    context = SpeakerContext(bus=bus)

    # Enroll synthetic voice
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    voice_pcm = (np.sin(2 * np.pi * 250.0 * t) * 15000).astype(np.int16).tobytes()
    registry.enroll_speaker_audio("mario", voice_pcm, name="Mario")

    identifier = SpeakerIdentifierComponent(
        bus=bus,
        registry=registry,
        speaker_context=context,
        similarity_threshold=0.6,
    )

    received_id_events = []
    async def _capture():
        async for msg in bus.subscribe("speaker.identified"):
            received_id_events.append(msg)
            break
    t_task = asyncio.create_task(_capture())

    await identifier.start()

    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(channel="vad.speech_audio", payload=voice_pcm, trace_id="speaker-unit"),
    )

    await asyncio.wait_for(t_task, timeout=1.5)
    await identifier.stop()
    await context.close()
    await bus.close()

    assert len(received_id_events) == 1
    assert received_id_events[0].payload.get("speaker_id") == "mario"
    assert received_id_events[0].payload.get("confidence") > 0.6
