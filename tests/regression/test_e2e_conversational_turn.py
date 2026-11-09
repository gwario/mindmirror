"""Golden Scenario GS1: Full Conversational Turn (Happy Path) glass-to-glass regression test.

Chains every pipeline component with probes and assertions in between each stage:
AudioSource ➔ LiveAudioCapture ➔ VADComponent ➔ STTComponent & SpeakerIdentifier
➔ ReasoningComponent ➔ TTSComponent ➔ PlaybackLoop ➔ VirtualAudioSink
➔ Telemetry (metrics.latency)

Asserts:
1. audio.raw receives PCM audio chunks from AudioSource.
2. vad.speech_audio receives segmented speech with audio metadata.
3. stt.transcript receives transcribed string in real-time.
4. speaker.identified identifies confirmed speaker and updates SpeakerContext.
5. ttt.thinking & reasoning.trace emit cognitive deliberation deltas.
6. ttt.response emits style-tagged vocal response ([STYLE] tag).
7. audio.playback emits synthesized PCM frames with style & speaker metadata.
8. VirtualAudioSink buffers synthesized audio without requiring a physical soundcard.
9. metrics.latency publishes end-to-end LatencyReport meeting the sub-1000ms SLA.
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
from mindmirror.memory.chroma_store import ChromaMemoryManager
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.telemetry import LatencyReport, TurnSpanTracker
from mindmirror.pipeline.testing.harness import (
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


class MockGS1STTEngine(STTInterface):
    """Mock STT engine generating deterministic transcript for synthetic speech."""

    def load_model(self) -> None:
        pass

    def transcribe(self, audio_data: np.ndarray | bytes, sample_rate: int = 16000) -> str:
        return "Can you explain the MindMirror pipeline architecture?"


class MockGS1TTSEngine(TTSInterface):
    """Mock TTS engine generating deterministic audio chunks for synthesized text."""

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


class MockGS1ReasoningStrategy(ReasoningStrategy):
    """Emits intermediate reasoning step, thinking delta, and style-tagged response."""

    async def reason_and_respond(self, prompt: str, bus, trace_id: str = "", speaker_id: str | None = None) -> str:
        # Step 1: Discrete reasoning step
        await bus.publish(
            "reasoning.trace",
            PipelineMessage(
                channel="reasoning.trace",
                phase=StreamPhase.REASONING_STEP,
                payload={"step": "retrieved_architecture_spec", "confidence": 0.99},
                trace_id=trace_id,
            ),
        )
        # Step 2: Internal thinking
        await bus.publish(
            "ttt.thinking",
            PipelineMessage(
                channel="ttt.thinking",
                phase=StreamPhase.THINKING,
                payload="Analyzing multi-component pipeline graph...",
                trace_id=trace_id,
            ),
        )
        # Step 3: Vocalised response with [EXCITED] style tag
        response = "[EXCITED] MindMirror is a modular, event-driven voice AI architecture with full supervision!"
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload=response,
                trace_id=trace_id,
                speaker_id=speaker_id,
            ),
        )
        return response


@pytest.mark.asyncio
async def test_gs1_full_conversational_turn_end_to_end(tmp_path: Path):
    """GS1: Exercises the complete glass-to-glass pipeline with inter-component assertions."""
    bus = LocalBus()
    virtual_sink = VirtualAudioSink()
    playback_state = PlaybackState(bus=bus)

    # Isolated temporary persistence
    db_dir = tmp_path / "gs1_db"
    profile_store = SpeakerProfileStore(db_dir=db_dir / "profiles")
    registry = SpeakerRegistry(db_dir=db_dir / "registry")
    memory_manager = ChromaMemoryManager(db_dir=db_dir / "memories")
    await memory_manager.attach_bus(bus)
    speaker_context = SpeakerContext(bus=bus, profile_store=profile_store)

    # 1. Enroll Mario in Profile Store & Voice Registry
    profile_mario = SpeakerProfile(
        speaker_id="mario",
        name="Mario",
        vad_silence_duration=0.8,
        tts_voice_id="en_GB-alan-medium",
        tts_pitch=1.05,
    )
    profile_store.save_profile(profile_mario)

    # Synthetic reference voice audio
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    mario_voice_pcm = (np.sin(2 * np.pi * 230.0 * t) * 16000).astype(np.int16).tobytes()
    registry.enroll_speaker_audio("mario", mario_voice_pcm, name="Mario")

    # 2. Inter-Component Probe Message Baskets
    probes = {
        "audio.raw": [],
        "vad.speech_audio": [],
        "stt.transcript": [],
        "speaker.identified": [],
        "reasoning.trace": [],
        "ttt.thinking": [],
        "ttt.response": [],
        "audio.playback": [],
        "metrics.latency": [],
    }

    async def _make_probe(channel: str):
        async for msg in bus.subscribe(channel):
            probes[channel].append(msg)

    probe_tasks = [asyncio.create_task(_make_probe(ch)) for ch in probes]

    # 3. Instantiate All Supervised Pipeline Components
    # Generate synthetic input: 0.3s speech + 0.3s silence
    speech_pcm = mario_voice_pcm
    silence_pcm = (np.zeros(16000 // 2)).astype(np.int16).tobytes()
    full_audio_pcm = speech_pcm + silence_pcm

    synth_source = SyntheticAudioSource(
        pcm_bytes=full_audio_pcm,
        chunk_size=1024,
        sample_rate=16000,
        frame_delay=0.001,  # 100x fast-forward simulation
    )

    capture_component = LiveAudioCapture(bus=bus, source=synth_source, sample_rate=16000)
    vad_component = VADComponent(bus=bus, sample_rate=16000, silence_duration=0.15, min_speech_duration=0.1)
    stt_component = STTComponent(bus=bus, stt_engine=MockGS1STTEngine())
    speaker_identifier = SpeakerIdentifierComponent(
        bus=bus,
        registry=registry,
        profile_store=profile_store,
        speaker_context=speaker_context,
        similarity_threshold=0.6,
    )
    reasoning_component = ReasoningComponent(
        bus=bus,
        strategy=MockGS1ReasoningStrategy(),
        speaker_context=speaker_context,
    )
    tts_component = TTSComponent(
        bus=bus,
        tts_engine=MockGS1TTSEngine(chunk_count=4),
        profile_store=profile_store,
        speaker_context=speaker_context,
    )
    playback_loop = PlaybackLoop(bus=bus, sink=virtual_sink, sample_rate=16000)

    # Start all components in pipeline order
    # Start consumers first, then capture audio source last
    await playback_loop.start()
    await tts_component.start()
    await reasoning_component.start()
    await speaker_identifier.start()
    await stt_component.start()
    await vad_component.start()
    await asyncio.sleep(0.05)
    await capture_component.start()

    # Allow simulation to flow through the entire pipeline
    for _ in range(60):
        if (
            len(probes["audio.playback"]) >= 4
            and len(probes["metrics.latency"]) >= 1
            and len(probes["speaker.identified"]) >= 1
        ):
            break
        await asyncio.sleep(0.05)

    # =========================================================================
    # INTER-COMPONENT ASSERTIONS (Zero Hardware, Glass-to-Glass Contracts)
    # =========================================================================

    # 1. Capture ➔ VAD: audio.raw frames received
    assert len(probes["audio.raw"]) > 0, "Assertion Failed: audio.raw did not receive frames from capture"
    assert isinstance(probes["audio.raw"][0].payload, bytes)
    assert len(probes["audio.raw"][0].payload) == 1024

    # 2. VAD ➔ STT & Diarization: vad.speech_audio segmented speech emitted
    assert len(probes["vad.speech_audio"]) >= 1, "Assertion Failed: vad.speech_audio did not receive speech segment"
    vad_msg = probes["vad.speech_audio"][0]
    assert len(vad_msg.payload) > 0
    assert "duration_s" in vad_msg.metadata
    assert vad_msg.metadata["duration_s"] >= 0.1

    # 3. STT ➔ Reasoning: stt.transcript transcribed text
    assert len(probes["stt.transcript"]) >= 1, "Assertion Failed: stt.transcript did not receive transcript"
    stt_msg = probes["stt.transcript"][0]
    assert stt_msg.payload == "Can you explain the MindMirror pipeline architecture?"

    # 4. Diarization ➔ SpeakerContext: speaker.identified emitted and matched Mario
    assert len(probes["speaker.identified"]) >= 1, "Assertion Failed: speaker.identified was not emitted"
    spk_msg = probes["speaker.identified"][0]
    assert spk_msg.payload.get("speaker_id") == "mario"
    assert spk_msg.payload.get("confidence") > 0.6
    assert speaker_context.is_resolved is True
    assert speaker_context.current_speaker_id == "mario"

    # 5. Cognitive channels: discrete reasoning step & thinking tokens received
    assert len(probes["reasoning.trace"]) >= 1, "Assertion Failed: reasoning.trace was not published"
    assert probes["reasoning.trace"][0].payload["step"] == "retrieved_architecture_spec"

    assert len(probes["ttt.thinking"]) >= 1, "Assertion Failed: ttt.thinking was not published"
    assert "Analyzing" in str(probes["ttt.thinking"][0].payload)

    # 6. Reasoning ➔ TTS: ttt.response with [EXCITED] style tag
    assert len(probes["ttt.response"]) >= 1, "Assertion Failed: ttt.response was not published"
    ttt_msg = probes["ttt.response"][0]
    assert ttt_msg.payload.startswith("[EXCITED]")

    # 7. TTS ➔ Playback: audio.playback contains synthesized frames with parsed style & text
    assert len(probes["audio.playback"]) >= 4, "Assertion Failed: audio.playback did not receive all chunks"
    pb_msg = probes["audio.playback"][0]
    assert pb_msg.metadata.get("style") == "EXCITED"
    assert "MindMirror" in pb_msg.metadata.get("text", "")

    # Turn 2: With Mario now identified, verify successive turns adopt Mario's personal voice
    await bus.publish(
        "ttt.response",
        PipelineMessage(
            channel="ttt.response",
            phase=StreamPhase.RESPONDING,
            payload="[NEUTRAL] Verified speaker profile adoption.",
            speaker_id="mario",
        ),
    )
    for _ in range(30):
        if len(probes["audio.playback"]) >= 8:
            break
        await asyncio.sleep(0.02)

    assert len(probes["audio.playback"]) >= 8
    pb2_msg = probes["audio.playback"][-1]
    assert pb2_msg.metadata.get("speaker_id") == "mario"
    assert pb2_msg.metadata.get("voice_id") == "en_GB-alan-medium"
    assert pb2_msg.metadata.get("pitch") == 1.05

    # 8. Playback ➔ VirtualAudioSink: sink contains buffered audio
    buffered_bytes = virtual_sink.get_buffered_audio()
    assert len(buffered_bytes) > 0, "Assertion Failed: VirtualAudioSink buffer is empty"
    assert virtual_sink.total_samples() > 0

    # 9. Telemetry: metrics.latency published
    assert len(probes["metrics.latency"]) >= 1, "Assertion Failed: metrics.latency did not publish report"
    report = probes["metrics.latency"][0].payload
    assert isinstance(report, dict)
    assert report["total_glass_to_glass_ms"] > 0
    assert report["total_glass_to_glass_ms"] < 1000.0, f"SLA violated: {report['total_glass_to_glass_ms']}ms > 1000ms"

    # Clean up
    for pt in probe_tasks:
        pt.cancel()
    await playback_loop.stop()
    await tts_component.stop()
    await reasoning_component.stop()
    await speaker_identifier.stop()
    await stt_component.stop()
    await vad_component.stop()
    await capture_component.stop()
    await speaker_context.close()
    await memory_manager.close()
    await bus.close()
