"""Golden Scenario GS5: Parallel Background Speaker Diarization & Retroactive Memory regression test.

Contract assertions:
1. Speech turn begins; STT and Memory proceed optimistically while speaker ID is resolving.
2. Memory created during turn is stored with speaker_id="unresolved".
3. Background speaker identification arrives asynchronously (~300-400ms later) emitting speaker.identified.
4. Memory store catches event and automatically retroactively patches memory metadata in ChromaDB to confirmed speaker.
5. VAD dynamically adopts speaker's custom pause threshold (vad_silence_duration).
6. Successive turns (Turn N+1) adopt resolved speaker profile (TTS voice, pitch, speaking rate).
"""

import asyncio
from pathlib import Path
import numpy as np
import pytest

from mindmirror.audio.playback import PlaybackLoop, PlaybackState
from mindmirror.memory.chroma_store import ChromaMemoryManager
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import SyntheticAudioSource, VirtualAudioSink
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.identifier import (
    AcousticVoiceEmbedder,
    SpeakerIdentifierComponent,
    SpeakerRegistry,
)
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore
from mindmirror.stt.vad import VADComponent
from mindmirror.tts.component import TTSComponent
from mindmirror.tts.interface import TTSInterface


class MockTTSVoiceEngine(TTSInterface):
    """Mock TTS engine generating speech chunks."""

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        t = np.linspace(0, 0.05, 800, endpoint=False)
        sine = (np.sin(2 * np.pi * 300.0 * t) * 0.3 * 32767).astype(np.int16)
        return [sine.tobytes()]


def generate_speaker_voice_pcm(freq: float = 240.0, duration_s: float = 0.5) -> bytes:
    """Generates synthetic distinctive voice PCM audio for speaker enrollment and matching."""
    t = np.linspace(0, duration_s, int(16000 * duration_s), endpoint=False)
    # Rich harmonic voice tone
    harmonics = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
        + 0.2 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    return (harmonics * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@pytest.mark.asyncio
async def test_gs5_parallel_speaker_identification_and_retroactive_memory(tmp_path: Path):
    """GS5: Full end-to-end regression scenario verifying parallel identification,

    retroactive ChromaDB memory tagging, and dynamic runtime parameter tuning.
    """
    bus = LocalBus()
    virtual_sink = VirtualAudioSink()
    playback_state = PlaybackState(bus=bus)

    # 1. Initialize Memory, Profile Store, and Voice Registry in isolated temporary directory
    db_dir = tmp_path / "gs5_data"
    memory_manager = ChromaMemoryManager(db_dir=db_dir / "memories")
    await memory_manager.attach_bus(bus)

    profile_store = SpeakerProfileStore(db_dir=db_dir / "profiles")
    registry = SpeakerRegistry(db_dir=db_dir / "registry")
    speaker_context = SpeakerContext(bus=bus, profile_store=profile_store)

    # 2. Enroll Speaker Profile & Voice Audio for "mario"
    mario_profile = SpeakerProfile(
        speaker_id="mario",
        name="Mario",
        vad_silence_duration=1.2,        # Slow pause threshold (1.2s vs default 0.5s)
        tts_voice_id="en_GB-alan-medium",
        tts_pitch=1.15,
        tts_speaking_rate=0.95,
        ttt_reasoning_effort="high",
        prompt_context="Mario is a software engineer specializing in AI and distributed systems.",
    )
    profile_store.save_profile(mario_profile)

    mario_voice_audio = generate_speaker_voice_pcm(freq=220.0, duration_s=1.0)
    registry.enroll_speaker_audio(speaker_id="mario", audio_data=mario_voice_audio, name="Mario")

    # 3. Instantiate and Start Supervised Pipeline Components
    vad_component = VADComponent(bus=bus, sample_rate=16000, silence_duration=0.5)
    speaker_identifier = SpeakerIdentifierComponent(
        bus=bus,
        registry=registry,
        profile_store=profile_store,
        speaker_context=speaker_context,
        similarity_threshold=0.65,
        simulated_delay_s=0.3,           # Simulated 300ms parallel identification delay
    )
    mock_tts = MockTTSVoiceEngine()
    tts_component = TTSComponent(
        bus=bus,
        tts_engine=mock_tts,
        profile_store=profile_store,
        speaker_context=speaker_context,
    )
    playback_loop = PlaybackLoop(bus=bus, sink=virtual_sink, sample_rate=16000)

    # Track captured playback messages on the bus
    playback_messages: list[PipelineMessage] = []
    async def _capture_playback():
        async for msg in bus.subscribe("audio.playback"):
            playback_messages.append(msg)
    capture_task = asyncio.create_task(_capture_playback())

    await vad_component.start()
    await speaker_identifier.start()
    await tts_component.start()
    await playback_loop.start()

    # Initial state assertions
    assert vad_component.silence_duration == 0.5, "VAD should initialize with default 0.5s pause threshold"
    assert not speaker_context.is_resolved, "SpeakerContext should start as unresolved"

    # =========================================================================
    # TURN 1: Speech audio arrives; Memory recorded before Speaker ID resolves
    # =========================================================================

    # Publish Mario's voice chunk to vad.speech_audio (triggers parallel speaker identification)
    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(
            channel="vad.speech_audio",
            phase=StreamPhase.AUDIO,
            payload=mario_voice_audio,
            trace_id="turn-1-mario",
        ),
    )

    # Immediately (while speaker_identifier is still processing in background during its 300ms delay),
    # an incoming fact is saved to persistent memory
    mem_id = memory_manager.add_memory(
        fact="User is testing parallel speaker diarization with ChromaDB vector store",
        category="work",
    )

    # Contract assertion: Memory was saved optimistically as unresolved
    assert mem_id in memory_manager.get_unresolved_memory_ids(), (
        "Memory must initially be buffered under 'unresolved' speaker ID"
    )
    initial_memories = memory_manager.get_all_memories()
    assert len(initial_memories) == 1
    assert initial_memories[0]["speaker_id"] == "unresolved"

    # Wait for background speaker identification (~300ms) to resolve and propagate
    confirmed_speaker = await speaker_context.wait_for_speaker(timeout=1.5)

    assert confirmed_speaker == "mario", f"Expected speaker 'mario', got '{confirmed_speaker}'"
    assert speaker_context.is_resolved, "SpeakerContext must be marked resolved"

    # Allow bus subscribers to process auto-retroactive tagging
    for _ in range(30):
        if len(memory_manager.get_unresolved_memory_ids()) == 0:
            break
        await asyncio.sleep(0.02)

    # Contract assertions: Retroactive tagging verified
    assert len(memory_manager.get_unresolved_memory_ids()) == 0, (
        "Buffered unresolved memories should be empty after retroactive tagging"
    )
    tagged_memories = memory_manager.get_all_memories()
    assert len(tagged_memories) == 1
    assert tagged_memories[0]["speaker_id"] == "mario", (
        f"Memory must be retroactively tagged as 'mario', but was {tagged_memories[0]['speaker_id']}"
    )

    # Contract assertions: VAD silence threshold dynamically updated to Mario's 1.2s pace
    assert vad_component.silence_duration == 1.2, (
        f"VAD silence duration must update to 1.2s, currently {vad_component.silence_duration}"
    )

    # =========================================================================
    # TURN 2: Successive Turn adopts resolved Speaker Profile in TTS Synthesis
    # =========================================================================

    # Assistant vocalises response for Mario
    await bus.publish(
        "ttt.response",
        PipelineMessage(
            channel="ttt.response",
            phase=StreamPhase.RESPONDING,
            payload="[EXCITED] Welcome back Mario! How can I assist you with voice AI today?",
            speaker_id="mario",
            trace_id="turn-2-mario",
        ),
    )

    # Wait for playback frames
    for _ in range(30):
        if len(playback_messages) > 0:
            break
        await asyncio.sleep(0.02)

    assert len(playback_messages) > 0, "audio.playback messages should be published"
    first_playback = playback_messages[0]

    # Contract assertions: TTS metadata reflects Mario's personal voice parameters
    assert first_playback.metadata.get("speaker_id") == "mario"
    assert first_playback.metadata.get("voice_id") == "en_GB-alan-medium", (
        f"TTS voice_id should be 'en_GB-alan-medium', got {first_playback.metadata.get('voice_id')}"
    )
    assert first_playback.metadata.get("pitch") == 1.15
    assert first_playback.metadata.get("speaking_rate") == 0.95

    # Clean up
    capture_task.cancel()
    await playback_loop.stop()
    await tts_component.stop()
    await speaker_identifier.stop()
    await vad_component.stop()
    await speaker_context.close()
    await memory_manager.close()
    await bus.close()
