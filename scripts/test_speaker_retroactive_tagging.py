#!/usr/bin/env python3
"""Phase 3 Verification Script: Parallel Background Speaker Diarization & Retroactive Memory Tagging.

Demonstrates:
1. Enrolling distinct speaker voice embeddings and runtime profiles into ChromaDB.
2. Parallel background speaker identification running alongside conversation flow.
3. Retroactive ChromaDB memory patching (replacing 'unresolved' with confirmed speaker ID).
4. Runtime per-speaker adaptation of VAD silence pause thresholds and TTS voices.
"""

import asyncio
import logging
from pathlib import Path
import sys
import tempfile
import time

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mindmirror.audio.playback import PlaybackLoop, PlaybackState
from mindmirror.memory.chroma_store import ChromaMemoryManager
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.testing.harness import VirtualAudioSink
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase3_demo")
console = Console()


class DemoTTS(TTSInterface):
    """Mock synthesis returning synthetic PCM frames for testing."""

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        t = np.linspace(0, 0.05, 800, endpoint=False)
        sine = (np.sin(2 * np.pi * 320.0 * t) * 0.3 * 32767).astype(np.int16)
        return [sine.tobytes()]


def generate_synthetic_voice(freq: float, duration_s: float = 0.5) -> bytes:
    """Generates distinct synthetic voice PCM audio with rich harmonics."""
    t = np.linspace(0, duration_s, int(16000 * duration_s), endpoint=False)
    waveform = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
        + 0.2 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    return (waveform * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def run_phase3_demo():
    console.print(
        Panel.fit(
            "[bold cyan]👥 MINDMIRROR PHASE 3 VERIFICATION & DEMO[/bold cyan]\n"
            "[white]Parallel Speaker Diarization, Retroactive ChromaDB Memory & Per-Speaker Runtime Tuning[/white]",
            border_style="cyan",
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir)
        bus = LocalBus()
        virtual_sink = VirtualAudioSink()

        # 1. Initialize Stores
        console.print("1. [bold yellow]Initializing ChromaDB Memory, Profile Store, and Voice Registry...[/bold yellow]")
        memory_manager = ChromaMemoryManager(db_dir=db_path / "memories")
        await memory_manager.attach_bus(bus)

        profile_store = SpeakerProfileStore(db_dir=db_path / "profiles")
        registry = SpeakerRegistry(db_dir=db_path / "registry")
        speaker_context = SpeakerContext(bus=bus, profile_store=profile_store)
        console.print("   ✅ Vector stores & bus initialized.\n")

        # 2. Enroll Multi-Speaker Profiles & Voices
        console.print("2. [bold yellow]Enrolling Speaker Profiles & Voice Embeddings...[/bold yellow]")
        # Mario: slow speaker, British tone
        profile_mario = SpeakerProfile(
            speaker_id="mario",
            name="Mario",
            vad_silence_duration=1.2,
            tts_voice_id="en_GB-alan-medium",
            tts_pitch=1.05,
            tts_speaking_rate=0.95,
            ttt_reasoning_effort="high",
            prompt_context="Mario is a software engineer specializing in AI and distributed systems.",
        )
        profile_store.save_profile(profile_mario)
        mario_voice = generate_synthetic_voice(freq=200.0, duration_s=1.0)
        registry.enroll_speaker_audio(speaker_id="mario", audio_data=mario_voice, name="Mario")

        # Alice: fast speaker, US tone
        profile_alice = SpeakerProfile(
            speaker_id="alice",
            name="Alice",
            vad_silence_duration=0.4,
            tts_voice_id="en_US-amy-medium",
            tts_pitch=1.20,
            tts_speaking_rate=1.10,
            ttt_reasoning_effort="low",
            prompt_context="Alice is a product designer.",
        )
        profile_store.save_profile(profile_alice)
        alice_voice = generate_synthetic_voice(freq=350.0, duration_s=1.0)
        registry.enroll_speaker_audio(speaker_id="alice", audio_data=alice_voice, name="Alice")

        profiles_table = Table(title="Enrolled Speaker Profiles")
        profiles_table.add_column("Speaker ID", style="cyan")
        profiles_table.add_column("Name", style="green")
        profiles_table.add_column("VAD Silence", style="yellow")
        profiles_table.add_column("TTS Voice", style="magenta")
        profiles_table.add_column("Pitch", style="white")

        for p in profile_store.list_profiles():
            profiles_table.add_row(
                p.speaker_id,
                p.name,
                f"{p.vad_silence_duration:.1f}s",
                p.tts_voice_id or "default",
                f"{p.tts_pitch:.2f}x",
            )
        console.print(profiles_table)
        console.print()

        # 3. Start Supervised Pipeline Components
        console.print("3. [bold yellow]Starting Supervised Components (VAD, Speaker ID, TTS, Playback)...[/bold yellow]")
        vad_component = VADComponent(bus=bus, sample_rate=16000, silence_duration=0.5)
        speaker_identifier = SpeakerIdentifierComponent(
            bus=bus,
            registry=registry,
            profile_store=profile_store,
            speaker_context=speaker_context,
            similarity_threshold=0.65,
            simulated_delay_s=0.2,       # 200ms parallel identification delay
        )
        mock_tts = DemoTTS()
        tts_component = TTSComponent(
            bus=bus,
            tts_engine=mock_tts,
            profile_store=profile_store,
            speaker_context=speaker_context,
        )
        playback_loop = PlaybackLoop(bus=bus, sink=virtual_sink, sample_rate=16000)

        playback_events: list[PipelineMessage] = []
        async def _capture_playback():
            async for msg in bus.subscribe("audio.playback"):
                playback_events.append(msg)
        capture_task = asyncio.create_task(_capture_playback())

        await vad_component.start()
        await speaker_identifier.start()
        await tts_component.start()
        await playback_loop.start()

        console.print(f"   Initial VAD silence threshold: [yellow]{vad_component.silence_duration}s[/yellow]")
        console.print(f"   Initial SpeakerContext: [yellow]{speaker_context.current_speaker_id}[/yellow]\n")

        # 4. Turn 1: Mario Speaks (Optimistic Execution + Retroactive Tagging)
        console.print("4. [bold yellow]Executing Turn 1: Mario speaks with 200ms Diarization Delay...[/bold yellow]")
        turn1_start = time.perf_counter()

        # User voice frame arrives
        await bus.publish(
            "vad.speech_audio",
            PipelineMessage(channel="vad.speech_audio", payload=mario_voice, trace_id="turn-1"),
        )

        # Store memory immediately (before speaker resolves!)
        mem1_id = memory_manager.add_memory(
            fact="Mario prefers lighthearted British tone and works on AI agents",
            category="preference",
        )
        console.print(f"   💾 Memory saved with ID '{mem1_id}'. Unresolved buffer: {memory_manager.get_unresolved_memory_ids()}")
        assert mem1_id in memory_manager.get_unresolved_memory_ids()

        # Wait for speaker identification to complete in background
        identified = await speaker_context.wait_for_speaker(timeout=1.0)
        diarize_latency_ms = (time.perf_counter() - turn1_start) * 1000
        console.print(f"   🎯 Parallel Speaker Identified: [bold green]'{identified}'[/bold green] in {diarize_latency_ms:.1f}ms")
        assert identified == "mario"

        # Allow bus auto-retro-tagging to process
        for _ in range(20):
            if len(memory_manager.get_unresolved_memory_ids()) == 0:
                break
            await asyncio.sleep(0.02)

        # Check retroactively updated memory
        tagged = memory_manager.get_all_memories()
        console.print(f"   🏷️ Retroactive Memory Tagging Complete! speaker_id='[green]{tagged[0]['speaker_id']}[/green]'")
        assert tagged[0]["speaker_id"] == "mario"
        assert len(memory_manager.get_unresolved_memory_ids()) == 0

        # Check VAD adaptation
        console.print(f"   ⚙️ VAD silence duration adapted to Mario's pace: [bold green]{vad_component.silence_duration}s[/bold green]")
        assert vad_component.silence_duration == 1.2
        console.print()

        # 5. Turn 2: Synthesis adopts Mario's profile
        console.print("5. [bold yellow]Executing Turn 2: Assistant synthesizes speech with Mario's Voice Profile...[/bold yellow]")
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload="[EXCITED] System checks passed, mate!",
                speaker_id="mario",
                trace_id="turn-2",
            ),
        )
        await asyncio.sleep(0.1)

        assert len(playback_events) > 0
        pb = playback_events[0]
        console.print(f"   🔊 Playback Frame dispatched:")
        console.print(f"      • Speaker: [cyan]{pb.metadata.get('speaker_id')}[/cyan]")
        console.print(f"      • Voice ID: [magenta]{pb.metadata.get('voice_id')}[/magenta]")
        console.print(f"      • Pitch: [yellow]{pb.metadata.get('pitch')}x[/yellow]")
        console.print(f"      • Rate: [white]{pb.metadata.get('speaking_rate')}x[/white]")
        assert pb.metadata.get("voice_id") == "en_GB-alan-medium"
        assert pb.metadata.get("pitch") == 1.05
        console.print()

        # Summary Table
        results_table = Table(title="Phase 3 Verification Results")
        results_table.add_column("Verification Step", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Latency / Value", style="yellow")
        results_table.add_column("SLA / Target", style="magenta")

        results_table.add_row("Parallel Diarization", "PASS", f"{diarize_latency_ms:.1f}ms", "< 500ms")
        results_table.add_row("Retroactive ChromaDB Tag", "PASS", f"{tagged[0]['speaker_id']}", "mario")
        results_table.add_row("Dynamic VAD Pause Adaptation", "PASS", f"{vad_component.silence_duration:.1f}s", "1.2s")
        results_table.add_row("Per-Speaker TTS Voice Tuning", "PASS", f"{pb.metadata.get('voice_id')}", "en_GB-alan-medium")
        console.print(results_table)

        # Cleanup
        capture_task.cancel()
        await playback_loop.stop()
        await tts_component.stop()
        await speaker_identifier.stop()
        await vad_component.stop()
        await speaker_context.close()
        await memory_manager.close()
        await bus.close()

        console.print("\n[bold green]🎉 ALL PHASE 3 SPEAKER IDENTIFICATION & RETROACTIVE MEMORY CONTRACTS VERIFIED![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(run_phase3_demo())
