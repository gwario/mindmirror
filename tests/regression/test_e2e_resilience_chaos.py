"""Golden Scenario 7 (GS7): Component Crash & Supervisor Self-Healing Regression Test.

Validates that when a supervised pipeline component (e.g. STT or TTS) suffers a fatal
crash mid-session, the ComponentSupervisor:
1. Detects the failure promptly.
2. Applies exponential backoff.
3. Automatically respawns and restarts the component cleanly.
4. Preserves subsequent conversational turn processing with zero message corruption.
"""

import asyncio
import numpy as np
import pytest

from mindmirror.pipeline.component import ComponentHealth, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport.local import LocalBus


class CrashableSTTComponent(PipelineComponent):
    """STT component that simulates a fatal crash on command or unhandled exception."""

    def __init__(self, bus: LocalBus, instance_id: int = 1) -> None:
        super().__init__(name="stt", bus=bus)
        self.instance_id = instance_id
        self.processed_count = 0

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("vad.speech_audio", self._on_vad_speech)
        await asyncio.sleep(0.01)

    async def stop(self) -> None:
        await super().stop()

    async def _on_vad_speech(self, message: PipelineMessage) -> None:
        if message.payload == b"CRASH_PAYLOAD":
            self._status = ComponentHealth.FAILED
            self.record_error("Fatal unhandled worker crash")
            self._is_running = False
            return

        self.processed_count += 1
        transcript = f"Transcribed by instance {self.instance_id}: turn {self.processed_count}"
        await self.bus.publish(
            "stt.transcript",
            PipelineMessage(
                channel="stt.transcript",
                phase=StreamPhase.TRANSCRIPT,
                payload=transcript,
                trace_id=message.trace_id,
            ),
        )


@pytest.mark.asyncio
async def test_gs7_supervisor_chaos_and_self_healing():
    """GS7: Verifies supervisor detects crash, respawns STT component, and recovers pipeline."""
    bus = LocalBus()
    supervisor = ComponentSupervisor(check_interval=0.03)

    instance_counter = 0

    def stt_factory():
        nonlocal instance_counter
        instance_counter += 1
        return CrashableSTTComponent(bus=bus, instance_id=instance_counter)

    # Register STT with fast test backoff
    supervisor.register(
        "stt",
        stt_factory,
        min_backoff=0.05,
        max_backoff=0.5,
        backoff_factor=1.5,
    )
    await supervisor.start()

    # Track emitted transcripts
    transcripts: list[PipelineMessage] = []

    async def _transcript_listener():
        async for msg in bus.subscribe("stt.transcript"):
            transcripts.append(msg)

    listener_task = asyncio.create_task(_transcript_listener())
    await asyncio.sleep(0.02)

    # --- Turn 1: Normal processing before crash ---
    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(
            channel="vad.speech_audio",
            phase=StreamPhase.AUDIO,
            payload=b"normal_audio_1",
            trace_id="turn-1",
        ),
    )

    for _ in range(25):
        if len(transcripts) >= 1:
            break
        await asyncio.sleep(0.02)

    assert len(transcripts) == 1
    assert "instance 1: turn 1" in transcripts[0].payload

    first_stt = supervisor.get_component("stt")
    assert first_stt is not None
    assert first_stt.instance_id == 1
    assert first_stt.is_running

    # --- Trigger Chaos: Inject fatal crash payload into STT ---
    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(
            channel="vad.speech_audio",
            phase=StreamPhase.AUDIO,
            payload=b"CRASH_PAYLOAD",
            trace_id="chaos-turn",
        ),
    )
    await asyncio.sleep(0.05)

    # Verify component entered FAILED/stopped state and is no longer running
    assert not first_stt.is_running

    # --- Supervisor Self-Healing: Supervisor detects failure and respawns ---
    for _ in range(40):
        await asyncio.sleep(0.03)
        current_stt = supervisor.get_component("stt")
        if current_stt and current_stt is not first_stt and current_stt.is_running:
            break

    recovered_stt = supervisor.get_component("stt")
    assert recovered_stt is not None, "Supervisor failed to resurrect STT component"
    assert recovered_stt is not first_stt, "Supervisor did not instantiate fresh instance"
    assert recovered_stt.is_running, "Resurrected STT component is not running"
    assert recovered_stt.instance_id == 2, "Expected instance_id 2 after recovery"

    child_spec = supervisor.get_child_spec("stt")
    assert child_spec.crash_count >= 1, "Supervisor did not record crash count"

    # Give fresh subscription loop a brief moment to bind to the bus
    await asyncio.sleep(0.03)

    # --- Turn 2: Verify pipeline processes subsequent turn normally ---
    await bus.publish(
        "vad.speech_audio",
        PipelineMessage(
            channel="vad.speech_audio",
            phase=StreamPhase.AUDIO,
            payload=b"normal_audio_2",
            trace_id="turn-2",
        ),
    )

    for _ in range(30):
        if len(transcripts) >= 2:
            break
        await asyncio.sleep(0.02)

    assert len(transcripts) == 2, "Pipeline failed to process turn after self-healing recovery"
    assert "instance 2: turn 1" in transcripts[1].payload

    # Teardown
    listener_task.cancel()
    await supervisor.stop()
    await bus.close()
