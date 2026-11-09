"""Regression test for thinking token leakage protection.

Contract assertions:
1. Provider reasoning tokens (<think>...</think> or native thinking deltas) are demuxed.
2. StreamPhase.THINKING tokens are published strictly to 'ttt.thinking' (for UI/telemetry).
3. Only StreamPhase.RESPONDING tokens with emotion style tags are published to 'ttt.response'.
4. TTSComponent and audio.playback NEVER receive thinking tokens or synthesize speech for thinking deltas.
"""

import asyncio
import pytest

from mindmirror.llm.component import ReasoningComponent
from mindmirror.llm.reasoning.native.provider_native import ProviderNativeReasoningStrategy
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.tts.component import TTSComponent
from mindmirror.tts.interface import TTSInterface


class MockTTSEngine(TTSInterface):
    """Mock TTS engine recording synthesis requests and emitting deterministic PCM chunks."""

    def __init__(self) -> None:
        self.synthesized_calls: list[tuple[str, str]] = []

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        self.synthesized_calls.append((text, style))
        # Return dummy 16-bit PCM chunk
        return [b"\x00\x01" * 160]


@pytest.mark.asyncio
async def test_thinking_token_leakage_protection():
    """Validates that internal reasoning/thinking tokens are never forwarded to TTS synthesis or audio.playback."""
    bus = LocalBus()

    # Raw model output containing thinking tags and vocal speech
    secret_thinking = "Analyzing user intent. Step 1: Deliberating confidential reasoning path. Step 2: Formulating reply."
    spoken_answer = "Everything is running smoothly, Mario!"
    raw_llm_output = f"<think>{secret_thinking}</think>[EXCITED] {spoken_answer}"

    strategy = ProviderNativeReasoningStrategy(
        llm_generate_fn=lambda prompt: raw_llm_output,
        default_style="NEUTRAL",
    )
    reasoning_component = ReasoningComponent(bus=bus, strategy=strategy)

    tts_engine = MockTTSEngine()
    tts_component = TTSComponent(bus=bus, tts_engine=tts_engine)

    # Collect messages across pipeline topics
    thinking_messages: list[PipelineMessage] = []
    response_messages: list[PipelineMessage] = []
    audio_playback_messages: list[PipelineMessage] = []

    async def _collect_thinking():
        async for msg in bus.subscribe("ttt.thinking"):
            thinking_messages.append(msg)

    async def _collect_response():
        async for msg in bus.subscribe("ttt.response"):
            response_messages.append(msg)

    async def _collect_audio():
        async for msg in bus.subscribe("audio.playback"):
            audio_playback_messages.append(msg)

    t_think = asyncio.create_task(_collect_thinking())
    t_resp = asyncio.create_task(_collect_response())
    t_audio = asyncio.create_task(_collect_audio())

    await reasoning_component.start()
    await tts_component.start()
    await asyncio.sleep(0.02)

    # Inject user transcript
    await bus.publish(
        "stt.transcript",
        PipelineMessage(
            channel="stt.transcript",
            phase=StreamPhase.TRANSCRIPT,
            payload="How are systems performing today?",
            trace_id="trace_leakage_test_1",
            speaker_id="mario",
        ),
    )

    # Allow turn to execute through reasoning -> TTS
    await asyncio.sleep(0.1)

    # 1. Assert thinking tokens arrived on ttt.thinking
    assert len(thinking_messages) == 1
    assert thinking_messages[0].phase == StreamPhase.THINKING
    assert thinking_messages[0].payload == secret_thinking

    # 2. Assert ttt.response received ONLY the spoken text with style tag
    assert len(response_messages) == 1
    assert response_messages[0].phase == StreamPhase.RESPONDING
    assert response_messages[0].payload == f"[EXCITED] {spoken_answer}"
    assert secret_thinking not in response_messages[0].payload
    assert "<think>" not in response_messages[0].payload

    # 3. Assert TTS engine was invoked ONLY with clean text and parsed style
    assert len(tts_engine.synthesized_calls) == 1
    synth_text, synth_style = tts_engine.synthesized_calls[0]
    assert synth_text == spoken_answer
    assert synth_style == "EXCITED"
    assert secret_thinking not in synth_text

    # 4. Assert audio.playback received only speech chunks with correct metadata
    assert len(audio_playback_messages) >= 1
    for audio_msg in audio_playback_messages:
        assert audio_msg.phase == StreamPhase.AUDIO
        assert audio_msg.metadata["style"] == "EXCITED"
        assert audio_msg.metadata["text"] == spoken_answer
        assert secret_thinking not in audio_msg.metadata["text"]

    # Clean up
    t_think.cancel()
    t_resp.cancel()
    t_audio.cancel()
    await reasoning_component.stop()
    await tts_component.stop()
    await bus.close()
