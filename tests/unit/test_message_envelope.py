"""Unit tests for PipelineMessage, StreamPhase, and ReasoningStep serialization contracts."""

import time
import pytest
from mindmirror.pipeline.message import PipelineMessage, StreamPhase, ReasoningStep


def test_pipeline_message_defaults():
    msg = PipelineMessage(channel="stt.transcript", payload="Hello world")
    assert msg.channel == "stt.transcript"
    assert msg.payload == "Hello world"
    assert msg.phase == StreamPhase.CONTROL
    assert msg.message_id is not None
    assert msg.trace_id is not None
    assert isinstance(msg.timestamp, float)
    assert msg.speaker_id is None
    assert msg.metadata == {}


def test_pipeline_message_msgpack_roundtrip():
    original = PipelineMessage(
        channel="ttt.response",
        payload={"text": "[EXCITED] System operational!", "sentiment": "positive"},
        speaker_id="speaker_42",
        phase=StreamPhase.RESPONDING,
        metadata={"model": "gemini-2.5-flash", "tokens": 128},
    )

    data = original.to_bytes()
    assert isinstance(data, bytes)

    reconstructed = PipelineMessage.from_bytes(data)
    assert reconstructed.channel == original.channel
    assert reconstructed.payload == original.payload
    assert reconstructed.speaker_id == original.speaker_id
    assert reconstructed.phase == StreamPhase.RESPONDING
    assert reconstructed.message_id == original.message_id
    assert reconstructed.trace_id == original.trace_id
    assert reconstructed.timestamp == original.timestamp
    assert reconstructed.metadata == original.metadata


def test_pipeline_message_binary_pcm_payload():
    pcm_bytes = b"\x00\x01\x02\x03" * 256  # 1024 bytes of dummy PCM
    original = PipelineMessage(
        channel="audio.raw",
        payload=pcm_bytes,
        phase=StreamPhase.AUDIO,
    )

    data = original.to_bytes()
    reconstructed = PipelineMessage.from_bytes(data)

    assert reconstructed.channel == "audio.raw"
    assert reconstructed.phase == StreamPhase.AUDIO
    assert reconstructed.payload == pcm_bytes


def test_reasoning_step_serialization():
    step = ReasoningStep(
        algorithm="tree_of_thoughts",
        step_id="step_01",
        parent_id=None,
        action="branch_generated",
        thought="Exploring search query hypothesis",
        score=0.87,
        tool_calls=[{"name": "web_search", "args": {"query": "NATS vs ZeroMQ"}}],
        metadata={"branch_index": 2},
    )

    msg = PipelineMessage(
        channel="reasoning.trace",
        payload=step,
        phase=StreamPhase.REASONING_STEP,
    )

    data = msg.to_bytes()
    reconstructed = PipelineMessage.from_bytes(data)

    assert reconstructed.channel == "reasoning.trace"
    assert reconstructed.phase == StreamPhase.REASONING_STEP
    # Payload is unpacked as dict
    assert isinstance(reconstructed.payload, dict)
    assert reconstructed.payload["step_id"] == "step_01"
    assert reconstructed.payload["algorithm"] == "tree_of_thoughts"
    assert reconstructed.payload["score"] == 0.87
    assert reconstructed.payload["tool_calls"] == [{"name": "web_search", "args": {"query": "NATS vs ZeroMQ"}}]


def test_stream_phase_values():
    assert StreamPhase.AUDIO.value == "audio"
    assert StreamPhase.TRANSCRIPT.value == "transcript"
    assert StreamPhase.THINKING.value == "thinking"
    assert StreamPhase.REASONING_STEP.value == "step"
    assert StreamPhase.TOOL_CALL.value == "tool_call"
    assert StreamPhase.RESPONDING.value == "responding"
    assert StreamPhase.CONTROL.value == "control"
    assert StreamPhase.TELEMETRY.value == "telemetry"
