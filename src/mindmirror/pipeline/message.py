"""Pipeline message envelopes, stream phases, and serialization contracts for MindMirror."""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any
import uuid
import msgpack


class StreamPhase(str, Enum):
    """Lifecycle phase of a message within the pipeline stream."""
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    THINKING = "thinking"         # Internal LLM thinking tokens
    REASONING_STEP = "step"       # Discrete algorithm event (e.g. MCTS evaluation)
    TOOL_CALL = "tool_call"       # Speculative or committed tool execution
    RESPONDING = "responding"     # Spoken response text (streamed to TTS)
    CONTROL = "control"
    TELEMETRY = "telemetry"


@dataclass
class ReasoningStep:
    """Discrete step within custom reasoning search algorithms (e.g. MCTS, Tree-of-Thoughts)."""
    algorithm: str                # e.g. "tree_of_thoughts", "mcts", "reflexion"
    step_id: str
    parent_id: str | None
    action: str                   # e.g. "branch_generated", "tool_invoked", "critique", "prune", "score"
    thought: str
    score: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "step_id": self.step_id,
            "parent_id": self.parent_id,
            "action": self.action,
            "thought": self.thought,
            "score": self.score,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningStep":
        return cls(
            algorithm=d["algorithm"],
            step_id=d["step_id"],
            parent_id=d.get("parent_id"),
            action=d["action"],
            thought=d["thought"],
            score=d.get("score"),
            tool_calls=d.get("tool_calls", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class PipelineMessage:
    """Universal message envelope passing across transport buses (Local, ZMQ, NATS, Valkey)."""
    channel: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    speaker_id: str | None = None
    phase: StreamPhase = StreamPhase.CONTROL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation."""
        payload_data = self.payload
        if isinstance(payload_data, ReasoningStep):
            payload_data = payload_data.to_dict()
        return {
            "channel": self.channel,
            "payload": payload_data,
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "speaker_id": self.speaker_id,
            "phase": self.phase.value if isinstance(self.phase, StreamPhase) else str(self.phase),
            "metadata": self.metadata,
        }

    def to_bytes(self) -> bytes:
        """Serializes the envelope to compact binary format via msgpack."""
        return msgpack.packb(self.to_dict(), use_bin_type=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineMessage":
        """Reconstruct message from dictionary."""
        phase_raw = d.get("phase", StreamPhase.CONTROL.value)
        try:
            phase = StreamPhase(phase_raw)
        except (ValueError, KeyError):
            phase = StreamPhase.CONTROL

        return cls(
            channel=d["channel"],
            payload=d["payload"],
            message_id=d.get("message_id", str(uuid.uuid4())),
            trace_id=d.get("trace_id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", time.time()),
            speaker_id=d.get("speaker_id"),
            phase=phase,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PipelineMessage":
        """Deserializes message envelope from msgpack binary."""
        d = msgpack.unpackb(data, raw=False)
        return cls.from_dict(d)
