"""Interaction Feedback Store with Sliding Retention Window for MindMirror Turns."""

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import logging
import os
import time
from typing import Any, Optional
import uuid

from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


@dataclass
class InteractionFeedback:
    """Feedback payload for a single conversational turn linked to its telemetry trace."""

    trace_id: str
    rating: int  # 1 to 5 (or -1/1 for thumbs up/down)
    category: str = "general"  # e.g. 'style', 'latency', 'hallucination', 'tool_accuracy'
    correction: Optional[str] = None
    notes: Optional[str] = None
    speaker_id: Optional[str] = None
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionFeedback":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class InteractionFeedbackWindow:
    """Sliding retention window store for interaction feedback.

    Maintains the last `max_window_size` feedback entries in memory and optionally
    syncs to distributed Valkey storage (`mm:feedback:window`).
    """

    def __init__(
        self,
        max_window_size: int = 100,
        bus: Optional[MessageBus] = None,
        valkey_url: Optional[str] = None,
    ) -> None:
        self.max_window_size = max_window_size
        self.bus = bus
        self._valkey_url = valkey_url or os.getenv("VALKEY_URL")
        self._window: deque[InteractionFeedback] = deque(maxlen=max_window_size)
        self._by_trace: dict[str, InteractionFeedback] = {}
        self._sub_task: Optional[Any] = None

        if bus:
            self.attach_bus(bus)

    def attach_bus(self, bus: MessageBus) -> None:
        """Subscribes to control.feedback channel on the bus."""
        import asyncio
        self.bus = bus
        self._sub_task = asyncio.create_task(self._listen_for_feedback())

    async def _listen_for_feedback(self) -> None:
        """Consumes feedback events published on the message bus."""
        import asyncio
        if not self.bus:
            return
        try:
            async for msg in self.bus.subscribe("control.feedback"):
                payload = msg.payload
                if isinstance(payload, dict):
                    trace_id = payload.get("trace_id", msg.trace_id)
                    rating = int(payload.get("rating", 3))
                    self.record_feedback(
                        trace_id=trace_id,
                        rating=rating,
                        category=payload.get("category", "general"),
                        correction=payload.get("correction"),
                        notes=payload.get("notes"),
                        speaker_id=payload.get("speaker_id", msg.speaker_id),
                        metadata=payload.get("metadata", {}),
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in feedback listener loop: %s", e)

    def record_feedback(
        self,
        trace_id: str,
        rating: int,
        category: str = "general",
        correction: Optional[str] = None,
        notes: Optional[str] = None,
        speaker_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> InteractionFeedback:
        """Records feedback into the sliding window and updates index."""
        fb = InteractionFeedback(
            trace_id=trace_id,
            rating=rating,
            category=category,
            correction=correction,
            notes=notes,
            speaker_id=speaker_id,
            metadata=metadata or {},
        )

        # If window is at capacity, clean up the evicted trace from the index
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._by_trace.pop(evicted.trace_id, None)

        self._window.append(fb)
        self._by_trace[trace_id] = fb

        # Optional distributed Valkey storage
        if self._valkey_url:
            try:
                import redis
                r = redis.Redis.from_url(self._valkey_url)
                r.lpush("mm:feedback:sliding_window", json.dumps(fb.to_dict()))
                r.ltrim("mm:feedback:sliding_window", 0, self.max_window_size - 1)
                r.set(f"mm:feedback:trace:{trace_id}", json.dumps(fb.to_dict()), ex=86400)
            except Exception as e:
                logger.debug("Optional Valkey feedback sync skipped: %s", e)

        logger.info(
            "Recorded interaction feedback: trace='%s', rating=%d, category='%s'",
            trace_id,
            rating,
            category,
        )
        return fb

    def get_feedback(self, trace_id: str) -> Optional[InteractionFeedback]:
        """Retrieves feedback associated with a specific conversational turn trace."""
        return self._by_trace.get(trace_id)

    def get_recent(self, limit: int = 50) -> list[InteractionFeedback]:
        """Returns the most recent feedback entries in chronological order."""
        items = list(self._window)
        return items[-limit:]

    def get_summary_stats(self) -> dict[str, Any]:
        """Calculates statistical summary across the current feedback window."""
        if not self._window:
            return {"count": 0, "average_rating": 0.0, "categories": {}}

        total_rating = sum(f.rating for f in self._window)
        categories: dict[str, int] = {}
        for f in self._window:
            categories[f.category] = categories.get(f.category, 0) + 1

        return {
            "count": len(self._window),
            "average_rating": round(total_rating / len(self._window), 2),
            "categories": categories,
        }

    def clear(self) -> None:
        """Clears the sliding window and trace lookup dictionary."""
        self._window.clear()
        self._by_trace.clear()
