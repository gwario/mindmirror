"""Unit tests for InteractionFeedbackWindow (sliding retention window, trace lookup, stats)."""

import asyncio
import pytest

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.telemetry.feedback import InteractionFeedback, InteractionFeedbackWindow


def test_interaction_feedback_sliding_window_capacity():
    """Validates that InteractionFeedbackWindow enforces capacity and evicts oldest entries."""
    window = InteractionFeedbackWindow(max_window_size=3)

    window.record_feedback(trace_id="t1", rating=5, category="style")
    window.record_feedback(trace_id="t2", rating=4, category="latency")
    window.record_feedback(trace_id="t3", rating=5, category="style")

    assert len(window.get_recent()) == 3
    assert window.get_feedback("t1") is not None
    assert window.get_feedback("t3") is not None

    # Adding 4th feedback should evict t1
    window.record_feedback(trace_id="t4", rating=2, category="hallucination", correction="Wrong name")

    assert len(window.get_recent()) == 3
    assert window.get_feedback("t1") is None  # Evicted
    assert window.get_feedback("t2") is not None
    assert window.get_feedback("t4") is not None
    assert window.get_feedback("t4").correction == "Wrong name"

    # Test summary statistics
    stats = window.get_summary_stats()
    assert stats["count"] == 3
    # Average of 4, 5, 2 = 11/3 = 3.67
    assert stats["average_rating"] == 3.67
    assert stats["categories"]["style"] == 1
    assert stats["categories"]["latency"] == 1
    assert stats["categories"]["hallucination"] == 1


@pytest.mark.asyncio
async def test_interaction_feedback_bus_listener():
    """Validates that InteractionFeedbackWindow automatically records feedback from the message bus."""
    bus = LocalBus()
    window = InteractionFeedbackWindow(max_window_size=10, bus=bus)
    await asyncio.sleep(0.01)

    await bus.publish(
        "control.feedback",
        PipelineMessage(
            channel="control.feedback",
            trace_id="trace-bus-turn-1",
            payload={
                "rating": 5,
                "category": "voice_style",
                "notes": "Spot-on British accent mate!",
            },
            speaker_id="mario",
        ),
    )
    await asyncio.sleep(0.03)

    fb = window.get_feedback("trace-bus-turn-1")
    assert fb is not None
    assert fb.rating == 5
    assert fb.category == "voice_style"
    assert fb.speaker_id == "mario"
    assert "Spot-on" in fb.notes

    await bus.close()
