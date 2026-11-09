"""Root pytest fixtures and test configuration for MindMirror."""

import asyncio
from collections.abc import AsyncIterator
import os
from pathlib import Path
import sys
import uuid
import pytest

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mindmirror.pipeline.message import PipelineMessage, StreamPhase, ReasoningStep
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.pipeline.transport.zmq_bus import ZmqBus


@pytest.fixture
async def local_bus() -> AsyncIterator[LocalBus]:
    """Provides a clean, in-memory LocalBus instance per test."""
    bus = LocalBus()
    try:
        yield bus
    finally:
        await bus.close()


@pytest.fixture
async def zmq_bus() -> AsyncIterator[ZmqBus]:
    """Provides an isolated ZmqBus instance using unique temporary IPC endpoints."""
    prefix = f"ipc:///tmp/test_mindmirror_{uuid.uuid4().hex[:8]}_"
    bus = ZmqBus(endpoint_prefix=prefix)
    try:
        yield bus
    finally:
        await bus.close()


@pytest.fixture
def sample_message() -> PipelineMessage:
    """Provides a sample PipelineMessage for contract verification."""
    return PipelineMessage(
        channel="test.channel",
        payload={"text": "Hello, MindMirror!", "confidence": 0.98},
        speaker_id="mario",
        phase=StreamPhase.RESPONDING,
        metadata={"turn_id": 42},
    )
