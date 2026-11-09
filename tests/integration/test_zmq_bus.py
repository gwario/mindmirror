"""Integration tests for ZmqBus ZeroMQ IPC transport."""

import asyncio
import os
import stat
import time
import pytest
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.zmq_bus import ZmqBus


@pytest.mark.asyncio
async def test_zmq_bus_pub_sub(zmq_bus: ZmqBus):
    received = []

    # First subscribe so endpoint is listening
    async def _subscriber():
        async for msg in zmq_bus.subscribe("test.zmq.events"):
            received.append(msg)
            if len(received) >= 2:
                break

    task = asyncio.create_task(_subscriber())
    await asyncio.sleep(0.05)  # Allow socket creation & connection

    msg1 = PipelineMessage(channel="test.zmq.events", payload="ipc_first")
    msg2 = PipelineMessage(channel="test.zmq.events", payload="ipc_second")

    await zmq_bus.publish("test.zmq.events", msg1)
    await zmq_bus.publish("test.zmq.events", msg2)

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 2
    assert received[0].payload == "ipc_first"
    assert received[1].payload == "ipc_second"


@pytest.mark.asyncio
async def test_zmq_bus_socket_permissions_0600(zmq_bus: ZmqBus):
    channel = "secure.audio"
    msg = PipelineMessage(channel=channel, payload=b"\x00" * 128, phase=StreamPhase.AUDIO)
    await zmq_bus.publish(channel, msg)

    _, fs_path = zmq_bus._channel_to_endpoint(channel)
    assert fs_path is not None
    assert os.path.exists(fs_path), f"IPC socket file {fs_path} should exist"

    # Verify file permission bits are 0600 (read/write by owner only)
    mode = stat.S_IMODE(os.stat(fs_path).st_mode)
    assert mode == 0o600, f"Expected 0o600 permissions on {fs_path}, got {oct(mode)}"


@pytest.mark.asyncio
async def test_zmq_bus_burst_throughput(zmq_bus: ZmqBus):
    channel = "perf.burst"
    num_messages = 500
    received_count = 0

    async def _subscriber():
        nonlocal received_count
        async for _ in zmq_bus.subscribe(channel):
            received_count += 1
            if received_count >= num_messages:
                break

    sub_task = asyncio.create_task(_subscriber())
    await asyncio.sleep(0.05)

    start_time = time.perf_counter()
    for i in range(num_messages):
        msg = PipelineMessage(channel=channel, payload=f"item_{i}")
        await zmq_bus.publish(channel, msg)

    await asyncio.wait_for(sub_task, timeout=3.0)
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_messages) * 1000

    assert received_count == num_messages
    assert avg_latency_ms < 2.0, f"Average latency {avg_latency_ms:.3f}ms per message exceeded SLA"


@pytest.mark.asyncio
async def test_zmq_bus_cleanup_on_close(zmq_bus: ZmqBus):
    channel = "cleanup.test"
    await zmq_bus.publish(channel, PipelineMessage(channel=channel, payload="hello"))

    _, fs_path = zmq_bus._channel_to_endpoint(channel)
    assert fs_path is not None and os.path.exists(fs_path)

    await zmq_bus.close()
    assert not os.path.exists(fs_path), f"IPC file {fs_path} should be deleted on close()"
