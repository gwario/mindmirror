"""Integration tests for LocalBus in-memory transport."""

import asyncio
import pytest
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus


@pytest.mark.asyncio
async def test_local_bus_pub_sub(local_bus: LocalBus):
    received = []

    async def _subscriber():
        async for msg in local_bus.subscribe("test.events"):
            received.append(msg)
            if len(received) >= 2:
                break

    task = asyncio.create_task(_subscriber())
    await asyncio.sleep(0.01)

    msg1 = PipelineMessage(channel="test.events", payload="first", phase=StreamPhase.CONTROL)
    msg2 = PipelineMessage(channel="test.events", payload="second", phase=StreamPhase.CONTROL)

    await local_bus.publish("test.events", msg1)
    await local_bus.publish("test.events", msg2)

    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 2
    assert received[0].payload == "first"
    assert received[1].payload == "second"


@pytest.mark.asyncio
async def test_local_bus_multi_subscriber_fan_out(local_bus: LocalBus):
    sub1_msgs = []
    sub2_msgs = []

    async def _sub1():
        async for msg in local_bus.subscribe("broadcast.channel"):
            sub1_msgs.append(msg)
            if len(sub1_msgs) >= 1:
                break

    async def _sub2():
        async for msg in local_bus.subscribe("broadcast.channel"):
            sub2_msgs.append(msg)
            if len(sub2_msgs) >= 1:
                break

    t1 = asyncio.create_task(_sub1())
    t2 = asyncio.create_task(_sub2())
    await asyncio.sleep(0.01)

    msg = PipelineMessage(channel="broadcast.channel", payload={"alert": "ping"})
    await local_bus.publish("broadcast.channel", msg)

    await asyncio.gather(t1, t2)
    assert len(sub1_msgs) == 1
    assert len(sub2_msgs) == 1
    assert sub1_msgs[0].payload == {"alert": "ping"}
    assert sub2_msgs[0].payload == {"alert": "ping"}


@pytest.mark.asyncio
async def test_local_bus_channel_isolation(local_bus: LocalBus):
    chan_a_msgs = []

    async def _sub():
        async for msg in local_bus.subscribe("channel.a"):
            chan_a_msgs.append(msg)
            break

    task = asyncio.create_task(_sub())
    await asyncio.sleep(0.01)

    # Publish to channel B
    await local_bus.publish("channel.b", PipelineMessage(channel="channel.b", payload="wrong"))
    await asyncio.sleep(0.05)
    assert len(chan_a_msgs) == 0

    # Publish to channel A
    await local_bus.publish("channel.a", PipelineMessage(channel="channel.a", payload="correct"))
    await asyncio.wait_for(task, timeout=1.0)
    assert len(chan_a_msgs) == 1
    assert chan_a_msgs[0].payload == "correct"


@pytest.mark.asyncio
async def test_local_bus_request_reply_success(local_bus: LocalBus):
    async def _responder():
        async for req in local_bus.subscribe("service.math"):
            num = req.payload["number"]
            reply = PipelineMessage(
                channel=req.channel,
                payload={"result": num * 2},
                metadata={"correlation_id": req.metadata.get("correlation_id")},
            )
            await local_bus.publish("service.math", reply)
            break

    resp_task = asyncio.create_task(_responder())
    await asyncio.sleep(0.01)

    request_msg = PipelineMessage(channel="service.math", payload={"number": 21})
    response = await local_bus.request("service.math", request_msg, timeout=2.0)

    assert response.payload == {"result": 42}
    await resp_task


@pytest.mark.asyncio
async def test_local_bus_request_timeout(local_bus: LocalBus):
    req = PipelineMessage(channel="service.empty", payload="ping")
    with pytest.raises(asyncio.TimeoutError):
        await local_bus.request("service.empty", req, timeout=0.1)


@pytest.mark.asyncio
async def test_local_bus_clean_close(local_bus: LocalBus):
    sub_exited = False

    async def _subscriber():
        nonlocal sub_exited
        async for _ in local_bus.subscribe("any.channel"):
            pass
        sub_exited = True

    task = asyncio.create_task(_subscriber())
    await asyncio.sleep(0.01)

    await local_bus.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert sub_exited is True
