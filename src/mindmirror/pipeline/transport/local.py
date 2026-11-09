"""In-memory LocalBus implementation using asyncio.Queue for zero-dependency hermetic testing."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
import logging
from typing import Any

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class LocalBus(MessageBus):
    """In-memory message bus backed by asyncio.Queue.

    Provides multi-subscriber fan-out, request-reply correlation, and sub-millisecond
    in-process transport with zero external daemon dependencies.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[PipelineMessage | None]]] = (
            defaultdict(set)
        )
        self._reply_futures: dict[str, asyncio.Future[PipelineMessage]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def publish(self, channel: str, message: PipelineMessage) -> None:
        """Publishes a message to all subscribers of the channel."""
        if self._closed:
            raise RuntimeError("Cannot publish to closed LocalBus")

        # 1. Check if this is a reply to an outstanding request
        correlation_id = message.metadata.get("correlation_id")
        if (
            correlation_id
            and correlation_id in self._reply_futures
            and message.message_id != correlation_id
        ):
            future = self._reply_futures[correlation_id]
            if not future.done():
                future.set_result(message)

        # 2. Fan-out to all active subscribers for the channel
        async with self._lock:
            queues = list(self._subscribers.get(channel, set()))

        for q in queues:
            await q.put(message)

    async def subscribe(self, channel: str) -> AsyncIterator[PipelineMessage]:
        """Subscribes to a channel and yields messages as they are published."""
        if self._closed:
            raise RuntimeError("Cannot subscribe to closed LocalBus")

        q: asyncio.Queue[PipelineMessage | None] = asyncio.Queue()
        async with self._lock:
            self._subscribers[channel].add(q)

        try:
            while not self._closed:
                msg = await q.get()
                if msg is None:  # Sentinel for bus shutdown or unsubscribe
                    break
                yield msg
        finally:
            async with self._lock:
                if q in self._subscribers[channel]:
                    self._subscribers[channel].remove(q)
                if not self._subscribers[channel]:
                    del self._subscribers[channel]

    async def request(
        self, channel: str, message: PipelineMessage, timeout: float = 5.0
    ) -> PipelineMessage:
        """Publishes a request message and awaits a correlated reply."""
        if self._closed:
            raise RuntimeError("Cannot request on closed LocalBus")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PipelineMessage] = loop.create_future()
        correlation_id = message.message_id
        message.metadata["correlation_id"] = correlation_id
        self._reply_futures[correlation_id] = future

        try:
            await self.publish(channel, message)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._reply_futures.pop(correlation_id, None)

    async def close(self) -> None:
        """Closes the bus, signaling sentinel None to all subscribers and cancelling pending requests."""
        if self._closed:
            return
        self._closed = True

        async with self._lock:
            for channel, queues in list(self._subscribers.items()):
                for q in queues:
                    await q.put(None)
            self._subscribers.clear()

        # Cancel any pending reply futures
        for cid, fut in list(self._reply_futures.items()):
            if not fut.done():
                fut.cancel()
        self._reply_futures.clear()
