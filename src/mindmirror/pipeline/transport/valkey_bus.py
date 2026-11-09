"""Valkey / Redis transport connector for distributed deployments."""

import asyncio
from collections.abc import AsyncIterator
import logging
from typing import Any

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class ValkeyBus(MessageBus):
    """MessageBus implementation connecting to an open-source Valkey or Redis server."""

    def __init__(self, url: str = "redis://127.0.0.1:6379") -> None:
        try:
            import redis.asyncio as redis
        except ImportError as e:
            raise ImportError(
                "redis is required to use ValkeyBus. Install it via 'pip install redis'."
            ) from e

        self.url = url
        self._redis: Any = None
        self._closed = False

    async def connect(self) -> None:
        """Connect to Valkey/Redis instance."""
        import redis.asyncio as redis

        if self._redis is None:
            self._redis = redis.from_url(self.url)

    async def publish(self, channel: str, message: PipelineMessage) -> None:
        if self._closed:
            raise RuntimeError("Cannot publish to closed ValkeyBus")
        if self._redis is None:
            await self.connect()

        await self._redis.publish(channel, message.to_bytes())

    async def subscribe(self, channel: str) -> AsyncIterator[PipelineMessage]:
        if self._closed:
            raise RuntimeError("Cannot subscribe to closed ValkeyBus")
        if self._redis is None:
            await self.connect()

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for raw in pubsub.listen():
                if self._closed:
                    break
                if raw["type"] == "message":
                    yield PipelineMessage.from_bytes(raw["data"])
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def request(
        self, channel: str, message: PipelineMessage, timeout: float = 5.0
    ) -> PipelineMessage:
        if self._closed:
            raise RuntimeError("Cannot request on closed ValkeyBus")
        if self._redis is None:
            await self.connect()

        correlation_id = message.message_id
        reply_channel = f"reply.{correlation_id}"
        message.metadata["correlation_id"] = correlation_id
        message.metadata["reply_to"] = reply_channel

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(reply_channel)

        try:
            await self.publish(channel, message)

            async def _wait() -> PipelineMessage:
                async for raw in pubsub.listen():
                    if raw["type"] == "message":
                        msg = PipelineMessage.from_bytes(raw["data"])
                        if msg.metadata.get("correlation_id") == correlation_id:
                            return msg
                raise RuntimeError("Pubsub closed before reply received")

            return await asyncio.wait_for(_wait(), timeout=timeout)
        finally:
            await pubsub.unsubscribe(reply_channel)
            await pubsub.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._redis:
            await self._redis.close()
