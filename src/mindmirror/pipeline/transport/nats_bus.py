"""NATS Server transport connector for distributed/containerized deployments."""

import asyncio
from collections.abc import AsyncIterator
import logging
from typing import Any

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class NatsBus(MessageBus):
    """MessageBus implementation connecting to an open-source NATS server."""

    def __init__(self, servers: list[str] | str = "nats://127.0.0.1:4222") -> None:
        try:
            import nats
        except ImportError as e:
            raise ImportError(
                "nats-py is required to use NatsBus. Install it via 'pip install nats-py'."
            ) from e

        self.servers = [servers] if isinstance(servers, str) else servers
        self.nc: Any = None
        self._closed = False

    async def connect(self) -> None:
        """Connect to NATS cluster."""
        import nats

        if self.nc is None or self.nc.is_closed:
            self.nc = await nats.connect(servers=self.servers)

    async def publish(self, channel: str, message: PipelineMessage) -> None:
        if self._closed:
            raise RuntimeError("Cannot publish to closed NatsBus")
        if self.nc is None:
            await self.connect()

        # NATS subjects typically use dots (e.g., audio.raw)
        await self.nc.publish(channel, message.to_bytes())

    async def subscribe(self, channel: str) -> AsyncIterator[PipelineMessage]:
        if self._closed:
            raise RuntimeError("Cannot subscribe to closed NatsBus")
        if self.nc is None:
            await self.connect()

        sub = await self.nc.subscribe(channel)
        try:
            async for msg in sub.messages:
                if self._closed:
                    break
                yield PipelineMessage.from_bytes(msg.data)
        finally:
            await sub.unsubscribe()

    async def request(
        self, channel: str, message: PipelineMessage, timeout: float = 5.0
    ) -> PipelineMessage:
        if self._closed:
            raise RuntimeError("Cannot request on closed NatsBus")
        if self.nc is None:
            await self.connect()

        response = await self.nc.request(channel, message.to_bytes(), timeout=timeout)
        return PipelineMessage.from_bytes(response.data)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.nc and not self.nc.is_closed:
            await self.nc.drain()
            await self.nc.close()
