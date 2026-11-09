"""ZeroMQ IPC transport implementation for ultra-low latency local IPC."""

import asyncio
from collections.abc import AsyncIterator
import logging
import os
import re
from typing import Any
import zmq
import zmq.asyncio

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class ZmqBus(MessageBus):
    """Message bus connector using ZeroMQ IPC sockets (chmod 0600) for sub-millisecond local IPC."""

    def __init__(self, endpoint_prefix: str = "ipc:///tmp/mindmirror_") -> None:
        self.endpoint_prefix = endpoint_prefix
        self.ctx = zmq.asyncio.Context()
        self._pub_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._all_sockets: set[zmq.asyncio.Socket] = set()
        self._created_ipc_files: set[str] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    def _channel_to_endpoint(self, channel: str) -> tuple[str, str | None]:
        """Convert channel name to safe IPC endpoint and optional filesystem path."""
        safe_channel = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", channel)
        endpoint = f"{self.endpoint_prefix}{safe_channel}.ipc"
        fs_path = None
        if endpoint.startswith("ipc://"):
            fs_path = endpoint[len("ipc://"):]
        return endpoint, fs_path

    async def _get_or_create_pub(self, channel: str) -> zmq.asyncio.Socket:
        """Get or create bound PUB socket for a given channel."""
        async with self._lock:
            if channel in self._pub_sockets:
                return self._pub_sockets[channel]

            endpoint, fs_path = self._channel_to_endpoint(channel)
            sock = self.ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            self._all_sockets.add(sock)

            # If stale socket file exists, remove before binding
            if fs_path and os.path.exists(fs_path):
                try:
                    os.unlink(fs_path)
                except OSError:
                    pass

            sock.bind(endpoint)
            self._pub_sockets[channel] = sock

            # Apply chmod 0600 on created IPC socket file for local security
            if fs_path and os.path.exists(fs_path):
                try:
                    os.chmod(fs_path, 0o600)
                    self._created_ipc_files.add(fs_path)
                except OSError as e:
                    logger.warning(f"Could not chmod 0600 on {fs_path}: {e}")

            # Allow short handshake window for slow-joiner subscribers
            await asyncio.sleep(0.01)
            return sock

    async def publish(self, channel: str, message: PipelineMessage) -> None:
        """Publish a PipelineMessage to the channel's ZeroMQ IPC endpoint."""
        if self._closed:
            raise RuntimeError("Cannot publish to closed ZmqBus")

        sock = await self._get_or_create_pub(channel)
        await sock.send(message.to_bytes())

    async def subscribe(self, channel: str) -> AsyncIterator[PipelineMessage]:
        """Subscribes to a channel's ZeroMQ IPC endpoint and yields incoming messages."""
        if self._closed:
            raise RuntimeError("Cannot subscribe to closed ZmqBus")

        # Ensure PUB socket is bound and IPC socket file exists before connecting
        await self._get_or_create_pub(channel)

        endpoint, _ = self._channel_to_endpoint(channel)
        sub_sock = self.ctx.socket(zmq.SUB)
        sub_sock.setsockopt(zmq.LINGER, 0)
        self._all_sockets.add(sub_sock)
        sub_sock.connect(endpoint)
        sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        # Allow subscription handshake to register with publisher
        await asyncio.sleep(0.05)

        try:
            while not self._closed:
                try:
                    data = await sub_sock.recv()
                    yield PipelineMessage.from_bytes(data)
                except zmq.ZMQError as e:
                    if self._closed or e.errno == zmq.ETERM:
                        break
                    raise
        finally:
            self._all_sockets.discard(sub_sock)
            if not sub_sock.closed:
                sub_sock.close(linger=0)

    async def request(
        self, channel: str, message: PipelineMessage, timeout: float = 5.0
    ) -> PipelineMessage:
        """Publishes a request message and awaits a correlated reply on an ephemeral reply channel."""
        if self._closed:
            raise RuntimeError("Cannot request on closed ZmqBus")

        correlation_id = message.message_id
        reply_channel = f"reply_{correlation_id}"
        message.metadata["correlation_id"] = correlation_id
        message.metadata["reply_to"] = reply_channel

        # Start listening on the reply channel first
        endpoint, _ = self._channel_to_endpoint(reply_channel)
        sub_sock = self.ctx.socket(zmq.SUB)
        sub_sock.setsockopt(zmq.LINGER, 0)
        self._all_sockets.add(sub_sock)
        sub_sock.connect(endpoint)
        sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        await asyncio.sleep(0.01)

        try:
            # Publish request
            await self.publish(channel, message)

            async def _wait_reply() -> PipelineMessage:
                while True:
                    data = await sub_sock.recv()
                    msg = PipelineMessage.from_bytes(data)
                    if msg.metadata.get("correlation_id") == correlation_id:
                        return msg

            return await asyncio.wait_for(_wait_reply(), timeout=timeout)
        finally:
            self._all_sockets.discard(sub_sock)
            if not sub_sock.closed:
                sub_sock.close(linger=0)

    async def close(self) -> None:
        """Close all sockets, remove IPC files, and terminate ZeroMQ context."""
        if self._closed:
            return
        self._closed = True

        async with self._lock:
            for sock in list(self._all_sockets):
                try:
                    if not sock.closed:
                        sock.close(linger=0)
                except Exception:
                    pass
            self._all_sockets.clear()
            self._pub_sockets.clear()

        # Remove created IPC socket files from disk
        for fs_path in list(self._created_ipc_files):
            if os.path.exists(fs_path):
                try:
                    os.unlink(fs_path)
                except OSError:
                    pass
        self._created_ipc_files.clear()

        try:
            self.ctx.destroy(linger=0)
        except Exception:
            self.ctx.term()
