"""MessageBus abstract base class defining the transport contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from mindmirror.pipeline.message import PipelineMessage


class MessageBus(ABC):
    """Abstract transport layer. All pipeline components communicate strictly through this interface."""

    @abstractmethod
    async def publish(self, channel: str, message: PipelineMessage) -> None:
        """Publishes a message to the specified topic/channel."""
        pass

    @abstractmethod
    async def subscribe(self, channel: str) -> AsyncIterator[PipelineMessage]:
        """Subscribes to a channel and yields incoming PipelineMessages."""
        pass

    @abstractmethod
    async def request(
        self, channel: str, message: PipelineMessage, timeout: float = 5.0
    ) -> PipelineMessage:
        """Sends a request message and awaits a correlated reply on reply channel within timeout."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes all transport connections and cleans up allocated resources."""
        pass
