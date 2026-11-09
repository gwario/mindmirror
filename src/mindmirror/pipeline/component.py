"""Base class defining standard lifecycle and health checks for pipeline components."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class ComponentHealth(str, Enum):
    """Health states of a component."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class HealthStatus:
    """Standardized component health report."""
    component_name: str
    status: ComponentHealth
    uptime_seconds: float
    error_count: int = 0
    last_error: str | None = None
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
        }


class PipelineComponent(ABC):
    """Abstract base class for all supervised MindMirror pipeline components.

    Enforces lifecycle contracts: start -> drain -> stop, and continuous health reporting.
    """

    def __init__(self, name: str, bus: MessageBus) -> None:
        self.name = name
        self.bus = bus
        self._start_time: float | None = None
        self._is_running = False
        self._error_count = 0
        self._last_error: str | None = None
        self._status = ComponentHealth.STOPPED
        self._subscription_tasks: list[asyncio.Task] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    def create_subscription_task(
        self,
        channel: str,
        handler: Callable[[PipelineMessage], Any],
    ) -> asyncio.Task:
        """Helper to create and track a background subscription task that feeds incoming messages to handler."""
        async def _subscription_loop():
            try:
                async for message in self.bus.subscribe(channel):
                    if not self._is_running:
                        break
                    res = handler(message)
                    if asyncio.iscoroutine(res):
                        await res
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.record_error(e)
                logger.error("Error in component '%s' subscription loop for '%s': %s", self.name, channel, e)

        task = asyncio.create_task(_subscription_loop())
        self._subscription_tasks.append(task)
        return task

    @abstractmethod
    async def start(self) -> None:
        """Starts the component's async event loop, subscribing to required channels."""
        self._start_time = time.time()
        self._is_running = True
        self._status = ComponentHealth.HEALTHY
        await asyncio.sleep(0.01)

    async def drain(self, timeout: float = 2.0) -> None:
        """Drains pending in-flight messages before shutdown.

        Components can override this to complete currently processing audio frames or inference tasks.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stops all background tasks and unsubscribes from channels."""
        for task in self._subscription_tasks:
            if not task.done():
                task.cancel()
        self._subscription_tasks.clear()
        self._is_running = False
        self._status = ComponentHealth.STOPPED

    async def health(self) -> HealthStatus:
        """Returns the current health status of the component."""
        uptime = (time.time() - self._start_time) if self._start_time else 0.0
        return HealthStatus(
            component_name=self.name,
            status=self._status,
            uptime_seconds=uptime,
            error_count=self._error_count,
            last_error=self._last_error,
            last_heartbeat=time.time(),
        )

    def record_error(self, error: Exception | str) -> None:
        """Helper to record internal error events for health monitoring."""
        self._error_count += 1
        self._last_error = str(error)
        self._status = ComponentHealth.DEGRADED
