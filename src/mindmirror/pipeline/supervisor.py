"""Erlang-style ComponentSupervisor with exponential backoff and zero-downtime blue-green redeploy."""

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable

from mindmirror.pipeline.component import ComponentHealth, HealthStatus, PipelineComponent

logger = logging.getLogger(__name__)


@dataclass
class ChildSpec:
    """Specification and runtime supervision state for a supervised pipeline component."""
    name: str
    factory: Callable[[], PipelineComponent]
    instance: PipelineComponent | None = None
    task: asyncio.Task | None = None
    crash_count: int = 0
    last_crash_time: float = 0.0
    last_start_time: float = 0.0
    min_backoff: float = 1.0
    max_backoff: float = 30.0
    backoff_factor: float = 2.0
    stability_window: float = 5.0  # seconds running healthy before resetting crash_count


class ComponentSupervisor:
    """Supervises pipeline components with crash detection, exponential backoff, and blue-green redeployment.

    Follows Erlang OTP-style supervision contracts:
    - Isolates child component failures.
    - Automatically restarts crashed components with exponential backoff.
    - Tracks crash metrics and component health states.
    - Provides zero-downtime blue-green `redeploy()` for live updates.
    """

    def __init__(self, check_interval: float = 0.1) -> None:
        self.check_interval = check_interval
        self._specs: dict[str, ChildSpec] = {}
        self._supervision_task: asyncio.Task | None = None
        self._is_running = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def register(
        self,
        name: str,
        factory: Callable[[], PipelineComponent],
        min_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        stability_window: float = 5.0,
    ) -> None:
        """Registers a component specification with the supervisor."""
        if name in self._specs and self._specs[name].instance and self._specs[name].instance.is_running:
            raise ValueError(f"Component '{name}' is already running. Use redeploy() for live updates.")

        self._specs[name] = ChildSpec(
            name=name,
            factory=factory,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
            backoff_factor=backoff_factor,
            stability_window=stability_window,
        )

    def get_component(self, name: str) -> PipelineComponent | None:
        """Retrieves active component instance by name."""
        spec = self._specs.get(name)
        return spec.instance if spec else None

    def get_child_spec(self, name: str) -> ChildSpec | None:
        """Retrieves child spec by name."""
        return self._specs.get(name)

    async def start(self) -> None:
        """Starts all registered components and the background supervision monitor loop."""
        async with self._lock:
            if self._is_running:
                return
            self._is_running = True

            for name, spec in self._specs.items():
                await self._start_child(spec)

            self._supervision_task = asyncio.create_task(self._supervision_loop())
            # Yield briefly to event loop to allow subscription tasks to register on the bus
            await asyncio.sleep(0.01)
            logger.info("ComponentSupervisor started with %d components.", len(self._specs))

    async def stop(self) -> None:
        """Stops the supervision loop and cleanly shuts down all child components."""
        async with self._lock:
            if not self._is_running:
                return
            self._is_running = False

            if self._supervision_task and not self._supervision_task.done():
                self._supervision_task.cancel()
                try:
                    await self._supervision_task
                except asyncio.CancelledError:
                    pass
                self._supervision_task = None

            for name, spec in self._specs.items():
                await self._stop_child(spec)

            logger.info("ComponentSupervisor stopped.")

    async def redeploy(
        self,
        name: str,
        new_factory: Callable[[], PipelineComponent],
        drain_timeout: float = 2.0,
    ) -> PipelineComponent:
        """Zero-downtime blue-green redeployment of a supervised component.

        1. Instantiates and boots the green (new) component.
        2. Verifies green component is healthy.
        3. Drains in-flight requests from the blue (old) component.
        4. Stops blue component and replaces reference in-place.
        """
        async with self._lock:
            if name not in self._specs:
                raise KeyError(f"Component '{name}' is not registered.")

            spec = self._specs[name]
            old_instance = spec.instance
            old_task = spec.task

            logger.info("Initiating blue-green redeploy for component '%s'...", name)

            # 1. Instantiate and start green component
            new_instance = new_factory()
            await new_instance.start()
            health = await new_instance.health()
            if health.status == ComponentHealth.FAILED:
                await new_instance.stop()
                raise RuntimeError(f"Green component '{name}' failed health check during redeploy.")

            # 2. Swap registry reference to green
            spec.factory = new_factory
            spec.instance = new_instance
            spec.last_start_time = time.time()
            spec.crash_count = 0

            # 3. Drain and stop blue component
            if old_instance:
                try:
                    await old_instance.drain(timeout=drain_timeout)
                except Exception as e:
                    logger.warning("Error draining old component '%s': %s", name, e)

                try:
                    await old_instance.stop()
                except Exception as e:
                    logger.warning("Error stopping old component '%s': %s", name, e)

            if old_task and not old_task.done():
                old_task.cancel()

            logger.info("Blue-green redeploy for component '%s' succeeded.", name)
            return new_instance

    async def get_health(self) -> dict[str, HealthStatus]:
        """Collects health reports from all supervised components."""
        statuses: dict[str, HealthStatus] = {}
        for name, spec in self._specs.items():
            if spec.instance:
                statuses[name] = await spec.instance.health()
            else:
                statuses[name] = HealthStatus(
                    component_name=name,
                    status=ComponentHealth.STOPPED,
                    uptime_seconds=0.0,
                    error_count=spec.crash_count,
                    last_error="Instance not created",
                )
        return statuses

    async def _start_child(self, spec: ChildSpec) -> None:
        """Instantiates and starts a single child component."""
        try:
            instance = spec.factory()
            spec.instance = instance
            spec.last_start_time = time.time()
            await instance.start()
            await asyncio.sleep(0.01)
            logger.info("Started component '%s'.", spec.name)
        except Exception as e:
            spec.crash_count += 1
            spec.last_crash_time = time.time()
            logger.error("Failed to start component '%s': %s", spec.name, e)
            if spec.instance:
                spec.instance.record_error(e)

    async def _stop_child(self, spec: ChildSpec) -> None:
        """Drains and stops a single child component."""
        if spec.task and not spec.task.done():
            spec.task.cancel()
            spec.task = None

        if spec.instance:
            try:
                await spec.instance.drain(timeout=1.0)
            except Exception as e:
                logger.warning("Error draining component '%s' during shutdown: %s", spec.name, e)
            try:
                await spec.instance.stop()
            except Exception as e:
                logger.warning("Error stopping component '%s' during shutdown: %s", spec.name, e)
            spec.instance = None

    async def _supervision_loop(self) -> None:
        """Continuous background monitor loop checking child component health."""
        while self._is_running:
            try:
                now = time.time()
                for name, spec in list(self._specs.items()):
                    if not self._is_running:
                        break

                    # Check if crash counter can be reset after stability window
                    if (
                        spec.instance
                        and spec.instance.is_running
                        and spec.crash_count > 0
                        and (now - spec.last_start_time > spec.stability_window)
                    ):
                        logger.debug("Component '%s' has remained stable; resetting crash counter.", name)
                        spec.crash_count = 0

                    # Check health
                    needs_restart = False
                    reason = ""
                    if spec.instance is None:
                        needs_restart = True
                        reason = "instance is None"
                    elif not spec.instance.is_running:
                        needs_restart = True
                        reason = "is_running is False"
                    else:
                        health = await spec.instance.health()
                        if health.status == ComponentHealth.FAILED:
                            needs_restart = True
                            reason = f"Health status is FAILED (last error: {health.last_error})"

                    if needs_restart and self._is_running:
                        spec.crash_count += 1
                        spec.last_crash_time = now
                        backoff = min(
                            spec.min_backoff * (spec.backoff_factor ** max(0, spec.crash_count - 1)),
                            spec.max_backoff,
                        )
                        logger.warning(
                            "Component '%s' crashed (%s). Restart #%d scheduled after %.2fs backoff.",
                            name,
                            reason,
                            spec.crash_count,
                            backoff,
                        )

                        # Clean up failed instance
                        if spec.instance:
                            try:
                                await spec.instance.stop()
                            except Exception:
                                pass
                            spec.instance = None

                        # Wait backoff duration
                        await asyncio.sleep(backoff)

                        if self._is_running:
                            logger.info("Attempting restart of component '%s'...", name)
                            await self._start_child(spec)

                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Unexpected error in supervisor loop: %s", e)
                await asyncio.sleep(self.check_interval)
