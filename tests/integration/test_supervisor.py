"""Integration tests for ComponentSupervisor (crash recovery, backoff, and zero-downtime redeploy)."""

import asyncio
import time
import pytest

from mindmirror.pipeline.component import ComponentHealth, HealthStatus, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport.local import LocalBus


class FlakyWorkerComponent(PipelineComponent):
    """Test component that can be triggered to simulate a fatal unhandled crash."""

    def __init__(self, name: str, bus: LocalBus, should_fail_on_start: bool = False) -> None:
        super().__init__(name=name, bus=bus)
        self.should_fail_on_start = should_fail_on_start
        self.processed_messages: list[str] = []
        self._sub_id: str | None = None

    async def start(self) -> None:
        if self.should_fail_on_start:
            raise RuntimeError("Fatal initialization failure!")
        await super().start()
        self.create_subscription_task(f"{self.name}.input", self._on_message)
        await asyncio.sleep(0.01)

    async def stop(self) -> None:
        await super().stop()

    async def _on_message(self, message: PipelineMessage) -> None:
        if message.payload == "CRASH_NOW":
            self._status = ComponentHealth.FAILED
            self.record_error("Simulated unhandled worker crash")
            self._is_running = False
            return
        self.processed_messages.append(str(message.payload))


@pytest.mark.asyncio
async def test_supervisor_registration_and_lifecycle():
    """Validates that ComponentSupervisor starts and stops children cleanly."""
    bus = LocalBus()
    supervisor = ComponentSupervisor(check_interval=0.05)

    comp_instances = []

    def factory():
        c = FlakyWorkerComponent("worker1", bus)
        comp_instances.append(c)
        return c

    supervisor.register("worker1", factory)
    await supervisor.start()

    assert supervisor.is_running
    worker = supervisor.get_component("worker1")
    assert worker is not None
    assert worker.is_running

    health = await supervisor.get_health()
    assert "worker1" in health
    assert health["worker1"].status == ComponentHealth.HEALTHY

    await supervisor.stop()
    assert not supervisor.is_running
    assert not worker.is_running
    await bus.close()


@pytest.mark.asyncio
async def test_supervisor_crash_detection_and_automatic_recovery():
    """Validates that when a component crashes, the supervisor detects it and restarts it."""
    bus = LocalBus()
    supervisor = ComponentSupervisor(check_interval=0.05)

    instantiations = 0

    def factory():
        nonlocal instantiations
        instantiations += 1
        return FlakyWorkerComponent(
            "flaky_worker",
            bus,
            should_fail_on_start=False,
        )

    # Fast backoff for test speed: 0.1s min_backoff
    supervisor.register("flaky_worker", factory, min_backoff=0.1, max_backoff=1.0, backoff_factor=2.0)
    await supervisor.start()

    first_instance = supervisor.get_component("flaky_worker")
    assert first_instance is not None
    assert first_instance.is_running
    assert instantiations == 1

    # Publish normal message
    await bus.publish(
        "flaky_worker.input",
        PipelineMessage(channel="flaky_worker.input", payload="hello"),
    )
    await asyncio.sleep(0.05)
    assert first_instance.processed_messages == ["hello"]

    # Trigger simulated crash
    await bus.publish(
        "flaky_worker.input",
        PipelineMessage(channel="flaky_worker.input", payload="CRASH_NOW"),
    )
    await asyncio.sleep(0.05)

    # Component is now marked FAILED / not running
    assert not first_instance.is_running

    # Wait for supervisor to restart it within backoff window (~0.1s - 0.3s)
    for _ in range(20):
        await asyncio.sleep(0.05)
        current = supervisor.get_component("flaky_worker")
        if current and current is not first_instance and current.is_running:
            break

    restarted_instance = supervisor.get_component("flaky_worker")
    assert restarted_instance is not None
    assert restarted_instance is not first_instance
    assert restarted_instance.is_running
    assert instantiations >= 2

    child_spec = supervisor.get_child_spec("flaky_worker")
    assert child_spec.crash_count >= 1

    # Allow restarted component's subscription loop to register with LocalBus
    await asyncio.sleep(0.03)

    # Send message to restarted component
    await bus.publish(
        "flaky_worker.input",
        PipelineMessage(channel="flaky_worker.input", payload="recovered"),
    )
    for _ in range(40):
        await asyncio.sleep(0.02)
        if restarted_instance.processed_messages:
            break
    assert restarted_instance.processed_messages == ["recovered"]

    await supervisor.stop()
    await bus.close()


@pytest.mark.asyncio
async def test_supervisor_blue_green_redeploy():
    """Validates zero-downtime blue-green redeployment of a supervised component."""
    bus = LocalBus()
    supervisor = ComponentSupervisor(check_interval=0.05)

    class VersionedComponent(PipelineComponent):
        def __init__(self, name: str, version: str, bus: LocalBus) -> None:
            super().__init__(name=name, bus=bus)
            self.version = version
            self.drained = False

        async def start(self) -> None:
            await super().start()

        async def drain(self, timeout: float = 2.0) -> None:
            self.drained = True

        async def stop(self) -> None:
            await super().stop()

    # Blue (v1) deployment
    supervisor.register("app_service", lambda: VersionedComponent("app_service", "v1", bus))
    await supervisor.start()

    blue_instance = supervisor.get_component("app_service")
    assert blue_instance.version == "v1"
    assert blue_instance.is_running

    # Zero-downtime redeploy to Green (v2)
    green_instance = await supervisor.redeploy(
        "app_service",
        lambda: VersionedComponent("app_service", "v2", bus),
        drain_timeout=0.1,
    )

    # Blue should have been drained and stopped
    assert blue_instance.drained
    assert not blue_instance.is_running

    # Green should be live in registry
    assert green_instance.version == "v2"
    assert green_instance.is_running
    assert supervisor.get_component("app_service") is green_instance

    await supervisor.stop()
    await bus.close()
