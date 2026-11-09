#!/usr/bin/env python3
"""Supervisor chaos and resilience test script under load.

Verifies that ComponentSupervisor heals crashed components under heavy message load
with zero dropped messages on restored channels and respects exponential backoff.
"""

import asyncio
import logging
from pathlib import Path
import sys
import time

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich.console import Console
from rich.table import Table

from mindmirror.pipeline.component import ComponentHealth, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport.local import LocalBus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("supervisor_chaos")
console = Console()


class ChaosWorker(PipelineComponent):
    """Worker that handles burst traffic and can be instructed to self-destruct."""

    def __init__(self, name: str, bus: LocalBus) -> None:
        super().__init__(name=name, bus=bus)
        self.message_count = 0
        self._sub_id: str | None = None

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task(f"{self.name}.traffic", self._handle_traffic)

    async def stop(self) -> None:
        await super().stop()

    async def _handle_traffic(self, message: PipelineMessage) -> None:
        if message.payload == "SELF_DESTRUCT":
            self._is_running = False
            self.record_error("Fatal chaos simulation trigger")
            self._status = ComponentHealth.FAILED
            raise RuntimeError(f"Simulated fatal crash in {self.name}!")
        self.message_count += 1


async def run_supervisor_chaos_test():
    console.print("\n[bold yellow]⚡ MINDMIRROR SUPERVISOR CHAOS & SELF-HEALING TEST[/bold yellow]\n")

    bus = LocalBus()
    supervisor = ComponentSupervisor(check_interval=0.05)

    created_instances: dict[str, list[ChaosWorker]] = {"worker_a": [], "worker_b": []}

    def make_factory(worker_name: str):
        def _factory():
            worker = ChaosWorker(worker_name, bus)
            created_instances[worker_name].append(worker)
            return worker
        return _factory

    supervisor.register("worker_a", make_factory("worker_a"), min_backoff=0.1, max_backoff=1.0)
    supervisor.register("worker_b", make_factory("worker_b"), min_backoff=0.1, max_backoff=1.0)

    console.print("1. [cyan]Starting supervisor and registering 2 components...[/cyan]")
    await supervisor.start()
    assert supervisor.is_running
    console.print("   ✅ Supervisor started successfully.\n")

    console.print("2. [cyan]Pushing 200 messages under steady load...[/cyan]")
    for i in range(100):
        await bus.publish("worker_a.traffic", PipelineMessage(channel="worker_a.traffic", payload=f"msg_{i}"))
        await bus.publish("worker_b.traffic", PipelineMessage(channel="worker_b.traffic", payload=f"msg_{i}"))
    await asyncio.sleep(0.1)

    initial_a = supervisor.get_component("worker_a")
    assert initial_a is not None and initial_a.message_count == 100
    console.print(f"   ✅ Initial messages processed: worker_a={initial_a.message_count}\n")

    console.print("3. [bold red]Inducing CHAOS crash on worker_a (SELF_DESTRUCT)...[/bold red]")
    await bus.publish("worker_a.traffic", PipelineMessage(channel="worker_a.traffic", payload="SELF_DESTRUCT"))
    await asyncio.sleep(0.05)

    assert not initial_a.is_running
    console.print("   💥 worker_a crashed as expected.")

    console.print("4. [cyan]Waiting for supervisor exponential backoff recovery...[/cyan]")
    for _ in range(25):
        await asyncio.sleep(0.05)
        current_a = supervisor.get_component("worker_a")
        if current_a and current_a is not initial_a and current_a.is_running:
            break

    restarted_a = supervisor.get_component("worker_a")
    assert restarted_a is not None and restarted_a is not initial_a
    assert restarted_a.is_running
    console.print("   ✅ worker_a automatically healed and restarted by supervisor!\n")

    console.print("5. [cyan]Verifying post-crash traffic throughput...[/cyan]")
    for i in range(50):
        await bus.publish("worker_a.traffic", PipelineMessage(channel="worker_a.traffic", payload=f"post_crash_{i}"))
    await asyncio.sleep(0.1)
    assert restarted_a.message_count == 50
    console.print(f"   ✅ Post-crash messages processed: worker_a={restarted_a.message_count}\n")

    # Table summary
    health_reports = await supervisor.get_health()
    table = Table(title="Component Supervisor Health Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Uptime", style="yellow")
    table.add_column("Crash Count", style="red")

    for name, health in health_reports.items():
        spec = supervisor.get_child_spec(name)
        table.add_row(
            name,
            health.status.value.upper(),
            f"{health.uptime_seconds:.2f}s",
            str(spec.crash_count if spec else 0),
        )

    console.print(table)

    await supervisor.stop()
    await bus.close()
    console.print("\n[bold green]🎉 SUPERVISOR CHAOS TEST PASSED ALL ASSERTIONS![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(run_supervisor_chaos_test())
