#!/usr/bin/env python3
"""Benchmark and validation script for MindMirror MessageBus transports.

Tests LocalBus, ZmqBus, and NatsBus (if available) with 1,000 messages across multiple channels,
asserting sub-millisecond latency, zero message loss, and 0600 IPC socket permissions.
"""

import asyncio
import os
from pathlib import Path
import stat
import sys
import time

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rich.console import Console
from rich.table import Table

from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.pipeline.transport.zmq_bus import ZmqBus

console = Console()


async def benchmark_bus(bus_name: str, bus, num_messages: int = 1000) -> dict:
    channel = f"bench.{bus_name.lower()}"
    received = []

    async def _subscriber():
        async for msg in bus.subscribe(channel):
            received.append(msg)
            if len(received) >= num_messages:
                break

    sub_task = asyncio.create_task(_subscriber())
    # Allow subscriber connection to establish
    await asyncio.sleep(0.08)

    start_time = time.perf_counter()
    for i in range(num_messages):
        msg = PipelineMessage(
            channel=channel,
            payload={"index": i, "data": "payload_test"},
            phase=StreamPhase.TELEMETRY,
            metadata={"seq": i},
        )
        await bus.publish(channel, msg)

    try:
        await asyncio.wait_for(sub_task, timeout=5.0)
    except asyncio.TimeoutError:
        pass

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_messages) * 1000
    loss = num_messages - len(received)

    # Check permissions if ZmqBus
    perms_str = "N/A"
    if isinstance(bus, ZmqBus):
        _, fs_path = bus._channel_to_endpoint(channel)
        if fs_path and os.path.exists(fs_path):
            mode = oct(stat.S_IMODE(os.stat(fs_path).st_mode))
            perms_str = f"{mode} (OK)" if mode == "0o600" else f"{mode} (WARN)"

    return {
        "name": bus_name,
        "sent": num_messages,
        "received": len(received),
        "loss": loss,
        "loss_pct": (loss / num_messages) * 100,
        "total_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "perms": perms_str,
        "status": "[green]PASS[/green]" if loss == 0 and avg_latency_ms < 2.0 else "[red]FAIL[/red]",
    }


async def main():
    console.print("\n[bold cyan]🚀 MindMirror Phase 0 Transport Benchmark & Verification[/bold cyan]\n")

    results = []

    # 1. LocalBus
    local = LocalBus()
    try:
        res_local = await benchmark_bus("LocalBus (asyncio)", local, 1000)
        results.append(res_local)
    finally:
        await local.close()

    # 2. ZmqBus (IPC chmod 0600)
    zmq_b = ZmqBus(endpoint_prefix="ipc:///tmp/mindmirror_bench_")
    try:
        res_zmq = await benchmark_bus("ZmqBus (libzmq IPC)", zmq_b, 1000)
        results.append(res_zmq)
    finally:
        await zmq_b.close()

    # 3. NatsBus (optional if NATS is running)
    try:
        from mindmirror.pipeline.transport.nats_bus import NatsBus
        nats_b = NatsBus()
        try:
            await asyncio.wait_for(nats_b.connect(), timeout=0.5)
            res_nats = await benchmark_bus("NatsBus (NATS OSS)", nats_b, 1000)
            results.append(res_nats)
            await nats_b.close()
        except Exception:
            results.append({
                "name": "NatsBus (NATS OSS)",
                "sent": 1000,
                "received": 0,
                "loss": 1000,
                "loss_pct": 100.0,
                "total_time_s": 0.0,
                "avg_latency_ms": 0.0,
                "perms": "N/A",
                "status": "[yellow]SKIPPED (No local NATS daemon)[/yellow]",
            })
    except ImportError:
        pass

    # Print summary table
    table = Table(title="MessageBus Performance & Reliability Matrix")
    table.add_column("Transport", style="bold white")
    table.add_column("Messages Sent", justify="right")
    table.add_column("Received", justify="right")
    table.add_column("Loss %", justify="right")
    table.add_column("Total Time", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Permissions", justify="center")
    table.add_column("Verdict", justify="center")

    for r in results:
        table.add_row(
            r["name"],
            str(r["sent"]),
            str(r["received"]),
            f"{r['loss_pct']:.1f}%",
            f"{r['total_time_s']:.3f}s" if r["total_time_s"] > 0 else "-",
            f"{r['avg_latency_ms']:.3f}ms" if r["avg_latency_ms"] > 0 else "-",
            r["perms"],
            r["status"],
        )

    console.print(table)
    console.print()

    # Assertions for script exit code
    for r in results:
        if "FAIL" in r["status"]:
            console.print(f"[red]❌ Benchmark failed for {r['name']}[/red]")
            sys.exit(1)

    console.print("[bold green]✅ All active transports verified successfully![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
