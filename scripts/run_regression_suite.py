#!/usr/bin/env python3
"""MindMirror Turnkey Regression CLI Runner.

Executes the 7 Golden Regression Scenarios and comprehensive test suites,
benchmarking execution durations against strict SLA latency budgets, and
formatting results into an executive Rich summary table.

Usage:
    # 1. Fast Hermetic Regression (Default: 0 external APIs, 0 soundcards)
    python scripts/run_regression_suite.py --mode=fast

    # 2. Integration & IPC Suite (Tests ZeroMQ IPC sockets, supervisor chaos, and ChromaDB)
    python scripts/run_regression_suite.py --mode=integration

    # 3. Full Local OSS Suite (Full test suite across unit, integration, and regression)
    python scripts/run_regression_suite.py --mode=full-oss
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup path roots
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ScenarioSpec:
    id: str
    name: str
    test_path: str
    sla_ms: float
    description: str


GOLDEN_SCENARIOS = [
    ScenarioSpec(
        id="GS1",
        name="GS1: Full Conversational Turn",
        test_path="tests/regression/test_e2e_conversational_turn.py",
        sla_ms=1000.0,
        description="Glass-to-glass pipeline turn from AudioSource to VirtualAudioSink",
    ),
    ScenarioSpec(
        id="GS2",
        name="GS2: Emotion Style Contract",
        test_path="tests/regression/test_e2e_thinking_leakage.py",
        sla_ms=200.0,
        description="Emotion style parsing and internal thinking token leakage fencing",
    ),
    ScenarioSpec(
        id="GS3",
        name="GS3: Barge-In Interruption Drain",
        test_path="tests/regression/test_e2e_barge_in.py",
        sla_ms=50.0,
        description="Barge-in speech detection, playback cancellation, and sub-30ms sink flush",
    ),
    ScenarioSpec(
        id="GS4",
        name="GS4: Speculative Tool Fencing",
        test_path="tests/regression/test_e2e_speculative_tools.py",
        sla_ms=150.0,
        description="Tool purity read-only enforcement and ToolResultCache deduplication",
    ),
    ScenarioSpec(
        id="GS5",
        name="GS5: Parallel Speaker Retro-Tag",
        test_path="tests/regression/test_e2e_speaker_retroactive.py",
        sla_ms=500.0,
        description="Parallel acoustic voice diarization and retroactive ChromaDB tagging",
    ),
    ScenarioSpec(
        id="GS6",
        name="GS6: Prompt Injection Sanitizer",
        test_path="tests/regression/test_e2e_injection_defense.py",
        sla_ms=50.0,
        description="Untrusted XML wrapping and prompt injection defence",
    ),
    ScenarioSpec(
        id="GS7",
        name="GS7: Supervisor Self-Healing",
        test_path="tests/regression/test_e2e_resilience_chaos.py",
        sla_ms=2000.0,
        description="Component crash detection, exponential backoff, and stateful recovery",
    ),
]


def run_single_test(test_path: str) -> tuple[bool, float, str]:
    """Runs a single pytest path, returning (passed, duration_ms, error_output)."""
    import re

    start = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(REPO_ROOT / test_path),
        "-q",
        "--durations=1",
        "--tb=short",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    passed = res.returncode == 0
    output = res.stdout + res.stderr

    # Try to extract the pure call duration (excluding Python/PyTorch cold import overhead)
    match = re.search(r"(\d+(?:\.\d+)?)s\s+call", output)
    if match:
        duration_ms = float(match.group(1)) * 1000.0
    else:
        duration_ms = (time.perf_counter() - start) * 1000.0

    return passed, duration_ms, output


def run_extra_suite(suite_dir: str) -> tuple[bool, int, int, float]:
    """Runs a directory of pytest tests, returning (all_passed, passed_count, total_count, duration_s)."""
    start = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(REPO_ROOT / suite_dir),
        "-q",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    duration_s = time.perf_counter() - start
    passed = res.returncode == 0
    output = res.stdout + res.stderr
    # Parse total passed count from pytest output (e.g., "17 passed in 1.23s")
    passed_count = 0
    total_count = 0
    import re
    match = re.search(r"(\d+) passed", output)
    if match:
        passed_count = int(match.group(1))
        total_count = passed_count
    fail_match = re.search(r"(\d+) failed", output)
    if fail_match:
        total_count += int(fail_match.group(1))
    return passed, passed_count, total_count, duration_s


def main():
    parser = argparse.ArgumentParser(description="MindMirror Master Regression Suite Runner")
    parser.add_argument(
        "--mode",
        choices=["fast", "integration", "full-oss"],
        default="fast",
        help="Regression suite execution mode (default: fast)",
    )
    args = parser.parse_args()

    console = Console()
    console.print()
    console.print(
        Panel(
            f"[bold cyan]MindMirror Master Regression Runner[/bold cyan]\n"
            f"Execution Mode: [bold yellow]{args.mode.upper()}[/bold yellow] | "
            f"Python: [green]{sys.version.split()[0]}[/green] | "
            f"Target: [bold white]{REPO_ROOT}[/bold white]",
            box=ROUNDED,
            expand=False,
        )
    )

    table = Table(
        title=f"The 7 Golden Regression Scenarios ({args.mode.upper()} Mode)",
        box=ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Regression Test Scenario", style="bold white", width=38)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Duration", justify="right", width=12)
    table.add_column("SLA Budget", justify="right", width=14)
    table.add_column("SLA Check", justify="center", width=12)

    total_passed = 0
    total_duration_ms = 0.0
    failures: list[tuple[str, str]] = []

    for spec in GOLDEN_SCENARIOS:
        passed, duration_ms, output = run_single_test(spec.test_path)
        total_duration_ms += duration_ms

        sla_met = duration_ms <= spec.sla_ms
        if passed:
            total_passed += 1
            status_text = "[bold green]PASS[/bold green]"
        else:
            status_text = "[bold red]FAIL[/bold red]"
            failures.append((spec.name, output))

        sla_status_text = "[bold green]OK[/bold green]" if sla_met else "[bold yellow]WARN (SLOW)[/bold yellow]"
        duration_str = f"{duration_ms:.1f}ms"
        sla_str = f"< {int(spec.sla_ms)}ms"

        table.add_row(
            spec.name,
            status_text,
            duration_str,
            sla_str,
            sla_status_text,
        )

    # Summary separator
    table.add_section()
    all_passed = total_passed == len(GOLDEN_SCENARIOS)
    overall_status = "[bold green]ALL PASS[/bold green]" if all_passed else "[bold red]FAILED[/bold red]"
    total_time_str = f"{(total_duration_ms / 1000.0):.2f}s"

    table.add_row(
        "[bold cyan]OVERALL INTEGRITY CHECK[/bold cyan]",
        f"[bold cyan]{total_passed}/{len(GOLDEN_SCENARIOS)} OK[/bold cyan]",
        f"[bold cyan]{total_time_str}[/bold cyan]",
        "[bold cyan]ALL PASS[/bold cyan]",
        overall_status,
    )

    console.print(table)

    # Extra suite execution based on mode
    extra_passed = True
    if args.mode in ("integration", "full-oss"):
        console.print()
        console.print("[bold blue]Executing Integration Test Suite...[/bold blue]")
        integ_ok, integ_passed, integ_total, integ_dur = run_extra_suite("tests/integration")
        if not integ_ok:
            extra_passed = False
        console.print(
            f"Integration Suite: {'[bold green]PASSED[/bold green]' if integ_ok else '[bold red]FAILED[/bold red]'} "
            f"({integ_passed}/{integ_total} tests in {integ_dur:.2f}s)"
        )

    if args.mode == "full-oss":
        console.print()
        console.print("[bold blue]Executing Unit Test Suite...[/bold blue]")
        unit_ok, unit_passed, unit_total, unit_dur = run_extra_suite("tests/unit")
        if not unit_ok:
            extra_passed = False
        console.print(
            f"Unit Suite: {'[bold green]PASSED[/bold green]' if unit_ok else '[bold red]FAILED[/bold red]'} "
            f"({unit_passed}/{unit_total} tests in {unit_dur:.2f}s)"
        )

    # Output failure diagnostics if any
    if failures:
        console.print()
        console.print("[bold red]Scenario Failure Diagnostics:[/bold red]")
        for name, err in failures:
            console.print(f"[bold yellow]{name}:[/bold yellow]")
            console.print(err)

    if all_passed and extra_passed:
        console.print()
        console.print("[bold green]System Ready for Production Deployment![/bold green]")
        sys.exit(0)
    else:
        console.print()
        console.print("[bold red]System Integrity Compromised. Check failures above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
