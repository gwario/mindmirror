"""Unit tests for pluggable custom reasoning strategies and SymbolicHeuristicReasoningStrategy PoC."""

import asyncio
import pytest

from mindmirror.llm.reasoning.custom.poc_symbolic_reasoner import SymbolicHeuristicReasoningStrategy
from mindmirror.llm.reasoning.registry import ReasoningStrategyRegistry
from mindmirror.pipeline.message import PipelineMessage, ReasoningStep, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus


@pytest.mark.asyncio
async def test_reasoning_strategy_registry_and_poc_execution():
    """Validates that custom reasoning engines can be registered and executed through the bus."""
    # 1. Verify PoC is registered
    strategies = ReasoningStrategyRegistry.list_strategies()
    assert "symbolic_poc" in strategies

    strategy = ReasoningStrategyRegistry.get("symbolic_poc")
    assert isinstance(strategy, SymbolicHeuristicReasoningStrategy)

    # 2. Execute PoC with LocalBus probes
    bus = LocalBus()
    reasoning_traces = []
    thinking_tokens = []
    responses = []

    async def _on_trace():
        async for msg in bus.subscribe("reasoning.trace"):
            reasoning_traces.append(msg)

    async def _on_thinking():
        async for msg in bus.subscribe("ttt.thinking"):
            thinking_tokens.append(msg)

    async def _on_response():
        async for msg in bus.subscribe("ttt.response"):
            responses.append(msg)

    t1 = asyncio.create_task(_on_trace())
    t2 = asyncio.create_task(_on_thinking())
    t3 = asyncio.create_task(_on_response())
    await asyncio.sleep(0.01)

    result = await strategy.reason_and_respond(
        prompt="Explain quantum encryption quickly!",
        bus=bus,
        trace_id="test-custom-trace",
        speaker_id="mario",
    )

    await asyncio.sleep(0.05)
    t1.cancel()
    t2.cancel()
    t3.cancel()
    await bus.close()

    # Invariants:
    # 1. Output string begins with valid style tag
    assert result.startswith("[EXCITED]") or result.startswith("[SERIOUS]") or result.startswith("[NEUTRAL]")

    # 2. Discrete reasoning algorithm steps emitted to reasoning.trace
    assert len(reasoning_traces) >= 2
    step1 = reasoning_traces[0].payload
    assert isinstance(step1, ReasoningStep)
    assert step1.algorithm == "symbolic_heuristic"
    assert step1.action == "intent_decomposition"

    # 3. Thinking stream emitted
    assert len(thinking_tokens) >= 1

    # 4. Response emitted to ttt.response
    assert len(responses) == 1
    assert responses[0].payload == result
