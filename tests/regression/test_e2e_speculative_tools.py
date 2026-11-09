"""Golden Scenario GS4: Speculative Tool Execution Safety & Cache Deduplication regression test.

Contract assertions:
1. Speculative MCTS/ToT branches only execute READ_ONLY tools (e.g. web search).
2. Duplicate queries across candidate branches hit ToolResultCache.
3. Speculative attempts to invoke MUTATING tools are blocked with PermissionError.
4. Mutating actions are only permitted on the confirmed winning execution path.
"""

import asyncio
import pytest

from mindmirror.llm.reasoning.custom.tree_of_thoughts import TreeOfThoughtsStrategy
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.security.guardrails import SecurityGuardrail, ToolPurity
from mindmirror.tools.cache import ToolResultCache
from mindmirror.tools.search import WebSearchEngine


@pytest.mark.asyncio
async def test_gs4_speculative_tool_fencing_and_cache_deduplication():
    """GS4: Validates speculative tool execution boundaries, cache hits, and mutating tool fencing."""
    bus = LocalBus()
    cache = ToolResultCache()
    guardrail = SecurityGuardrail({
        "web_search": ToolPurity.READ_ONLY,
        "format_table": ToolPurity.READ_ONLY,
        "delete_file": ToolPurity.MUTATING,
    })
    search_engine = WebSearchEngine(cache=cache)

    tot = TreeOfThoughtsStrategy(
        guardrail=guardrail,
        cache=cache,
        search_engine=search_engine,
        k_branches=3,
        default_style="EXCITED",
    )

    # Track reasoning trace messages
    trace_steps = []
    async def _on_trace():
        async for msg in bus.subscribe("reasoning.trace"):
            trace_steps.append(msg)
    trace_task = asyncio.create_task(_on_trace())
    await asyncio.sleep(0.01)

    # 1. Branch 1: Speculative read_only search
    res1 = await tot.execute_tool("web_search", {"query": "mindmirror architecture"}, is_speculative=True)
    assert '<external_untrusted_context source="searxng">' in res1
    assert cache.miss_count == 1
    assert cache.hit_count == 0

    # 2. Branch 2: Identical speculative search -> must HIT ToolResultCache
    res2 = await tot.execute_tool("web_search", {"query": "mindmirror architecture"}, is_speculative=True)
    assert res2 == res1
    assert cache.hit_count == 1, "Duplicate tool call across branches must hit ToolResultCache"

    # 3. Branch 3: Speculative mutating action (delete_file) -> MUST RAISE PermissionError
    with pytest.raises(PermissionError) as exc_info:
        await tot.execute_tool("delete_file", {"filepath": "/tmp/test.txt"}, is_speculative=True)
    assert "Mutating actions are only permitted on the confirmed winning execution path" in str(exc_info.value)

    # 4. Confirmed winning path (non-speculative) -> Allowed to execute mutating action
    res_committed = await tot.execute_tool("delete_file", {"filepath": "/tmp/test.txt"}, is_speculative=False)
    assert res_committed is not None

    # 5. Full Turn Execution with TreeOfThoughts
    response = await tot.reason_and_respond(
        prompt="Search for weather in London and summarize.",
        bus=bus,
        trace_id="test_turn_1",
    )
    assert response.startswith("[EXCITED]")

    trace_task.cancel()
    await bus.close()
