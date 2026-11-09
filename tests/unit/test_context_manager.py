"""Unit tests for ContextManager (thinking token pruning and token budget enforcement)."""

from mindmirror.llm.context.manager import ContextManager, MessageRole
from mindmirror.pipeline.message import ReasoningStep


def test_context_manager_prunes_thinking_and_reasoning_rollouts():
    """Validates that prune_turn_history purges thinking blocks and reasoning search steps

    while preserving user and assistant messages.
    """
    ctx = ContextManager(system_prompt="You are MindMirror assistant.")

    # Turn 1
    ctx.add_user_message("What is the weather today?")
    ctx.add_thinking("Internal contemplation: user wants weather in their location.")
    ctx.add_reasoning_step(
        ReasoningStep(
            algorithm="tree_of_thoughts",
            step_id="step_1",
            parent_id=None,
            action="branch_generated",
            thought="Check current weather forecast tool",
        )
    )
    ctx.add_tool_interaction(
        tool_name="web_search",
        arguments={"query": "weather"},
        result="Sunny and 22C",
        is_speculative=True,
    )
    ctx.add_assistant_message("[EXCITED] It's 22C and sunny outside!")

    # Before pruning: contains all messages including thinking and reasoning
    all_msgs = ctx.get_messages(include_thinking=True)
    roles_before = [m["role"] for m in all_msgs]
    assert MessageRole.THINKING.value in roles_before
    assert MessageRole.REASONING_STEP.value in roles_before

    # Execute pruning on turn completion
    ctx.prune_turn_history()

    # After pruning: only system, user, and final assistant message remain
    pruned_msgs = ctx.get_messages(include_thinking=True)
    roles_after = [m["role"] for m in pruned_msgs]
    assert roles_after == [MessageRole.SYSTEM.value, MessageRole.USER.value, MessageRole.ASSISTANT.value]

    # Contents are preserved
    assert pruned_msgs[0]["content"] == "You are MindMirror assistant."
    assert pruned_msgs[1]["content"] == "What is the weather today?"
    assert pruned_msgs[2]["content"] == "[EXCITED] It's 22C and sunny outside!"


def test_context_manager_token_budget_enforcement():
    """Validates that enforce_token_budget removes oldest non-system turns when budget is exceeded."""
    ctx = ContextManager(max_token_budget=100, system_prompt="System Directive")

    # Add several long turns
    for i in range(5):
        ctx.add_user_message(f"Turn {i}: " + "word " * 50)
        ctx.add_assistant_message(f"Answer {i}: " + "reply " * 50)

    # Initial token estimation should exceed 100
    assert ctx.estimate_tokens() > 100

    ctx.enforce_token_budget(budget=80)

    # Context should now be under or equal to budget (or trimmed down)
    assert ctx.estimate_tokens() <= 100
    # System message must always remain the first message
    assert ctx.messages[0].role == MessageRole.SYSTEM
    assert ctx.messages[0].content == "System Directive"
