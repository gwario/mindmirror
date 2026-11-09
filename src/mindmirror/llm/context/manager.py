"""Reasoning-aware context window manager for thinking token pruning and token budget enforcement."""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any

from mindmirror.pipeline.message import ReasoningStep

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    THINKING = "thinking"
    REASONING_STEP = "reasoning_step"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class ContextMessage:
    role: MessageRole
    content: str
    speaker_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "speaker_id": self.speaker_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ContextManager:
    """Manages multi-turn conversation history, prunes intermediate reasoning/thinking deltas,

    and enforces sliding-window token budgets.
    """

    def __init__(self, max_token_budget: int = 4096, system_prompt: str | None = None) -> None:
        self.max_token_budget = max_token_budget
        self._messages: list[ContextMessage] = []
        if system_prompt:
            self._messages.append(ContextMessage(role=MessageRole.SYSTEM, content=system_prompt))

    @property
    def messages(self) -> list[ContextMessage]:
        return list(self._messages)

    def add_user_message(self, content: str, speaker_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._messages.append(
            ContextMessage(
                role=MessageRole.USER,
                content=content,
                speaker_id=speaker_id,
                metadata=metadata or {},
            )
        )

    def add_thinking(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Adds internal LLM deliberation / thinking block (will be pruned on turn completion)."""
        self._messages.append(
            ContextMessage(
                role=MessageRole.THINKING,
                content=content,
                metadata=metadata or {},
            )
        )

    def add_reasoning_step(self, step: ReasoningStep | dict[str, Any]) -> None:
        """Adds discrete search algorithm step (will be pruned on turn completion)."""
        content = step.thought if isinstance(step, ReasoningStep) else str(step.get("thought", ""))
        meta = step.to_dict() if isinstance(step, ReasoningStep) else step
        self._messages.append(
            ContextMessage(
                role=MessageRole.REASONING_STEP,
                content=content,
                metadata=meta,
            )
        )

    def add_tool_interaction(self, tool_name: str, arguments: dict[str, Any], result: str, is_speculative: bool = False) -> None:
        self._messages.append(
            ContextMessage(
                role=MessageRole.TOOL_CALL,
                content=f"Call tool '{tool_name}' with {arguments}",
                metadata={"tool_name": tool_name, "arguments": arguments, "is_speculative": is_speculative},
            )
        )
        self._messages.append(
            ContextMessage(
                role=MessageRole.TOOL_RESULT,
                content=result,
                metadata={"tool_name": tool_name, "is_speculative": is_speculative},
            )
        )

    def add_assistant_message(self, content: str, speaker_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Adds final vocalised assistant answer."""
        self._messages.append(
            ContextMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                speaker_id=speaker_id,
                metadata=metadata or {},
            )
        )

    def prune_turn_history(self) -> None:
        """Prunes historical thinking blocks, reasoning exploration steps, and speculative tool calls.

        Retains only SYSTEM, USER, and final ASSISTANT answers (and confirmed non-speculative tools)
        to prevent context window bloat and catastrophic token exhaustion across turns.
        """
        pruned: list[ContextMessage] = []
        for msg in self._messages:
            if msg.role in (MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT):
                pruned.append(msg)
            elif msg.role in (MessageRole.TOOL_CALL, MessageRole.TOOL_RESULT):
                # Keep confirmed non-speculative tool interactions if desired
                if not msg.metadata.get("is_speculative", False):
                    pruned.append(msg)
            # THINKING and REASONING_STEP are completely purged from long-term turn history
        self._messages = pruned

    def estimate_tokens(self) -> int:
        """Fast character-based heuristic estimation of context token count (~4 characters per token)."""
        total_chars = sum(len(m.content) for m in self._messages)
        return max(1, total_chars // 4)

    def enforce_token_budget(self, budget: int | None = None) -> None:
        """Trims oldest conversation turns (preserving system prompt) until within token budget."""
        target_budget = budget or self.max_token_budget
        while self.estimate_tokens() > target_budget and len(self._messages) > 2:
            # Find oldest non-system message index
            idx_to_remove = -1
            for i, msg in enumerate(self._messages):
                if msg.role != MessageRole.SYSTEM:
                    idx_to_remove = i
                    break
            if idx_to_remove >= 0:
                self._messages.pop(idx_to_remove)
            else:
                break

    def get_messages(self, include_thinking: bool = False) -> list[dict[str, Any]]:
        """Returns message dictionary representation, optionally filtering out thinking tokens."""
        res = []
        for m in self._messages:
            if not include_thinking and m.role in (MessageRole.THINKING, MessageRole.REASONING_STEP):
                continue
            res.append(m.to_dict())
        return res
