"""Reasoning strategy interface and discrete algorithm search contracts."""

from abc import ABC, abstractmethod
from typing import Any

from mindmirror.pipeline.message import PipelineMessage, ReasoningStep, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus


class ReasoningStrategy(ABC):
    """Abstract Strategy defining how MindMirror reasons, deliberates, and emits responses."""

    @abstractmethod
    async def reason_and_respond(
        self,
        prompt: str,
        bus: MessageBus,
        trace_id: str | None = None,
        speaker_id: str | None = None,
    ) -> str:
        """Executes reasoning deliberation over the input prompt.

        Contracts:
        1. Emits discrete algorithm steps to channel 'reasoning.trace' with StreamPhase.REASONING_STEP.
        2. Emits internal thinking deltas to channel 'ttt.thinking' with StreamPhase.THINKING.
        3. Emits final vocalised text with style tag ([NEUTRAL], [EXCITED], [SERIOUS], [LAZY])
           to channel 'ttt.response' with StreamPhase.RESPONDING.
        4. Returns the final vocalised response string.
        """
        pass
