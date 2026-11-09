"""Provider native reasoning streaming parser (Gemini thinkingBudget, Anthropic, DeepSeek <think>)."""

import asyncio
from collections.abc import Coroutine
import logging
import re
import uuid
from typing import Any, Callable

from mindmirror.llm.reasoning.interface import ReasoningStrategy
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)

THINK_TAG_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


class ProviderNativeReasoningStrategy(ReasoningStrategy):
    """Parses and demuxes native LLM provider reasoning tokens.

    Extracts internal thinking blocks (<think>...</think> or native thinking deltas)
    and publishes them to 'ttt.thinking' while routing only clean spoken speech
    with [STYLE] tags to 'ttt.response'.
    """

    def __init__(
        self,
        llm_generate_fn: Callable[[str], str | Coroutine[Any, Any, str]] | None = None,
        default_style: str = "NEUTRAL",
    ) -> None:
        self.llm_generate_fn = llm_generate_fn
        self.default_style = default_style

    @classmethod
    def demux_thinking_and_response(cls, raw_text: str, default_style: str = "NEUTRAL") -> tuple[str, str]:
        """Splits raw provider response into (thinking_content, vocal_response).

        Guarantees that thinking tokens are completely removed from vocal response text.
        """
        match = THINK_TAG_REGEX.search(raw_text)
        if match:
            thinking = match.group(1).strip()
            # Everything outside the <think> tag is the spoken response
            vocal = THINK_TAG_REGEX.sub("", raw_text).strip()
        else:
            thinking = ""
            vocal = raw_text.strip()

        # Enforce emotion style tag on spoken response
        if vocal and not vocal.startswith("["):
            vocal = f"[{default_style}] {vocal}"

        return thinking, vocal

    async def reason_and_respond(
        self,
        prompt: str,
        bus: MessageBus,
        trace_id: str | None = None,
        speaker_id: str | None = None,
    ) -> str:
        tid = trace_id or str(uuid.uuid4())

        if self.llm_generate_fn:
            res = self.llm_generate_fn(prompt)
            if asyncio.iscoroutine(res):
                res = await res
            raw_text = str(res)
        else:
            raw_text = (
                f"<think>Analyzing user request '{prompt}' through internal cognitive deliberation.</think>\n"
                f"[{self.default_style}] I have processed your request."
            )

        thinking, vocal = self.demux_thinking_and_response(raw_text, default_style=self.default_style)

        # 1. Publish thinking tokens to telemetry/UI channel (strictly excluded from audio output)
        if thinking:
            await bus.publish(
                "ttt.thinking",
                PipelineMessage(
                    channel="ttt.thinking",
                    phase=StreamPhase.THINKING,
                    payload=thinking,
                    trace_id=tid,
                    speaker_id=speaker_id,
                ),
            )

        # 2. Publish vocal response to TTS synthesis channel
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload=vocal,
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )

        return vocal


# Backwards-compatible alias
ProviderNativeThinkingStrategy = ProviderNativeReasoningStrategy

