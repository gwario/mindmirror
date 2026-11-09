"""Local LLM Reasoning Strategy supporting Ollama, vLLM, and llama.cpp local endpoints."""

import json
import logging
import os
import re
from typing import Any, Optional
import httpx

from mindmirror.llm.reasoning.interface import ReasoningStrategy
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class LocalLLMReasoningStrategy(ReasoningStrategy):
    """Reasoning strategy connecting to a local OSS LLM server (Ollama, vLLM, or llama.cpp).

    Supports:
    - Ollama API (http://localhost:11434/api/generate or /v1/chat/completions)
    - vLLM / llama.cpp OpenAI-compatible server (http://localhost:8000/v1/chat/completions)
    - Streaming thinking blocks and emotional style tag enforcement
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        default_style: str = "NEUTRAL",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCAL_LLM_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("LOCAL_LLM_MODEL") or "llama3.2:latest"
        self.default_style = default_style
        self.timeout = timeout

    async def reason_and_respond(
        self,
        prompt: str,
        bus: MessageBus,
        trace_id: str | None = None,
        speaker_id: str | None = None,
    ) -> str:
        tid = trace_id or "local-llm-trace"
        system_prompt = (
            "You are MindMirror, an empathetic, hyper-responsive cognitive companion. "
            "You MUST begin your response with one of these emotional style tags: "
            "[NEUTRAL], [EXCITED], [SERIOUS], or [LAZY]. "
            "Think carefully before replying."
        )

        # Notify thinking started
        await bus.publish(
            "ttt.thinking",
            PipelineMessage(
                channel="ttt.thinking",
                phase=StreamPhase.THINKING,
                payload=f"Consulting local model '{self.model}'...",
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )

        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        response_text = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    response_text = data["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning("Local LLM error %d: %s; using heuristic fallback", res.status_code, res.text)
                    response_text = f"[{self.default_style}] I received your message through the local engine."
        except Exception as e:
            logger.warning("Could not reach local LLM at %s (%s); falling back to default.", endpoint, e)
            response_text = f"[{self.default_style}] Acknowledged. Operating via local reasoning engine fallback."

        # Ensure valid style tag prefix
        if not re.match(r"^\[(NEUTRAL|EXCITED|SERIOUS|LAZY)\]", response_text, re.IGNORECASE):
            response_text = f"[{self.default_style}] {response_text}"

        # Publish final response
        await bus.publish(
            "ttt.response",
            PipelineMessage(
                channel="ttt.response",
                phase=StreamPhase.RESPONDING,
                payload=response_text,
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )

        return response_text
