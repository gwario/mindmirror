"""Proof of Concept (PoC) Custom Symbolic & Heuristic Reasoning Strategy.

Demonstrates how developers can attach an unconstrained, specialized custom reasoning
engine (e.g. symbolic solver, heuristic evaluator, MCTS, or constraint logic) into
MindMirror's pipeline.
"""

import asyncio
import logging
import time
from typing import Any, Optional
import uuid

from mindmirror.llm.reasoning.interface import ReasoningStrategy
from mindmirror.llm.reasoning.registry import ReasoningStrategyRegistry
from mindmirror.pipeline.message import PipelineMessage, ReasoningStep, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class SymbolicHeuristicReasoningStrategy(ReasoningStrategy):
    """Custom Reasoning Engine PoC.

    Executes a 3-stage deliberation pipeline:
    Stage 1: Intent Decomposition & Constraint Extraction
    Stage 2: Heuristic Candidate Search & Evaluation (emitting discrete ReasoningSteps)
    Stage 3: Empathy & Emotional Style Synthesis
    """

    def __init__(
        self,
        heuristic_weights: Optional[dict[str, float]] = None,
        default_style: str = "EXCITED",
    ) -> None:
        self.heuristic_weights = heuristic_weights or {"empathy": 0.4, "clarity": 0.4, "conciseness": 0.2}
        self.default_style = default_style

    async def reason_and_respond(
        self,
        prompt: str,
        bus: MessageBus,
        trace_id: str | None = None,
        speaker_id: str | None = None,
    ) -> str:
        tid = trace_id or str(uuid.uuid4())
        logger.info("Custom SymbolicHeuristicReasoning executing for prompt: '%s'", prompt)

        # ---------------------------------------------------------------------
        # Stage 1: Intent Decomposition
        # ---------------------------------------------------------------------
        step1_id = str(uuid.uuid4())
        step1 = ReasoningStep(
            algorithm="symbolic_heuristic",
            step_id=step1_id,
            parent_id=None,
            action="intent_decomposition",
            thought=f"Extracted symbolic entities and constraints from query: '{prompt[:40]}...'",
            score=0.95,
        )
        await bus.publish(
            "reasoning.trace",
            PipelineMessage(
                channel="reasoning.trace",
                phase=StreamPhase.REASONING_STEP,
                payload=step1,
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )

        # Emit internal thinking stream
        await bus.publish(
            "ttt.thinking",
            PipelineMessage(
                channel="ttt.thinking",
                phase=StreamPhase.THINKING,
                payload="Analyzing acoustic semantics and calculating optimal conversational stance...",
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )
        await asyncio.sleep(0.02)

        # ---------------------------------------------------------------------
        # Stage 2: Heuristic Candidate Exploration
        # ---------------------------------------------------------------------
        candidates = [
            {"style": "EXCITED", "text": f"Brilliant question, mate! Based on custom reasoning, '{prompt}' has been evaluated."},
            {"style": "SERIOUS", "text": f"Understood. We have prioritized safety and precision for '{prompt}'."},
            {"style": "NEUTRAL", "text": f"Here is the synthesized response to '{prompt}'."},
        ]

        best_candidate = candidates[0] if "!" in prompt or "quick" in prompt.lower() else candidates[0]
        step2_id = str(uuid.uuid4())
        step2 = ReasoningStep(
            algorithm="symbolic_heuristic",
            step_id=step2_id,
            parent_id=step1_id,
            action="candidate_selected",
            thought=f"Evaluated {len(candidates)} candidate branches; selected style '{best_candidate['style']}'.",
            score=0.98,
        )
        await bus.publish(
            "reasoning.trace",
            PipelineMessage(
                channel="reasoning.trace",
                phase=StreamPhase.REASONING_STEP,
                payload=step2,
                trace_id=tid,
                speaker_id=speaker_id,
            ),
        )

        # ---------------------------------------------------------------------
        # Stage 3: Emotional Style Formatting & Emission
        # ---------------------------------------------------------------------
        response_text = f"[{best_candidate['style']}] {best_candidate['text']}"

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


# Automatically register PoC strategy in registry
ReasoningStrategyRegistry.register("symbolic_poc", SymbolicHeuristicReasoningStrategy)
