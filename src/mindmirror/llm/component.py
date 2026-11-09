"""Supervised Reasoning / TTT pipeline component connecting stt.transcript to ttt.response."""

import asyncio
import logging
from typing import Any

from mindmirror.llm.context.manager import ContextManager
from mindmirror.llm.reasoning.interface import ReasoningStrategy
from mindmirror.pipeline.component import PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.telemetry import TurnSpanTracker
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


class ReasoningComponent(PipelineComponent):
    """Supervised component executing reasoning strategies and managing conversation context."""

    def __init__(
        self,
        bus: MessageBus,
        strategy: ReasoningStrategy,
        context_manager: ContextManager | None = None,
        speaker_context: Any | None = None,
        memory_manager: Any | None = None,
        name: str = "reasoning_engine",
    ) -> None:
        super().__init__(name=name, bus=bus)
        self.strategy = strategy
        self.context_manager = context_manager or ContextManager()
        self.speaker_context = speaker_context
        self.memory_manager = memory_manager
        self._turn_lock = asyncio.Lock()

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("stt.transcript", self._handle_transcript)
        await asyncio.sleep(0.01)
        logger.info("ReasoningComponent '%s' started, listening to 'stt.transcript'.", self.name)

    async def stop(self) -> None:
        await super().stop()
        logger.info("ReasoningComponent '%s' stopped.", self.name)

    async def _handle_transcript(self, message: PipelineMessage) -> None:
        transcript = str(message.payload) if message.payload else ""
        if not transcript.strip():
            return

        resolved_speaker = message.speaker_id
        if (not resolved_speaker or resolved_speaker == "unresolved") and self.speaker_context and self.speaker_context.is_resolved:
            resolved_speaker = self.speaker_context.current_speaker_id

        async with self._turn_lock:
            tracker = TurnSpanTracker(
                trace_id=message.trace_id,
                bus=self.bus,
                speaker_id=resolved_speaker,
            )
            tracker.start_span("reasoning")

            # 1. Update conversational context with user turn
            self.context_manager.add_user_message(transcript, speaker_id=resolved_speaker)

            try:
                # 2. Execute reasoning strategy
                response = await self.strategy.reason_and_respond(
                    prompt=transcript,
                    bus=self.bus,
                    trace_id=message.trace_id,
                    speaker_id=message.speaker_id,
                )

                # 3. Store assistant response in context
                self.context_manager.add_assistant_message(response, speaker_id=message.speaker_id)

                # 4. Prune intermediate reasoning/thinking deltas from context memory
                self.context_manager.prune_turn_history()

                # 5. Enforce token budget
                self.context_manager.enforce_token_budget()

            except Exception as e:
                self.record_error(e)
                logger.error("Error during reasoning execution: %s", e)
            finally:
                tracker.end_span("reasoning")
                await tracker.finalize_and_publish()
