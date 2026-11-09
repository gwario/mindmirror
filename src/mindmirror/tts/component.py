"""Supervised TTS pipeline component subscribing to ttt.response and publishing audio.playback."""

import asyncio
import logging
import re
from typing import Any

from mindmirror.pipeline.component import ComponentHealth, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus
from mindmirror.tts.interface import TTSInterface

logger = logging.getLogger(__name__)

STYLE_TAG_PATTERN = re.compile(r"^\s*\[(NEUTRAL|EXCITED|SERIOUS|LAZY)\]\s*", re.IGNORECASE)


def parse_style_tag(text: str) -> tuple[str, str]:
    """Parses emotion style tag protocol [NEUTRAL], [EXCITED], [SERIOUS], [LAZY].

    Returns (style_name, clean_text_without_tag).
    Defaults to ('NEUTRAL', text) if no tag is specified.
    """
    match = STYLE_TAG_PATTERN.match(text)
    if match:
        style = match.group(1).upper()
        clean_text = text[match.end() :].strip()
        return style, clean_text
    return "NEUTRAL", text.strip()


class TTSComponent(PipelineComponent):
    """Supervised component synthesizing assistant responses into audio chunks."""

    def __init__(
        self,
        bus: MessageBus,
        tts_engine: TTSInterface,
        profile_store: Any | None = None,
        speaker_context: Any | None = None,
        name: str = "tts_engine",
    ) -> None:
        super().__init__(name=name, bus=bus)
        self.tts_engine = tts_engine
        self.profile_store = profile_store
        self.speaker_context = speaker_context
        self._response_sub_id: str | None = None
        self._interrupt_sub_id: str | None = None
        self._interrupted = False
        self._active_synthesis_task: asyncio.Task | None = None

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("ttt.response", self._handle_response)
        self.create_subscription_task("control.interrupt", self._handle_interrupt)
        await asyncio.sleep(0.01)
        logger.info("TTSComponent '%s' started.", self.name)

    async def stop(self) -> None:
        if self._active_synthesis_task and not self._active_synthesis_task.done():
            self._active_synthesis_task.cancel()
            self._active_synthesis_task = None
        await super().stop()
        logger.info("TTSComponent '%s' stopped.", self.name)

    async def _handle_interrupt(self, message: PipelineMessage) -> None:
        self._interrupted = True
        logger.info("TTSComponent received control.interrupt; cancelling pending synthesis.")
        if self._active_synthesis_task and not self._active_synthesis_task.done():
            self._active_synthesis_task.cancel()

    async def _handle_response(self, message: PipelineMessage) -> None:
        text = str(message.payload) if message.payload else ""
        if not text.strip():
            return

        self._interrupted = False
        style, clean_text = parse_style_tag(text)

        # Resolve speaker_id from message or SpeakerContext
        speaker_id = message.speaker_id
        if not speaker_id and self.speaker_context and self.speaker_context.is_resolved:
            speaker_id = self.speaker_context.current_speaker_id

        logger.info("Synthesizing speech (style=%s, speaker=%s): '%s'...", style, speaker_id, clean_text)
        self._active_synthesis_task = asyncio.create_task(
            self._synthesize_and_publish(clean_text, style, message.trace_id, speaker_id)
        )
        try:
            await self._active_synthesis_task
        except asyncio.CancelledError:
            logger.debug("Active synthesis cancelled due to interruption.")
        finally:
            self._active_synthesis_task = None

    async def _synthesize_and_publish(
        self,
        clean_text: str,
        style: str,
        trace_id: str,
        speaker_id: str | None,
    ) -> None:
        try:
            # Resolve per-speaker TTS parameters if available
            profile = None
            if self.profile_store and speaker_id:
                profile = self.profile_store.get_profile(speaker_id)
            elif self.speaker_context and self.speaker_context.is_resolved:
                profile = self.speaker_context.current_profile

            chunks = await asyncio.to_thread(self.tts_engine.synthesize, clean_text, style)
            metadata: dict[str, Any] = {
                "style": style,
                "text": clean_text,
                "speaker_id": speaker_id,
            }
            if profile:
                metadata["voice_id"] = profile.tts_voice_id
                metadata["pitch"] = profile.tts_pitch
                metadata["speaking_rate"] = profile.tts_speaking_rate

            for chunk in chunks:
                if self._interrupted or not self._is_running:
                    logger.debug("Playback interrupted before chunk could be sent.")
                    break

                await self.bus.publish(
                    "audio.playback",
                    PipelineMessage(
                        channel="audio.playback",
                        phase=StreamPhase.AUDIO,
                        payload=chunk,
                        trace_id=trace_id,
                        speaker_id=speaker_id,
                        metadata=metadata,
                    ),
                )
        except Exception as e:
            self.record_error(e)
            logger.error("TTS synthesis error: %s", e)
