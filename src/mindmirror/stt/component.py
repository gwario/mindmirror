"""Supervised STT pipeline component subscribing to vad.speech_audio and publishing stt.transcript."""

import asyncio
import logging
from typing import Any
import numpy as np

from mindmirror.pipeline.component import ComponentHealth, PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus
from mindmirror.stt.interface import STTInterface

logger = logging.getLogger(__name__)


class STTComponent(PipelineComponent):
    """Supervised component transcribing segmented speech audio into text transcripts."""

    def __init__(
        self,
        bus: MessageBus,
        stt_engine: STTInterface,
        name: str = "stt_engine",
        default_sample_rate: int = 16000,
    ) -> None:
        super().__init__(name=name, bus=bus)
        self.stt_engine = stt_engine
        self.default_sample_rate = default_sample_rate
        self._subscription_id: str | None = None
        self._transcription_lock = asyncio.Lock()

    async def start(self) -> None:
        await super().start()
        try:
            # Model loading in thread pool to avoid blocking event loop
            await asyncio.to_thread(self.stt_engine.load_model)
        except Exception as e:
            logger.warning("STT Engine model pre-loading warning: %s", e)

        self.create_subscription_task("vad.speech_audio", self._handle_speech_audio)
        await asyncio.sleep(0.01)
        logger.info("STTComponent '%s' started, listening to 'vad.speech_audio'.", self.name)

    async def stop(self) -> None:
        await super().stop()
        logger.info("STTComponent '%s' stopped.", self.name)

    async def _handle_speech_audio(self, message: PipelineMessage) -> None:
        async with self._transcription_lock:
            try:
                payload = message.payload
                sample_rate = message.metadata.get("sample_rate", self.default_sample_rate)

                # Transcribe in worker thread to prevent event loop starvation
                text = await asyncio.to_thread(self.stt_engine.transcribe, payload, sample_rate)

                if text and text.strip():
                    clean_text = text.strip()
                    logger.info("STT Transcribed: '%s' (trace=%s)", clean_text, message.trace_id)
                    await self.bus.publish(
                        "stt.transcript",
                        PipelineMessage(
                            channel="stt.transcript",
                            phase=StreamPhase.TRANSCRIPT,
                            payload=clean_text,
                            trace_id=message.trace_id,
                            speaker_id=message.speaker_id,
                            metadata=message.metadata,
                        ),
                    )
            except Exception as e:
                self.record_error(e)
                logger.error("STT transcription exception: %s", e)
