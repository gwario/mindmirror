"""MindMirror Text-to-Speech Microservice Runner."""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport import create_message_bus
from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.profile import SpeakerProfileStore
from mindmirror.tts.component import TTSComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mindmirror.tts.service")


def get_tts_engine():
    """Instantiates PiperTTS or fallback TTS interface."""
    try:
        from mindmirror.tts.pipervoice.tts import PiperTTS
        model_name = os.getenv("PIPER_MODEL", "en_GB-alan-medium")
        engine = PiperTTS(voice=model_name)
        if not os.path.exists(engine.model_path):
            raise FileNotFoundError(f"Piper model file not found at {engine.model_path}")
        return engine
    except Exception as e:
        logger.warning("Could not load PiperTTS (%s); using fallback mock TTS engine.", e)
        from mindmirror.pipeline.testing.harness import MockGS1TTSEngine
        return MockGS1TTSEngine(chunk_count=3)


async def main():
    logger.info("Initializing MindMirror TTS Service...")
    bus = create_message_bus()
    supervisor = ComponentSupervisor(check_interval=0.5)

    db_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")) if not os.getenv("CHROMA_HOST") else None
    profile_store = SpeakerProfileStore(db_dir=db_dir)
    speaker_context = SpeakerContext(bus=bus)
    tts_engine = get_tts_engine()

    # Register TTS Component
    supervisor.register(
        "tts",
        lambda: TTSComponent(
            bus=bus,
            tts_engine=tts_engine,
            profile_store=profile_store,
            speaker_context=speaker_context,
        ),
    )

    await supervisor.start()
    logger.info("TTS Service running and supervised under ComponentSupervisor.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down TTS Service...")
        await supervisor.stop()
        await speaker_context.close()
        await bus.close()
        logger.info("TTS Service stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
