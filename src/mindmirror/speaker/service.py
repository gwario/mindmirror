"""MindMirror Speaker Identification & Biometrics Microservice Runner.

Dedicated microservice for acoustic voice diarization, multi-modal speaker identification,
and speaker profile management. Independent from STT to accommodate future visual
and biometric recognition signals.
"""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport import create_message_bus
from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.identifier import (
    AcousticVoiceEmbedder,
    SpeakerIdentifierComponent,
    SpeakerRegistry,
)
from mindmirror.speaker.profile import SpeakerProfileStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mindmirror.speaker.service")


async def main():
    logger.info("Initializing MindMirror Speaker Identification & Biometrics Service...")
    bus = create_message_bus()
    supervisor = ComponentSupervisor(check_interval=0.5)

    # Persistence directory for ChromaDB (or connects to remote ChromaDB if CHROMA_HOST set)
    db_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")) if not os.getenv("CHROMA_HOST") else None

    registry = SpeakerRegistry(db_dir=db_dir)
    profile_store = SpeakerProfileStore(db_dir=db_dir)
    speaker_context = SpeakerContext(bus=bus)

    threshold = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.6"))

    # Register Speaker Identifier Component
    supervisor.register(
        "speaker_identifier",
        lambda: SpeakerIdentifierComponent(
            bus=bus,
            registry=registry,
            profile_store=profile_store,
            speaker_context=speaker_context,
            similarity_threshold=threshold,
        ),
    )

    await supervisor.start()
    logger.info("Speaker Identification Service running and supervised under ComponentSupervisor.")

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
        logger.info("Shutting down Speaker Identification Service...")
        await supervisor.stop()
        await speaker_context.close()
        await bus.close()
        logger.info("Speaker Identification Service stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
