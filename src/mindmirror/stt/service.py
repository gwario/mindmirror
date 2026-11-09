"""MindMirror Speech-to-Text Microservice Runner."""

import asyncio
import logging
import os
import signal
import sys

from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport import create_message_bus
from mindmirror.stt.component import STTComponent
from mindmirror.stt.vad import VADComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mindmirror.stt.service")


def get_stt_engine():
    """Instantiates FasterWhisperSTT or fallback STT interface."""
    try:
        from mindmirror.stt.local_whisper.faster_whisper import FasterWhisperSTT
        model_size = os.getenv("WHISPER_MODEL", "base.en")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        engine = FasterWhisperSTT(model_size=model_size, device=device)
        engine.load_model()
        return engine
    except Exception as e:
        logger.warning("Could not load FasterWhisperSTT (%s); using mock STT engine.", e)
        from mindmirror.pipeline.testing.harness import MockGS1STTEngine
        return MockGS1STTEngine()


async def main():
    logger.info("Initializing MindMirror STT Service...")
    bus = create_message_bus()
    supervisor = ComponentSupervisor(check_interval=0.5)

    # Register VAD component
    supervisor.register(
        "vad",
        lambda: VADComponent(bus=bus, sample_rate=16000),
    )

    # Register STT component
    stt_engine = get_stt_engine()
    supervisor.register(
        "stt",
        lambda: STTComponent(bus=bus, stt_engine=stt_engine),
    )

    await supervisor.start()
    logger.info("STT Service running and supervised under ComponentSupervisor.")

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
        logger.info("Shutting down STT Service...")
        await supervisor.stop()
        await bus.close()
        logger.info("STT Service stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
