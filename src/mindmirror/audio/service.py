"""MindMirror Audio Host I/O Microservice Runner.

Supports:
- Physical soundcards & Bluetooth headsets (via pattern matching or device ID).
- Headless testing via audio files (AUDIO_INPUT_FILE, AUDIO_OUTPUT_FILE).
"""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

from mindmirror.audio.capture import LiveAudioCapture, LiveAudioSource
from mindmirror.audio.devices import find_audio_device
from mindmirror.audio.file_io import FileAudioSink, FileAudioSource
from mindmirror.audio.playback import PlaybackLoop
from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport import create_message_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mindmirror.audio.service")


def create_audio_source():
    """Instantiates audio source: FileAudioSource or LiveAudioSource (Bluetooth/Hardware)."""
    input_file = os.getenv("AUDIO_INPUT_FILE")
    if input_file and Path(input_file).exists():
        logger.info("Using FileAudioSource from file '%s'", input_file)
        loop_audio = os.getenv("AUDIO_INPUT_LOOP", "false").lower() in ("true", "1")
        return FileAudioSource(input_file, loop=loop_audio)

    device_pattern = os.getenv("AUDIO_INPUT_DEVICE")
    device_id, _ = find_audio_device(device_pattern, device_type="input")
    return LiveAudioSource(device_id=device_id, sample_rate=16000)


def create_audio_sink():
    """Instantiates audio sink: FileAudioSink or default HardwareAudioSink."""
    output_file = os.getenv("AUDIO_OUTPUT_FILE")
    if output_file:
        logger.info("Using FileAudioSink writing to '%s'", output_file)
        return FileAudioSink(output_file, sample_rate=16000)

    # Defaults to hardware soundcard output
    return None


async def main():
    logger.info("Initializing MindMirror Audio Host I/O Service...")
    bus = create_message_bus()
    supervisor = ComponentSupervisor(check_interval=0.5)

    sink = create_audio_sink()
    output_pattern = os.getenv("AUDIO_OUTPUT_DEVICE")
    output_device_id, _ = find_audio_device(output_pattern, device_type="output")

    # Register Playback Loop
    supervisor.register(
        "playback",
        lambda: PlaybackLoop(bus=bus, sink=sink, device_id=output_device_id, sample_rate=16000),
    )

    # Register Audio Capture (starts after playback)
    source = create_audio_source()
    supervisor.register(
        "capture",
        lambda: LiveAudioCapture(bus=bus, source=source, sample_rate=16000),
    )

    await supervisor.start()
    logger.info("Audio Service running and supervised under ComponentSupervisor.")

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
        logger.info("Shutting down Audio Host Service...")
        await supervisor.stop()
        await bus.close()
        logger.info("Audio Host Service stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
