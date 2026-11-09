from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np

class TTSInterface(ABC):
    """
    Abstract Base Class defining the contract for all Text-to-Speech (TTS) engines.
    """

    @abstractmethod
    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        """
        Runs the Text-to-Speech synthesis and playback loop.
        
        Args:
            log_queue: Multiprocessing Queue for logging and ui status.
            selected_device: Audio hardware output device name or index.
            text_queue: Multiprocessing Queue providing input text messages/style tuples.
            control_queue: Multiprocessing Queue receiving playback control commands.
        """
        pass

    def synthesize(self, text: str, style: str = "NEUTRAL") -> list[bytes]:
        """
        Pure synthesis method converting text and emotional style into raw PCM audio chunks.
        Decoupled from hardware audio output devices for headless simulation and testing.
        """
        return []
