from abc import ABC, abstractmethod
import numpy as np

class STTInterface(ABC):
    """
    Abstract Base Class defining the contract for all Speech-to-Text (STT) engines.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the underlying model weights locally or establishes the connection client.
        
        Raises:
            Exception: If loading or client initialization fails.
        """
        pass

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Transcribes a raw audio buffer to text.
        
        Args:
            audio_data (np.ndarray): Single-channel floating point numpy array.
            sample_rate (int): Sample rate of the provided audio data (e.g. 16000).
            
        Returns:
            str: Transcribed text segment. None if transcription fails or is empty.
        """
        pass

    def is_streaming(self) -> bool:
        """
        Returns True if the engine supports real-time chunk-by-chunk streaming transcription.
        """
        return False

    def start_stream(self, sample_rate: int) -> None:
        """
        Initializes a real-time streaming recognition session.
        """
        pass

    def send_chunk(self, chunk: np.ndarray) -> None:
        """
        Sends an audio chunk to the active streaming recognition session.
        """
        pass

    def end_stream(self) -> str:
        """
        Finalises the active streaming recognition session and returns the transcribed text.
        """
        return ""
