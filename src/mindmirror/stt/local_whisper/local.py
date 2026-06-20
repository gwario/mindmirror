import os
import torch
import tempfile
import soundfile as sf
import numpy as np

from mindmirror.stt.interface import STTInterface

class LocalWhisperSTT(STTInterface):
    """
    Local implementation of the STTInterface using OpenAI's Whisper model.
    Runs inference on local CPU or GPU (CUDA).
    """

    def __init__(self, model_name: str = "small", log_queue = None):
        self.model_name = model_name
        self.log_queue = log_queue
        self.model = None
        self.device = None

    def load_model(self) -> None:
        """
        Loads the Whisper model locally on GPU (CUDA) if available, otherwise on CPU.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(2)  # Limit CPU threads to prevent audio callback starvation
        
        try:
            import whisper
            if self.log_queue:
                self.log_queue.put({'type': 'info', 'text': f"Loading local Whisper on {self.device}..."})

            self.model = whisper.load_model(self.model_name, device=self.device)

            if self.log_queue:
                self.log_queue.put({'type': 'success', 'text': f"✅ Local Whisper Model ({self.model_name}) Loaded."})

        except ImportError:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': "Could not import 'whisper'. Install it first."})
            raise ImportError("whisper is required for local STT mode.")
        except Exception as e:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': f"Local Whisper Load Failed: {e}"})
            raise e

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Transcribes the given numpy audio data array using the local Whisper model.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                filename = f.name
                sf.write(filename, audio_data, sample_rate)

            try:
                use_fp16 = (self.device == "cuda")
                result = self.model.transcribe(filename, language='en', fp16=use_fp16)
                text = result['text'].strip()
            finally:
                try:
                    os.remove(filename)
                except OSError:
                    pass

            return text

        except Exception as e:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': f"Local STT Inference Error: {e}"})
            return None
