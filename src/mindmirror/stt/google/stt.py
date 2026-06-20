import os
import queue
import threading
import numpy as np
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from mindmirror import config
from mindmirror.stt.interface import STTInterface

class GoogleCloudSTT(STTInterface):
    """
    Google Cloud Speech-to-Text V2 implementation of the STTInterface.
    Uses regional client endpoints and Recognizer resources.
    Supports low-latency real-time chunk-by-chunk streaming transcription.
    """
    def __init__(self, language_code: str = None, model: str = None, location: str = None, project_id: str = None, log_queue = None):
        self.language_code = language_code or getattr(config, 'GOOGLE_STT_LANG', 'en-GB')
        self.model = model or getattr(config, 'GOOGLE_STT_MODEL', 'latest_long')
        self.location = location or getattr(config, 'GOOGLE_CLOUD_LOCATION', 'us-central1')
        self.project_id = project_id or getattr(config, 'GOOGLE_CLOUD_PROJECT', None)
        self.log_queue = log_queue
        
        self.client = None
        self.recognizer_path = None
        
        # Streaming session state
        self.stream_queue = None
        self.stream_thread = None
        self.stream_result = None

    def load_model(self) -> None:
        """Initializes regional SpeechClient and manages the Recognizer resource."""
        # 1. Establish regional endpoint client
        endpoint = f"{self.location}-speech.googleapis.com"
        self.client = SpeechClient(client_options={"api_endpoint": endpoint})
        
        # Try to resolve project_id from credentials file if not set
        if not self.project_id:
            import json
            key_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
            if key_path and os.path.exists(key_path):
                try:
                    with open(key_path, "r") as f:
                        self.project_id = json.load(f).get("project_id")
                except Exception:
                    pass
                    
        if not self.project_id:
            raise ValueError("GCP Project ID could not be determined. Set GOOGLE_CLOUD_PROJECT.")

        # Define resource path for the recognizer
        recognizer_id = "mindmirror-stt"
        self.recognizer_path = f"projects/{self.project_id}/locations/{self.location}/recognizers/{recognizer_id}"
        
        # 2. Check for existing recognizer, otherwise create it
        try:
            self.client.get_recognizer(name=self.recognizer_path)
            if self.log_queue:
                self.log_queue.put({'type': 'success', 'text': f"✅ Google STT Recognizer found: {recognizer_id}"})
        except Exception:
            if self.log_queue:
                self.log_queue.put({'type': 'info', 'text': f"Creating Google STT Recognizer '{recognizer_id}'..."})
            try:
                recognizer_config = cloud_speech.Recognizer(
                    model=self.model,
                    language_codes=[self.language_code],
                )
                request = cloud_speech.CreateRecognizerRequest(
                    parent=f"projects/{self.project_id}/locations/{self.location}",
                    recognizer_id=recognizer_id,
                    recognizer=recognizer_config,
                )
                operation = self.client.create_recognizer(request=request)
                operation.result() # Wait for completion
                if self.log_queue:
                    self.log_queue.put({'type': 'success', 'text': "✅ Google STT Recognizer created successfully."})
            except Exception as e:
                if self.log_queue:
                    self.log_queue.put({'type': 'error', 'text': f"Failed to setup Google STT Recognizer: {e}"})
                raise e

    def is_streaming(self) -> bool:
        return True

    def start_stream(self, sample_rate: int) -> None:
        """Starts a background worker thread to process dynamic audio streams."""
        self.stream_queue = queue.Queue()
        self.stream_result = []
        
        def worker():
            try:
                def request_generator():
                    # Initial config request containing recognizer & streaming configurations
                    config_params = cloud_speech.RecognitionConfig(
                        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=sample_rate,
                            audio_channel_count=1,
                        ),
                        language_codes=[self.language_code],
                        model=self.model,
                    )
                    streaming_config = cloud_speech.StreamingRecognitionConfig(
                        config=config_params,
                    )
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self.recognizer_path,
                        streaming_config=streaming_config,
                    )
                    
                    # Subsequent requests containing raw audio chunks
                    while True:
                        pcm_chunk = self.stream_queue.get()
                        if pcm_chunk is None:
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=pcm_chunk)
                
                # Consume streaming responses from API
                responses = self.client.streaming_recognize(requests=request_generator())
                for response in responses:
                    for result in response.results:
                        if result.alternatives:
                            self.stream_result.append(result.alternatives[0].transcript)
            except Exception as e:
                if self.log_queue:
                    self.log_queue.put({'type': 'error', 'text': f"Google STT Stream Error: {e}"})

        self.stream_thread = threading.Thread(target=worker, daemon=True)
        self.stream_thread.start()

    def send_chunk(self, chunk: np.ndarray) -> None:
        """Pushes an incoming chunk to the streaming queue."""
        if self.stream_queue:
            pcm_chunk = (chunk * 32768.0).astype(np.int16).tobytes()
            self.stream_queue.put(pcm_chunk)

    def end_stream(self) -> str:
        """Sends sentinel, joins worker thread, and returns aggregated text results."""
        if self.stream_queue:
            self.stream_queue.put(None)
        if self.stream_thread:
            self.stream_thread.join()
        
        transcript = " ".join(self.stream_result).strip()
        
        # Reset session state
        self.stream_queue = None
        self.stream_thread = None
        self.stream_result = None
        
        return transcript if transcript else None

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Stateless batch transcription using SpeechClient.recognize."""
        if not self.client:
            raise RuntimeError("Model/Client not loaded. Call load_model() first.")
        try:
            pcm_bytes = (audio_data * 32768.0).astype(np.int16).tobytes()
            
            config_params = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=[self.language_code],
                model=self.model,
            )
            
            request = cloud_speech.RecognizeRequest(
                recognizer=self.recognizer_path,
                config=config_params,
                content=pcm_bytes,
            )
            
            response = self.client.recognize(request=request)
            
            transcripts = []
            for result in response.results:
                if result.alternatives:
                    transcripts.append(result.alternatives[0].transcript)
            
            transcript = " ".join(transcripts).strip()
            return transcript if transcript else None
            
        except Exception as e:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': f"Google STT Batch Transcribe Error: {e}"})
            return None
