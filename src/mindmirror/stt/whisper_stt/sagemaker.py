import os
import tempfile
import soundfile as sf
import json
import numpy as np
from dotenv import load_dotenv

from mindmirror import config
from mindmirror.stt.interface import STTInterface

class SageMakerWhisperSTT(STTInterface):
    """
    Remote implementation of the STTInterface using an AWS SageMaker endpoint.
    """

    def __init__(self, region: str = None, endpoint_name: str = None, log_queue = None):
        self.region = region or getattr(config, 'AWS_DEFAULT_REGION', 'eu-central-1')
        self.endpoint_name = endpoint_name or getattr(config, 'SAGEMAKER_WHISPER_ENDPOINT_NAME', 'whisper-large-v3-endpoint')
        self.log_queue = log_queue
        self.client = None

    def load_model(self) -> None:
        """
        Initializes the AWS SageMaker Runtime client.
        """
        try:
            import boto3
            load_dotenv()  # Ensure .env is loaded

            if self.log_queue:
                self.log_queue.put({
                    'type': 'info', 
                    'text': f"Initializing AWS SageMaker Client in {self.region}..."
                })
            
            session = boto3.Session()
            self.client = session.client('sagemaker-runtime', region_name=self.region)

            if self.log_queue:
                self.log_queue.put({
                    'type': 'success', 
                    'text': f"✅ SageMaker Runtime initialized ({self.region})."
                })

        except ImportError:
            if self.log_queue:
                self.log_queue.put({
                    'type': 'error', 
                    'text': "Could not import 'boto3'. Run 'pip install boto3'."
                })
            raise ImportError("boto3 is required for SageMaker STT mode.")
        except Exception as e:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': f"SageMaker Client Load Failed: {e}"})
            raise e

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Transcribes the given numpy audio data array by invoking the SageMaker endpoint.
        """
        if not self.client:
            raise RuntimeError("SageMaker client is not initialized. Call load_model() first.")

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                filename = f.name
                sf.write(filename, audio_data, sample_rate)

            try:
                with open(filename, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                response = self.client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='audio/x-audio',
                    Body=audio_bytes
                )
                
                response_body = response['Body'].read().decode('utf-8')
                result = json.loads(response_body)

                if isinstance(result, dict):
                    text = result.get('text', result.get('prediction', '')).strip()
                elif isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    text = item.get('text', item.get('prediction', '')) if isinstance(item, dict) else str(item)
                else:
                    text = str(result)
            finally:
                try:
                    os.remove(filename)
                except OSError:
                    pass

            return text

        except Exception as e:
            if self.log_queue:
                self.log_queue.put({'type': 'error', 'text': f"SageMaker STT Inference Error: {e}"})
            return None
