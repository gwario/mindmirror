# Speech-to-Text (STT) Module

The Speech-to-Text module is responsible for capturing audio input, processing it through Voice Activity Detection (VAD) to isolate spoken speech, detecting interruptions during assistant output, and transcribing user speech into text.

## Interface Definition

Any Speech-to-Text implementation must adhere to the contract defined in [interface.py](interface.py):

*   **`load_model(self) -> None`**: Initialises the model weights locally or setups remote boto3 clients.
*   **`transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str`**: Decodes a single-channel raw float audio buffer and returns the transcribed text.

The codebase implements this interface across two distinct classes:
1.  **`LocalWhisperSTT`** in [local.py](local_whisper/local.py): Uses the local `whisper` library. Handles CUDA-based GPU routing or multi-threaded CPU limits dynamically.
2.  **`SageMakerWhisperSTT`** in [sagemaker.py](aws_whisper/sagemaker.py): Establishes a remote `boto3` SageMaker Runtime client and forwards binary audio requests to the specified model endpoint.

---

## Configuration and Parameterisation

All STT settings are configured via class selections in [main.py](../main.py), environment variables in the `.env` file, and central constants in [config.py](../config.py).

### 1. Engine Selection

The STT class is configured inside [main.py](../main.py) directly:

*   **Local Whisper STT** (`LocalWhisperSTT`):
    *   Runs the model directly on your machine.
    *   **CUDA (GPU)**: Highly recommended. If CUDA is detected, Whisper runs on GPU using half-precision (`fp16=True`) for optimal sub-second latency.
    *   **CPU**: Whisper runs on CPU with limited threads (configured to 2 threads in code to prevent soundcard driver starving/audio stutter). Response times can be slow on low-powered machines.
    *   Uses the Whisper **small** model configuration by default.
*   **AWS SageMaker Remote Whisper STT** (`SageMakerWhisperSTT`):
    *   Offloads inference tasks asynchronously to an external AWS SageMaker Endpoint hosting Whisper.
    *   Recommended if you are running the assistant on a CPU-only hardware environment to achieve low response latency.
*   **Google Cloud Remote STT v2** (`GoogleCloudSTT`):
    *   Uses Google Cloud's Speech-to-Text V2 API.
    *   Streams audio chunks dynamically in real-time to regional Google Speech endpoints for lowest response latency.
    *   Requires Google Cloud service account key authentication.

### 2. Environment Variables (`.env`)

Define these settings in the root `.env` file depending on the selected mode:

```env
# AWS Configuration (required if using SageMaker STT)
AWS_DEFAULT_REGION=eu-central-1
AWS_SAGEMAKER_WHISPER_ENDPOINT_NAME=whisper-large-v3-endpoint

# AWS Credentials (if not already configured via AWS CLI profile)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Google Cloud STT Configuration (required if using Google STT)
GOOGLE_STT_MODEL=latest_long
GOOGLE_STT_LANG=en-gb
```

### 3. VAD and DSP Settings
Centralised constants in [config.py](../config.py) adjust the sound capture thresholds:
*   `SPEECH_THRESHOLD_MULTIPLIER` (`4.0`): Dynamic threshold multiplier determining when user input speech starts.
*   `SILENCE_DURATION` (`2.0`): Duration of silence in seconds to mark the end of user input and trigger transcription.
*   `MIN_AUDIO_LENGTH` (`0.8`): Minimum duration of speech required in seconds before triggering transcription (filters out short mouth noises or clicks).
