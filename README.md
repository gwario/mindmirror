# MindMirror: Local AI Voice Assistant Pipeline

MindMirror is an end-to-end, highly modular conversational AI voice assistant. It integrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) into a seamless, real-time pipeline. Designed with a clean, object-oriented architecture, it allows for easy swapping of models and configuration, making it a robust platform for local voice applications.

## Origin

MindMirror originated as a personal research experiment exploring the concept of *self-directed conversational AI* — specifically, whether a system could facilitate a real-time dialogue in which a user interacts with a language model that responds in their own synthesised voice. The hypothesis was that such a setup could serve as an audible externalisation of inner monologue, prompting deeper reflection through the medium of conversation. As the architecture evolved and additional components were integrated, the project naturally matured into a broader, general-purpose local voice assistant pipeline.

## Features
- **Real-Time Voice Activity Detection (VAD):** Accurately captures speech while filtering out background noise using custom DSP and VAD logic.
- **Interruption Support:** The system gracefully handles being interrupted while speaking, ducking its own audio and listening for keywords.
- **Modular Pipeline Architecture:** Built using `multiprocessing` queues. The main entry point manages hardware selection, coordinates queue topologies, launches child processes, and handles graceful shutdowns.
- **Multi-Model Support:** Each pipeline stage is backed by swappable implementations behind a common interface:
  - **STT:** `LocalWhisperSTT` (on-device, CUDA-accelerated), `SageMakerWhisperSTT` (AWS-offloaded, CPU-friendly), or `GoogleCloudSTT` (Google Cloud Speech-to-Text V2, streaming).
  - **LLM:** `GeminiLLMClient` (Google Gemini via Vertex AI) with optional **MCP tool-use support** — connects to one or more MCP servers (stdio or SSE) and exposes their tools to the model for agentic interactions.
  - **TTS:** `PiperTTS` (lightweight, fast, CPU-friendly), `F5TTS` (diffusion-based voice cloning, GPU recommended), or `GoogleCloudTTS` (Google Cloud Text-to-Speech with style control and multi-voice support).
- **Centralized Configuration:** A single `config.py` acts as the source of truth for all thresholds, device settings, and model paths.

## Project Structure
```text
mindmirror/
├── scripts/
│   ├── record_sample.py       # Utility for recording dataset samples for voice cloning
│   └── test_inference.py      # Utility for testing fine-tuned voice models
└── src/
    └── mindmirror/
        ├── config.py          # Centralized settings (Thresholds, Paths, Constants)
        ├── main.py            # Entry point containing composition, queue setups, and lifecycles
        ├── audio/             # Decoupled audio subsystem (DSP, IO, Devices)
        ├── stt/               # Speech-to-Text classes and loop runner
        ├── llm/               # Text-to-Thought (LLM) client and loop runner
        ├── tts/               # Text-to-Speech classes and loop runner
        └── ui/                # Terminal-based visual meters and console logging
```

## How to Run

1. Create a Conda environment:
   ```bash
   conda create -n mindmirror python=3.11 -y
   conda activate mindmirror
   pip install -r requirements.txt
   ```
2. Set up your API Keys and Settings:
   - Create a `.env` file in the root directory.
   - Configure the following settings:
      ```env
      # ==============================================================================
      # Google Cloud Platform (Vertex AI & Text-to-Speech)
      # ==============================================================================
      GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
      GOOGLE_CLOUD_PROJECT=886498416680         # Your Google Cloud Project ID
      GOOGLE_CLOUD_LOCATION=us-central1         # Your Google Cloud Location

      # Model Selection
      GOOGLE_TTT_MODEL=gemini-2.5-flash-lite         # Default Gemini model
      GOOGLE_TTS_MODEL=gemini-2.5-flash-preview-tts  # Default TTS model
      GOOGLE_TTS_VOICE=Aoede                     # Default TTS voice
      GOOGLE_TTS_LANG=en-gb                      # Default TTS language (align with GOOGLE_STT_LANG)


      # ==============================================================================
      # Amazon Web Services (SageMaker Speech-to-Text & LLM)
      # ==============================================================================
      # AWS Credentials (if not configured via AWS CLI profile)
      AWS_ACCESS_KEY_ID=your_aws_access_key
      AWS_SECRET_ACCESS_KEY=your_aws_secret_key
      AWS_DEFAULT_REGION=eu-central-1          # AWS Region for SageMaker

      # SageMaker Endpoints
      AWS_SAGEMAKER_WHISPER_ENDPOINT_NAME=whisper-large-v3-endpoint
      AWS_SAGEMAKER_LLM_ENDPOINT_NAME=llama-3-8b-instruct-endpoint  # Required if using SageMaker LLM mode


      # ==============================================================================
      # Hugging Face
      # ==============================================================================
      HF_TOKEN=your_hugging_face_token_here    # Optional (Hugging Face Access Token)
      ```
   - **Google Cloud Authentication**:
     - *Option A*: Create a GCP Service Account key file, download the JSON file, and set the `GOOGLE_APPLICATION_CREDENTIALS` path above.
     - *Option B*: Install Google Cloud CLI (`gcloud`) and run `gcloud auth application-default login` on your system.
   - *Recommendation on STT & TTS selection in [main.py](src/mindmirror/main.py)*:
     - Open [main.py](src/mindmirror/main.py) to edit classes and options directly:
     - **STT Selection**: If you have a local GPU (**CUDA available**), we recommend using `LocalWhisperSTT` to run locally for free with sub-second latency. If running on **CPU only (no CUDA)**, uncomment `SageMakerWhisperSTT` to offload inference to AWS for faster response times.
     - **TTS Selection**: If you have a local GPU (**CUDA available**), you can choose either `PiperTTS` or `F5TTS`. If running on **CPU only (no CUDA)**, we strongly recommend using `PiperTTS` as F5-TTS uses diffusion and will be extremely slow to synthesize speech on a CPU.
3. Configure your TTS:
   - Create the directory `src/mindmirror/tts/pipervoice/en/semaine/` (or update `src/mindmirror/config.py` to point elsewhere).
   - Download the default **Semaine (medium)** British English voice model files from the official Rhasspy repository:
     - [en_GB-semaine-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx?download=true)
     - [en_GB-semaine-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json?download=true)
   - Place both files in that `semaine/` folder.
4. Run the application:
   ```bash
   PYTHONPATH=src python3 src/mindmirror/main.py
   ```

### Checking Available Google Cloud Models:
You can verify which conversational (TTT) and Text-to-Speech (TTS) models are accessible under your Google Cloud project using the checking script:
```bash
PYTHONPATH=src python3 scripts/check_google_models.py
```
This script queries the Vertex AI model catalog, filters for chat/conversational and TTS models, performs live connectivity checks, and prints a list of models that are actively available to use in your configuration.

### For Custom Voice (F5-TTS Fine-tuning):
See the full step-by-step guide in the [TTS module README](src/mindmirror/tts/README.md#custom-voice-training-f5-tts-fine-tuning), which covers dataset recording, preparation, training, and inference testing.

## Development Milestones (Historical Context)

### MS1: Get (english) speech input via mic and output it as text
* Went with whisper because speechbrain gave me some dependency issues with pytorch

### MS2: Send text to ai API and print response
* Went with gemini as it has straight forward api and sufficient free plan
* Integrated MS1 and MS2 with multiprocessing queues

### MS3: TTS with off-the-shelf voice model
* Went with PiperVoice, because easy to use with lots of voice models available
* Integrated MS1, MS2 and MS3

### MS4: Voice cloning
* Went with F5-TTS for high-quality diffusion-based voice cloning — see the [TTS module README](src/mindmirror/tts/README.md#custom-voice-training-f5-tts-fine-tuning) for the full fine-tuning workflow.