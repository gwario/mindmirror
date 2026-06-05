# MindMirror: Local AI Voice Assistant Pipeline

MindMirror is an end-to-end, highly modular conversational AI voice assistant. It integrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) into a seamless, real-time pipeline. Designed with a clean, object-oriented architecture, it allows for easy swapping of models and configuration, making it a robust platform for local voice applications.

## Features
- **Real-Time Voice Activity Detection (VAD):** Accurately captures speech while filtering out background noise using custom DSP and VAD logic.
- **Interruption Support:** The system gracefully handles being interrupted while speaking, ducking its own audio and listening for keywords.
- **Modular Pipeline Architecture:** Built using `multiprocessing` queues. The application orchestrator handles hardware selection, process lifecycle, and graceful shutdowns.
- **Multi-Model Support:** 
  - **STT:** Powered by OpenAI Whisper.
  - **LLM:** Powered by Google Gemini.
  - **TTS:** Supports both lightweight, fast inference (PiperVoice) and advanced, fine-tuned voice cloning models (F5-TTS).
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
        ├── main.py            # Simple entry point
        ├── audio/             # Decoupled audio subsystem (DSP, IO, Devices)
        ├── models/            # Wrappers for STT, LLM, and TTS models
        ├── pipeline/          # Process Orchestrator and Queue management
        └── ui/                # Terminal-based visual meters and console logging
```

## How to Run

1. Create a Conda environment:
   ```bash
   conda create -n mindmirror python=3.11 -y
   conda activate mindmirror
   pip install -r requirements.txt
   ```
2. Set up your API Keys:
   - Add your HuggingFace and Google Gemini API keys to a `.env` file in the root directory.
3. Configure your TTS:
   - Download a PiperVoice model of your choice into `pipervoice/en/` (or update `src/mindmirror/config.py`).
4. Run the application:
   ```bash
   PYTHONPATH=src python3 src/mindmirror/main.py
   ```

### For Custom Voice (F5-TTS Fine-tuning):
1. Clone and install F5-TTS repository: clone and then `(mindmirror) repos/mindmirror$ ` `pip install -e .`
2. Record voice sample: `(mindmirror) repos/mindmirror$ ` `PYTHONPATH=src python3 scripts/record_sample.py`
3. Download the pretrained model "F5TTS_v1_Base" from huggingface
4. Adjust to the pretrained vocab `PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")` in `repos/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py`
5. Prepare dataset: `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/datasets/prepare_csv_wavs.py ../mindmirror/data/MyVoice/ ./data/MyVoice_pinyin`
6. Adjust `num_workers=os.cpu_count()` in `repos/F5-TTS/src/f5_tts/train/finetune_cli.py`
7. Train on top of model F5TTS v1 base
   * `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name MyVoice --finetune --pretrain ckpts/F5TTS_v1_Base/model_1250000.safetensors --tokenizer pinyin --learning_rate 5e-5 --epochs 50 --batch_size_type sample --batch_size_per_gpu 1 --grad_accumulation_steps 4     --save_per_updates 5000 --keep_last_n_checkpoints 1`
8. Test loading the model `(mindmirror) repos/mindmirror$ ` `python scripts/test_inference.py` (or verify_model.py if it still exists)
9. Test inference
   * `(mindmirror) repos/F5-TTS$` `python src/f5_tts/infer/infer_cli.py --model F5TTS_v1_Base --ckpt_file ckpts/MyVoice/model_last.pt --ref_audio ../mindmirror/voice_samples/wavs_clean/paragraph_01.wav --ref_text "The birch canoe slid on the smooth planks." --gen_text "This is a test. I am checking if my voice model is overtrained."`
10. Run the main pipeline: `(mindmirror) repos/mindmirror$ ` `PYTHONPATH=src python3 src/mindmirror/main.py`

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
* Trying with F5-TTS
  1. Install F5-TTS repository: clone and then `pip install -e .` 
  2. Record voice sample: `PYTHONPATH=src python3 scripts/record_sample.py`
  3. Prepare dataset: `python src/f5_tts/train/datasets/prepare_csv_wavs.py ../mindmirror/data/MyVoice/ ./data/MyVoice_pinyin`
  4. Train on top of model F5TTS v1 base
  5. Test inference
  6. Run `PYTHONPATH=src python3 src/mindmirror/main.py`