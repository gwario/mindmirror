# Text-to-Speech (TTS) Module

The Text-to-Speech module is responsible for synthesizing text response segments from the TTT pipeline into audio waveforms and playing them back on the selected hardware audio output.

## Supported Engines

The system features a dual-engine architecture. Selection is made by editing [main.py](../main.py) directly:

1. **PiperVoice Engine** (`PiperTTS`)

Piper is a fast, local neural text-to-speech system that runs efficiently on CPU architectures.

### Configuration
*   **Model Path**: Configured via `PIPER_MODEL_PATH` in [config.py](../config.py):
    ```python
    PIPER_MODEL_PATH = str(PROJECT_ROOT / "src/mindmirror/tts/pipervoice/en/semaine/en_GB-semaine-medium.onnx")
    ```
*   **Model Download**: The default voice is **Semaine (medium)**. Create the directory `src/mindmirror/tts/pipervoice/en/semaine/` and download the model files from the official Rhasspy repository:
    *   [en_GB-semaine-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx?download=true)
    *   [en_GB-semaine-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json?download=true)
    Both files must be placed in the `semaine/` directory. For other voices, download their corresponding files from the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main) repository and place them inside the `src/mindmirror/tts/pipervoice/en/` subdirectory.

---

## 2. F5-TTS Voice Cloning Engine (`F5TTS`)

F5-TTS is a non-autoregressive flow-matching diffusion model for high-quality voice cloning and synthesis. It requires a local GPU (CUDA) for usable real-time synthesis.

### Configuration
All fine-tuning configurations and path mappings are stored in [config.py](../config.py):
*   `F5_VOICE_NAME` (`"MyVoice"`): The name identifier for your custom cloned voice.
*   `F5_LIB_PATH`: Path pointing to your cloned `F5-TTS` repository (assumed to be alongside the main project folder).
*   `F5_NFE_STEPS` (`16`): Number of Function Evaluations (steps) for the ODE solver. Lower values (e.g. 16) reduce synthesis time at a minor cost to audio quality.
*   `F5_MIN_CHUNK_LENGTH` (`40`): Minimum length threshold to prevent f5-tts synthesis glitches.

### Style Parameterisation (`F5_STYLES`)
F5-TTS utilizes reference audio samples to clone voices. The `F5_STYLES` dictionary maps the assistant's style tags to specific audio files, reference transcripts, classifier-free guidance (CFG) scales, and output playback speed modifiers:

```python
F5_STYLES = {
    "neutral": {
        "ref_audio": str(F5_WAVS_DIR / "en_A_01.wav"), 
        "ref_text": "Here is the summary of your notifications.", 
        "cfg": 2.0, 
        "speed": 1.0
    },
    "serious": {
        "ref_audio": str(F5_WAVS_DIR / "en_B_01.wav"), 
        "ref_text": "I'm sorry, but I cannot complete that request right now.", 
        "cfg": 2.2, 
        "speed": 1.1
    },
    "excited": {
        "ref_audio": str(F5_WAVS_DIR / "en_C_01.wav"), 
        "ref_text": "Wow! That worked perfectly on the first try!", 
        "cfg": 1.5, 
        "speed": 0.95
    },
    "lazy": {
        "ref_audio": str(F5_WAVS_DIR / "en_D_01.wav"), 
        "ref_text": "Yeah, I think that's... mostly correct, actually.", 
        "cfg": 2.5, 
        "speed": 1.0
    }
}
```

---

## Custom Voice Training (F5-TTS Fine-Tuning)

To train F5-TTS on your own voice, follow these steps. All commands assume two sibling repos: `repos/mindmirror/` and `repos/F5-TTS/`.

1.  **Install F5-TTS**: Clone the F5-TTS repository, then install it in editable mode:
    ```bash
    # (mindmirror) repos/mindmirror$
    pip install -e ../F5-TTS
    ```
2.  **Record voice samples**:
    ```bash
    # (mindmirror) repos/mindmirror$
    PYTHONPATH=src python3 scripts/record_sample.py
    ```
3.  **Download the pretrained model**: Download `F5TTS_v1_Base` from Hugging Face.
4.  **Adjust the pretrained vocab path** in `repos/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py`:
    ```python
    PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")
    ```
5.  **Prepare the dataset**:
    ```bash
    # (mindmirror) repos/F5-TTS$
    python src/f5_tts/train/datasets/prepare_csv_wavs.py ../mindmirror/data/MyVoice/ ./data/MyVoice_pinyin
    ```
6.  **Adjust worker count** in `repos/F5-TTS/src/f5_tts/train/finetune_cli.py`:
    ```python
    num_workers = os.cpu_count()
    ```
7.  **Fine-tune the model**:
    ```bash
    # (mindmirror) repos/F5-TTS$
    python src/f5_tts/train/finetune_cli.py \
      --exp_name F5TTS_v1_Base \
      --dataset_name MyVoice \
      --finetune \
      --pretrain ckpts/F5TTS_v1_Base/model_1250000.safetensors \
      --tokenizer pinyin \
      --learning_rate 5e-5 \
      --epochs 50 \
      --batch_size_type sample \
      --batch_size_per_gpu 1 \
      --grad_accumulation_steps 4 \
      --save_per_updates 5000 \
      --keep_last_n_checkpoints 1
    ```
8.  **Test the checkpoint**:
    ```bash
    # (mindmirror) repos/mindmirror$
    python scripts/test_inference.py
    ```
9.  **Test inference directly** (optional sanity check):
    ```bash
    # (mindmirror) repos/F5-TTS$
    python src/f5_tts/infer/infer_cli.py \
      --model F5TTS_v1_Base \
      --ckpt_file ckpts/MyVoice/model_last.pt \
      --ref_audio ../mindmirror/voice_samples/wavs_clean/paragraph_01.wav \
      --ref_text "The birch canoe slid on the smooth planks." \
      --gen_text "This is a test. I am checking if my voice model is overtrained."
    ```
10. **Run the main pipeline** with your new voice:
    ```bash
    # (mindmirror) repos/mindmirror$
    PYTHONPATH=src python3 src/mindmirror/main.py
    ```
