import os
from pathlib import Path

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCK_FILE = str(PROJECT_ROOT / "speaking.lock")
PLAYBACK_LOCK = str(PROJECT_ROOT / "playback.lock")

# --- AUDIO SETTINGS ---
PREFERRED_SR = 48000
TARGET_SR = 16000
CHUNK_DURATION = 0.1
INPUT_DEVICE_TEST_DURATION = 3.0
OUTPUT_DEVICE_TEST_DURATION = 1.5

# --- DSP SETTINGS ---
HIGHPASS_FREQ = 80
NR_STRENGTH = 0.35
TRIM_DB = 30
PAD_DURATION = 0.1
NORMALIZE_TARGET = 0.90
MIN_NOISE_FLOOR = 0.005
INITIAL_NOISE_FLOOR = 0.017

# --- VAD SETTINGS ---
SPEECH_THRESHOLD_MULTIPLIER = 4.0
SILENCE_THRESHOLD_MULTIPLIER = 2.5
MIN_AUDIO_LENGTH = 0.8
SILENCE_DURATION = 2.0
COOLDOWN_DURATION = 0.8
KEYBOARD_BUFFER = 0.5
PRE_ROLL_DURATION = 0.5

# --- SYSTEM TIMINGS ---
LOOP_SLEEP_TIME = 0.1
QUEUE_TIMEOUT = 0.5

# --- INTERRUPTION DETECTION ---
INTERRUPT_ENERGY_MULTIPLIER = 2.5
INTERRUPT_BASELINE_WINDOW = 10
INTERRUPT_RECORDING_DURATION = 1.5
DUCK_VOLUME = 0.3
INTERRUPT_KEYWORDS = [
    "stop", "pause", "wait", "hold on", "quiet", "silence",
    "shut up", "never mind", "cancel", "enough"
]

# --- TTS SETTINGS (PIPER) ---
PIPER_MODEL_PATH = str(PROJECT_ROOT / "src/mindmirror/models/tts/pipervoice/en/semaine/en_GB-semaine-medium.onnx")

# --- TTS SETTINGS (F5) ---
F5_VOICE_NAME = "MyVoice"
# Assuming F5-TTS is cloned alongside the main project root
F5_LIB_PATH = PROJECT_ROOT.parent / "F5-TTS"
F5_CKPT_PATH = str(F5_LIB_PATH / f"ckpts/{F5_VOICE_NAME}/model_last.pt")
F5_VOCAB_FILE = str(F5_LIB_PATH / f"data/{F5_VOICE_NAME}_pinyin/vocab.txt")
F5_WAVS_DIR = PROJECT_ROOT / f"data/{F5_VOICE_NAME}/wavs"
F5_NFE_STEPS = 16
F5_MIN_CHUNK_LENGTH = 40
F5_STYLES = {
    "neutral": {"ref_audio": str(F5_WAVS_DIR / "en_A_01.wav"), "ref_text": "Here is the summary of your notifications.", "cfg": 2.0, "speed": 1.0},
    "serious": {"ref_audio": str(F5_WAVS_DIR / "en_B_01.wav"), "ref_text": "I'm sorry, but I cannot complete that request right now.", "cfg": 2.2, "speed": 1.1},
    "excited": {"ref_audio": str(F5_WAVS_DIR / "en_C_01.wav"), "ref_text": "Wow! That worked perfectly on the first try!", "cfg": 1.5, "speed": 0.95},
    "lazy": {"ref_audio": str(F5_WAVS_DIR / "en_D_01.wav"), "ref_text": "Yeah, I think that's... mostly correct, actually.", "cfg": 2.5, "speed": 1.0}
}

# --- LLM SETTINGS (GEMINI) ---
GEMINI_MODEL = 'gemini-flash-lite-latest'
