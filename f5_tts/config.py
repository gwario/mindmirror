from pathlib import Path

# --- PATH LOGIC ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
F5_LIB_PATH = PROJECT_ROOT.parent / "F5-TTS"

VOICE_NAME = "MyVoice"

# Checkpoints are inside the F5-TTS library folder
CKPT_PATH = F5_LIB_PATH / f"ckpts/{VOICE_NAME}/model_last.pt"
VOCAB_FILE = F5_LIB_PATH / f"data/{VOICE_NAME}_pinyin/vocab.txt"

# Audio data is inside YOUR Project Root
WAVS_DIR = PROJECT_ROOT / f"data/{VOICE_NAME}/wavs"
LOCK_FILE = PROJECT_ROOT / "speaking.lock"

# --- SETTINGS ---
NFE_STEPS = 16
# A chunk must be at least this long to be spoken alone.
# If shorter, it gets glued to the next sentence.
MIN_CHUNK_LENGTH = 40  # Characters (approx 6-8 words)

STYLES = {
    "neutral": {
        "ref_audio": str(WAVS_DIR / "en_A_01.wav"),
        "ref_text": "Here is the summary of your notifications.",
        "cfg": 2.0, "speed": 1.0
    },
    "serious": {
        "ref_audio": str(WAVS_DIR / "en_B_01.wav"),
        "ref_text": "I'm sorry, but I cannot complete that request right now.",
        "cfg": 2.2, "speed": 1.1
    },
    "excited": {
        "ref_audio": str(WAVS_DIR / "en_C_01.wav"),
        "ref_text": "Wow! That worked perfectly on the first try!",
        "cfg": 1.5, "speed": 0.95
    },
    "lazy": {
        "ref_audio": str(WAVS_DIR / "en_D_01.wav"),
        "ref_text": "Yeah, I think that's... mostly correct, actually.",
        "cfg": 2.5, "speed": 1.0
    }
}