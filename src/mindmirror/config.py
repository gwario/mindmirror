import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
HIGHPASS_FREQ = int(os.getenv("HIGHPASS_FREQ", "80"))
NR_STRENGTH = float(os.getenv("NR_STRENGTH", "0.35"))
TRIM_DB = int(os.getenv("TRIM_DB", "30"))
PAD_DURATION = 0.1
NORMALIZE_TARGET = 0.90
MIN_NOISE_FLOOR = 0.005
INITIAL_NOISE_FLOOR = 0.017

# --- VAD SETTINGS ---
SPEECH_THRESHOLD_MULTIPLIER = 2.2
SILENCE_THRESHOLD_MULTIPLIER = 1.4
MIN_AUDIO_LENGTH = 0.5
SILENCE_DURATION = 1.2
COOLDOWN_DURATION = 0.5
KEYBOARD_BUFFER = 0.5
PRE_ROLL_DURATION = 0.5

# --- SYSTEM TIMINGS ---
LOOP_SLEEP_TIME = 0.1
QUEUE_TIMEOUT = 0.5
POST_PLAYBACK_COOLDOWN = 0.4

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
PIPER_MODEL_PATH = str(PROJECT_ROOT / "src/mindmirror/tts/pipervoice/en/semaine/en_GB-semaine-medium.onnx")

# --- TTS SETTINGS (GOOGLE CLOUD) ---
GOOGLE_TTS_MODEL = os.getenv("GOOGLE_TTS_MODEL")
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE")
GOOGLE_TTS_LANG = os.getenv("GOOGLE_TTS_LANG")
GOOGLE_TTS_STYLES = {
    "neutral": {
        "speaking_rate": 1.15, 
        "pitch": 0.0,
        "prompt": "Read aloud in a warm, welcoming tone."
    },
    "excited": {
        "speaking_rate": 1.1, 
        "pitch": 0.5,
        "prompt": "Read aloud in an excited, high-energy, and enthusiastic tone."
    },
    "serious": {
        "speaking_rate": 1.05, 
        "pitch": -0.5,
        "prompt": "Read aloud in a professional, slightly casual tone."
    },
    "lazy": {
        "speaking_rate": 1.0, 
        "pitch": -1.5,
        "prompt": "Read aloud in a casual and relaxed tone."
    },
}

# --- TTS SETTINGS (ELEVENLABS) ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_STYLES = {
    "neutral": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True},
    "excited": {"stability": 0.3, "similarity_boost": 0.85, "style": 0.3, "use_speaker_boost": True},
    "serious": {"stability": 0.7, "similarity_boost": 0.65, "style": 0.0, "use_speaker_boost": True},
    "lazy": {"stability": 0.4, "similarity_boost": 0.5, "style": 0.1, "use_speaker_boost": True},
}

# --- TTS SETTINGS (F5) ---
F5_VOICE_NAME = "MyVoice"
# Assuming F5-TTS is cloned alongside the main project root
F5_LIB_PATH = PROJECT_ROOT.parent / "F5-TTS"
F5_CKPT_PATH = str(F5_LIB_PATH / f"ckpts/{F5_VOICE_NAME}/model_last.pt")
F5_VOCAB_FILE = str(F5_LIB_PATH / f"data/{F5_VOICE_NAME}_pinyin/vocab.txt")  # Pretrained pinyin vocab copied here by prepare_csv_wavs.py
F5_WAVS_DIR = PROJECT_ROOT / f"data/{F5_VOICE_NAME}/wavs"
F5_NFE_STEPS = 16
F5_MIN_CHUNK_LENGTH = 40
F5_STYLES = {
    "neutral": {"ref_audio": str(F5_WAVS_DIR / "en_A_01.wav"), "ref_text": "Here is the summary of your notifications.", "cfg": 2.0, "speed": 1.0},
    "serious": {"ref_audio": str(F5_WAVS_DIR / "en_B_01.wav"), "ref_text": "I'm sorry, but I cannot complete that request right now.", "cfg": 2.2, "speed": 1.1},
    "excited": {"ref_audio": str(F5_WAVS_DIR / "en_C_01.wav"), "ref_text": "Wow! That worked perfectly on the first try!", "cfg": 1.5, "speed": 0.95},
    "lazy": {"ref_audio": str(F5_WAVS_DIR / "en_D_01.wav"), "ref_text": "Yeah, I think that's... mostly correct, actually.", "cfg": 2.5, "speed": 1.0}
}

# --- PIPELINE GREETING SETTINGS ---
GREET_ON_START = os.getenv("GREET_ON_START", "true").lower() == "true"
GREETING_TRIGGER_TEXT = os.getenv("GREETING_TRIGGER_TEXT", "Greet the user shortly to start the conversation.")

# --- GOOGLE APPLICATION CREDENTIALS ---
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# --- STT SETTINGS (GOOGLE CLOUD) ---
GOOGLE_STT_MODEL = os.getenv("GOOGLE_STT_MODEL")
GOOGLE_STT_LANG = os.getenv("GOOGLE_STT_LANG")

# --- LLM SETTINGS (GEMINI) ---
GOOGLE_TTT_MODEL = os.getenv("GOOGLE_TTT_MODEL")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# --- LANGCHAIN TTT SETTINGS ---
LANGCHAIN_TTT_PROVIDER = os.getenv("LANGCHAIN_TTT_PROVIDER", "google")
LANGCHAIN_TTT_MODEL = os.getenv("LANGCHAIN_TTT_MODEL", GOOGLE_TTT_MODEL or "gemini-2.5-flash-lite")
LANGCHAIN_TEMPERATURE = float(os.getenv("LANGCHAIN_TEMPERATURE", "0.7"))

# --- PERSISTENT MEMORY & RUNTIME AGENT SETTINGS ---
RUNTIME_AGENT_FILE = PROJECT_ROOT / "data" / "AGENTS.md"
SKILLS_DIR = PROJECT_ROOT / "data" / "skills"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
ENABLE_PERSISTENT_MEMORY = os.getenv("ENABLE_PERSISTENT_MEMORY", "true").lower() == "true"

# --- STT SETTINGS (WHISPER) ---
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_SAGEMAKER_WHISPER_ENDPOINT_NAME = os.getenv("AWS_SAGEMAKER_WHISPER_ENDPOINT_NAME")

# --- MCP SETTINGS ---
USE_MOCK_MCP = os.getenv("USE_MOCK_MCP", "true").lower() == "true"

if USE_MOCK_MCP:
    import sys
    MCP_SERVERS = [
        {
            "name": "fraud-detection",
            "type": "stdio",
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "src/mindmirror/mcp_client/mock_server.py")]
        }
    ]
else:
    MCP_SERVERS = [
        {
            "name": "fraud-detection",
            "type": "sse",
            "url": os.getenv("MCP_FRAUD_SSE_URL", "http://localhost:8088/sse")
        }
    ]

