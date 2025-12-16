# Paths
LOCK_FILE = "speaking.lock"

# Audio Settings
CHUNK_DURATION = 0.1     # Seconds per read
MIN_AUDIO_LENGTH = 0.8   # Minimum phrase length to transcribe
SILENCE_DURATION = 2.0   # How long to wait after speech stops
VOLUME_PERCENTILE = 75   # For silence detection robustness

# VAD Multipliers (Adjust sensitivity here)
SPEECH_THRESHOLD_MULTIPLIER = 4.0
SILENCE_THRESHOLD_MULTIPLIER = 2.5