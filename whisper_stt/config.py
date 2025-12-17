import os

# --- PATHS ---
# Must match the path used in f5_tts/config.py
LOCK_FILE = "speaking.lock"

# --- AUDIO SETTINGS ---
CHUNK_DURATION = 0.1      # Seconds per audio read (Latency vs Stability)
SAMPLE_RATE_DEFAULT = 16000 # Fallback if device doesn't specify

# --- VAD (Voice Activity Detection) ---
# How much louder than background noise must speech be?
SPEECH_THRESHOLD_MULTIPLIER = 4.0
SILENCE_THRESHOLD_MULTIPLIER = 2.5

# Minimum absolute volume (RMS) to consider as noise floor.
# Prevents VAD from becoming "hyper-sensitive" in dead silent rooms.
MIN_NOISE_FLOOR = 0.005

# Initial guess for noise floor before calibration
INITIAL_NOISE_FLOOR = 0.017

# --- SENTENCE LOGIC ---
MIN_AUDIO_LENGTH = 0.8    # Ignore blips shorter than this (0.8s)
SILENCE_DURATION = 2.0    # How long to wait after speech stops to cut the sentence

# --- ECHO PREVENTION (COOLDOWN) ---
# How long to stay "deaf" after the bot stops speaking.
# Helps ignore room reverb/echo.
COOLDOWN_DURATION = 0.8   # Seconds

# --- PROCESS TIMINGS ---
LOOP_SLEEP_TIME = 0.1     # How long to sleep when lock is active
QUEUE_TIMEOUT = 0.5       # Non-blocking queue read timeout