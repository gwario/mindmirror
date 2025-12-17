import numpy as np
from collections import deque
from .config import (
    SPEECH_THRESHOLD_MULTIPLIER,
    SILENCE_THRESHOLD_MULTIPLIER,
    MIN_NOISE_FLOOR,
    INITIAL_NOISE_FLOOR
)

class VADEngine:
    def __init__(self):
        self.noise_window = deque(maxlen=100)
        self.noise_window.append(INITIAL_NOISE_FLOOR)

    def process_chunk(self, audio_chunk, adapt=True):
        """
        Returns: (is_speech, is_silence, rms, noise_floor)
        adapt: If False, threshold is calculated but noise floor is NOT updated.
        """
        # 1. Calculate Energy
        rms = np.std(audio_chunk)
        if rms < 1e-7:
            return False, True, 0.0, self.get_noise_floor()

        # 2. Update Background Noise Model
        if adapt:
            self.noise_window.append(rms)

        noise_floor = self.get_noise_floor()

        # 3. Decision
        # Ensure thresholds never drop below practical limits
        speech_thresh = max(noise_floor * SPEECH_THRESHOLD_MULTIPLIER, MIN_NOISE_FLOOR * 2)
        silence_thresh = max(noise_floor * SILENCE_THRESHOLD_MULTIPLIER, MIN_NOISE_FLOOR)

        is_speech = rms > speech_thresh
        is_silence = rms < silence_thresh

        return is_speech, is_silence, rms, noise_floor

    def get_noise_floor(self):
        # 10th percentile rule + absolute minimum floor
        return max(np.percentile(self.noise_window, 10), MIN_NOISE_FLOOR)