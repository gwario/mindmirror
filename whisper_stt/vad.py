import numpy as np
import sounddevice as sd
from .config import VOLUME_PERCENTILE


def calibrate_noise_floor(device, sample_rate, duration=3.0, log_queue=None):
    """
    Records ambient noise for a few seconds to determine the baseline.
    """
    if log_queue:
        log_queue.put({'type': 'info', 'text': f"Calibrating... Stay quiet for {duration}s..."})

    noise_samples = []

    def callback(indata, frames, time, status):
        noise_samples.append(np.abs(indata).max())

    with sd.InputStream(device=device, channels=1, samplerate=sample_rate, callback=callback):
        sd.sleep(int(duration * 1000))

    # We take the 90th percentile to ignore random loud pops
    noise_floor = np.percentile(noise_samples, 90)

    if log_queue:
        log_queue.put({'type': 'info', 'text': f"Noise floor detected: {noise_floor:.4f}"})

    return noise_floor

def is_window_silent(volume_window, silence_threshold):
    """
    Returns True if the majority of the recent audio history is below the threshold.
    """
    if len(volume_window) < volume_window.maxlen:
        return False
    return np.percentile(volume_window, VOLUME_PERCENTILE) < silence_threshold

def get_volume(audio_chunk):
    return np.abs(audio_chunk).max()