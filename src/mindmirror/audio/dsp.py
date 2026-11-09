"""Digital Signal Processing (DSP) utilities for noise reduction, filtering, and normalization."""

import logging
from typing import Any
import librosa
import noisereduce as nr
import numpy as np
import scipy.signal

from mindmirror import config

logger = logging.getLogger(__name__)


def reduce_noise(
    audio_data: np.ndarray,
    sample_rate: int,
    noise_profile: np.ndarray | None = None,
    prop_decrease: float | None = None,
) -> np.ndarray:
    """Applies spectral gating noise reduction using a captured noise profile or stationary noise assumption."""
    if noise_profile is None or len(noise_profile) == 0:
        return audio_data

    strength = prop_decrease if prop_decrease is not None else config.NR_STRENGTH
    n_fft = min(512, len(audio_data))
    if noise_profile is not None and len(noise_profile) > 0:
        n_fft = min(n_fft, len(noise_profile))
    # Ensure n_fft is power of 2 or at least 64
    n_fft = max(64, 2 ** int(np.log2(n_fft))) if n_fft >= 64 else n_fft

    try:
        cleaned = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            y_noise=noise_profile,
            prop_decrease=strength,
            n_fft=n_fft,
            stationary=True,
        )
        return cleaned
    except Exception as e:
        logger.warning("Noise reduction failed: %s", e)
        return audio_data


def apply_highpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """Applies a Butterworth high-pass filter to remove low-frequency rumble."""
    freq = cutoff_freq if cutoff_freq is not None else config.HIGHPASS_FREQ
    nyquist = 0.5 * sample_rate
    normalized_cutoff = min(freq / nyquist, 0.99)
    sos = scipy.signal.butter(order, normalized_cutoff, btype="hp", output="sos")
    return scipy.signal.sosfilt(sos, audio_data)


def apply_bandpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    lowcut: float = 80.0,
    highcut: float = 7500.0,
    order: int = 4,
) -> np.ndarray:
    """Applies a Butterworth band-pass filter isolating typical human vocal frequencies."""
    nyquist = 0.5 * sample_rate
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.99)
    if low >= high:
        return audio_data
    sos = scipy.signal.butter(order, [low, high], btype="bandpass", output="sos")
    return scipy.signal.sosfilt(sos, audio_data)


def normalize_audio(
    audio_data: np.ndarray,
    target_level: float = 0.90,
    max_gain: float = 3.0,
) -> np.ndarray:
    """Scales audio waveform peak to the target normalization level with a maximum gain ceiling."""
    max_val = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
    if max_val > 0.0:
        target_gain = min(target_level / max_val, max_gain)
        return audio_data * target_gain
    return audio_data


def resampled(generated_audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resamples audio from source_sample_rate to target_sample_rate using Fourier method."""
    if source_sample_rate != target_sample_rate and len(generated_audio) > 0:
        num_samples = int(len(generated_audio) * target_sample_rate / source_sample_rate)
        return scipy.signal.resample(generated_audio, num_samples)
    return generated_audio


def apply_dsp_cleaning(
    audio_data: np.ndarray,
    rate: int,
    noise_profile: np.ndarray | None = None,
    console: Any = None,
) -> np.ndarray:
    """Complete cleaning pipeline: noise reduction -> high-pass filtering -> silence trimming -> padding -> normalization."""
    if console:
        console.print("[dim]   🧼 Cleaning audio...[/dim]", end="")

    # 1. Noise Reduction
    audio_data = reduce_noise(audio_data, rate, noise_profile)

    # 2. High-Pass Filter
    audio_data = apply_highpass_filter(audio_data, rate)

    # 3. Trim Silence
    try:
        audio_data, _ = librosa.effects.trim(audio_data, top_db=config.TRIM_DB)
    except Exception:
        pass

    # 4. Add Padding
    pad_samples = int(config.PAD_DURATION * rate)
    audio_data = np.pad(audio_data, (pad_samples, pad_samples), mode="constant")

    # 5. Normalize
    audio_data = normalize_audio(audio_data, target_level=config.NORMALIZE_TARGET)

    if console:
        console.print("[green] Done.[/green]")
    return audio_data
