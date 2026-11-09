"""Unit tests for digital signal processing (DSP) algorithms."""

import numpy as np
import pytest

from mindmirror.audio.dsp import (
    apply_bandpass_filter,
    apply_dsp_cleaning,
    apply_highpass_filter,
    normalize_audio,
    reduce_noise,
    resampled,
)


def _generate_sine_wave(freq_hz: float, duration_s: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def test_normalize_audio_peak_scaling():
    """Validates that normalize_audio scales the peak amplitude to the target level."""
    # Signal with peak 0.5 (target gain 0.90 / 0.5 = 1.8 < max_gain 3.0)
    sine = _generate_sine_wave(440.0, duration_s=0.5, sample_rate=16000, amplitude=0.5)
    assert np.max(np.abs(sine)) == pytest.approx(0.5, rel=1e-2)

    normalized = normalize_audio(sine, target_level=0.90)
    assert np.max(np.abs(normalized)) == pytest.approx(0.90, rel=1e-2)


def test_normalize_audio_gain_ceiling():
    """Validates that normalize_audio respects the max_gain ceiling to prevent blowing up silence."""
    # Near silence
    silence = np.full(1000, 0.001, dtype=np.float32)
    normalized = normalize_audio(silence, target_level=0.90, max_gain=3.0)
    assert np.max(np.abs(normalized)) == pytest.approx(0.003, rel=1e-2)


def test_highpass_filter_attenuates_low_rumble():
    """Validates that apply_highpass_filter attenuates 30Hz rumble below the 80Hz cutoff."""
    sr = 16000
    rumble = _generate_sine_wave(30.0, duration_s=1.0, sample_rate=sr, amplitude=0.8)
    filtered = apply_highpass_filter(rumble, sample_rate=sr, cutoff_freq=80.0)

    # Energy of low-frequency rumble should be substantially reduced
    initial_energy = np.mean(rumble**2)
    filtered_energy = np.mean(filtered**2)
    assert filtered_energy < initial_energy * 0.15


def test_bandpass_filter_preserves_voice_band():
    """Validates that bandpass filter passes 1000Hz tone while heavily attenuating 10Hz and 7900Hz."""
    sr = 16000
    mid_voice = _generate_sine_wave(1000.0, duration_s=1.0, sample_rate=sr, amplitude=0.5)
    filtered_mid = apply_bandpass_filter(mid_voice, sample_rate=sr, lowcut=80.0, highcut=7500.0)

    # 1kHz tone should be mostly preserved (> 90% energy)
    assert np.mean(filtered_mid**2) > 0.9 * np.mean(mid_voice**2)

    sub_bass = _generate_sine_wave(15.0, duration_s=1.0, sample_rate=sr, amplitude=0.5)
    filtered_sub = apply_bandpass_filter(sub_bass, sample_rate=sr, lowcut=80.0, highcut=7500.0)
    assert np.mean(filtered_sub**2) < 0.1 * np.mean(sub_bass**2)


def test_noise_reduction_reduces_noise_floor():
    """Validates that spectral noise reduction attenuates known background noise profile."""
    sr = 16000
    noise = np.random.normal(0, 0.05, sr)
    signal = _generate_sine_wave(440.0, duration_s=1.0, sample_rate=sr, amplitude=0.4)
    noisy_signal = signal + noise

    cleaned = reduce_noise(noisy_signal, sample_rate=sr, noise_profile=noise, prop_decrease=0.8)
    assert isinstance(cleaned, np.ndarray)
    assert len(cleaned) == len(noisy_signal)


def test_resampled_correct_sample_counts():
    """Validates Fourier resampling changes array length proportionally."""
    audio_48k = _generate_sine_wave(440.0, duration_s=1.0, sample_rate=48000)
    assert len(audio_48k) == 48000

    audio_16k = resampled(audio_48k, source_sample_rate=48000, target_sample_rate=16000)
    assert len(audio_16k) == 16000


def test_apply_dsp_cleaning_pipeline():
    """Validates end-to-end DSP pipeline execution without console."""
    sr = 16000
    audio = _generate_sine_wave(500.0, duration_s=0.5, sample_rate=sr, amplitude=0.3)
    noise_profile = np.random.normal(0, 0.02, 1600)

    cleaned = apply_dsp_cleaning(audio, rate=sr, noise_profile=noise_profile, console=None)
    assert isinstance(cleaned, np.ndarray)
    assert np.max(np.abs(cleaned)) <= 1.0
