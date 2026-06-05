import numpy as np
import scipy.signal
import noisereduce as nr
import librosa

from mindmirror import config
def apply_dsp_cleaning(audio_data, rate, noise_profile, console):
    console.print("[dim]   🧼 Cleaning audio...[/dim]", end="")

    # 1. Noise Reduction
    if noise_profile is not None:
        try:
            audio_data = nr.reduce_noise(y=audio_data, sr=rate, y_noise=noise_profile,
                                         prop_decrease=config.NR_STRENGTH, n_fft=2048, stationary=True)
        except: pass

    # 2. High-Pass Filter
    sos = scipy.signal.butter(4, config.HIGHPASS_FREQ, 'hp', fs=rate, output='sos')
    audio_data = scipy.signal.sosfilt(sos, audio_data)

    # 3. Trim Silence
    try:
        audio_data, _ = librosa.effects.trim(audio_data, top_db=config.TRIM_DB)
    except: pass

    # 4. Add Padding
    pad_samples = int(config.PAD_DURATION * rate)
    audio_data = np.pad(audio_data, (pad_samples, pad_samples), mode='constant')

    # 5. Normalize
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        target_gain = config.NORMALIZE_TARGET / max_val
        if target_gain > 3.0: target_gain = 3.0
        audio_data = audio_data * target_gain

    console.print("[green] Done.[/green]")
    return audio_data

def resampled(generated_audio, source_sample_rate: int, target_sample_rate: int):
    if source_sample_rate != target_sample_rate:
        num_samples = int(len(generated_audio) * target_sample_rate / source_sample_rate)
        resampled_audio = scipy.signal.resample(generated_audio, num_samples)
    else:
        resampled_audio = generated_audio
    return resampled_audio
