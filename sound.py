import sys
import time
import numpy as np
import scipy.signal
import sounddevice as sd
import noisereduce as nr
import librosa
from collections import deque

# --- AUDIO CONSTANTS ---
PREFERRED_SR = 48000
TARGET_SR = 24000
SILENCE_DURATION = 1.5
KEYBOARD_BUFFER = 1.0
PRE_ROLL_DURATION = 0.5

# --- DSP SETTINGS ---
HIGHPASS_FREQ = 80       # Kill rumble < 80Hz
NR_STRENGTH = 0.35       # Noise Reduction amount
TRIM_DB = 30             # Trim threshold
PAD_DURATION = 0.1       # Padding start/end
NORMALIZE_TARGET = 0.90  # Normalize Peak

def select_audio_device():
    print("Available audio devices:")
    print(sd.query_devices())
    print("\n" + "="*50 + "\n")

    device_input = input("Enter device ID (or press Enter for default): ").strip()
    if device_input == "":
        return None, None
    try:
        did = int(device_input)
        return did, PREFERRED_SR
    except:
        return None, None

def get_valid_samplerate(device_id):
    try:
        with sd.InputStream(device=device_id, samplerate=PREFERRED_SR, channels=1):
            pass
        return PREFERRED_SR
    except Exception:
        if device_id is None:
            return int(sd.query_devices(kind='input')['default_samplerate'])
        return int(sd.query_devices(device_id)['default_samplerate'])

def calibrate_noise_floor(device_id, native_sr, console):
    console.print("\n[bold yellow]🎙️  CALIBRATING (Stay Silent)...[/bold yellow]")
    console.print("[dim]   Learning room noise profile...[/dim]")

    noise_buffer = []

    def callback(indata, frames, time, status):
        noise_buffer.append(indata.copy())

    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=native_sr, callback=callback):
            time.sleep(2.0)
    except Exception as e:
        console.print(f"[red]CRITICAL ERROR: Could not open mic at {native_sr}Hz[/red]")
        sys.exit(1)

    full_noise_profile = np.concatenate(noise_buffer, axis=0).flatten()

    vol_readings = [np.linalg.norm(chunk) * 10 for chunk in noise_buffer]
    avg_vol = np.mean(vol_readings)
    thresh = max(avg_vol * 4.0, 0.02)

    console.print(f"   ✅ Noise Floor: {avg_vol:.4f}")
    console.print(f"   ✅ Threshold:   [green]{thresh:.4f}[/green]")

    return thresh, full_noise_profile

def apply_dsp_cleaning(audio_data, rate, noise_profile, console):
    console.print("[dim]   🧼 Cleaning audio...[/dim]", end="")

    # 1. Noise Reduction
    if noise_profile is not None:
        try:
            audio_data = nr.reduce_noise(y=audio_data, sr=rate, y_noise=noise_profile,
                                         prop_decrease=NR_STRENGTH, n_fft=2048, stationary=True)
        except: pass

    # 2. High-Pass Filter
    sos = scipy.signal.butter(4, HIGHPASS_FREQ, 'hp', fs=rate, output='sos')
    audio_data = scipy.signal.sosfilt(sos, audio_data)

    # 3. Trim Silence
    try:
        audio_data, _ = librosa.effects.trim(audio_data, top_db=TRIM_DB)
    except: pass

    # 4. Add Padding
    pad_samples = int(PAD_DURATION * rate)
    audio_data = np.pad(audio_data, (pad_samples, pad_samples), mode='constant')

    # 5. Normalize
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        target_gain = NORMALIZE_TARGET / max_val
        if target_gain > 3.0: target_gain = 3.0
        audio_data = audio_data * target_gain

    console.print("[green] Done.[/green]")
    return audio_data

def record_clip(device_id, native_sr, threshold, noise_profile, console):

    # Keyboard Buffer
    sys.stdout.write("\r   \033[90m...getting ready (shhh)...\033[0m")
    sys.stdout.flush()
    time.sleep(KEYBOARD_BUFFER)
    sys.stdout.write("\r" + " " * 50 + "\r")

    state = {"recording": [], "is_started": False, "silence_chunks": 0, "current_vol": 0.0}
    chunk_size = 1024
    silence_limit = int((SILENCE_DURATION * native_sr) / chunk_size)

    # --- PRE-ROLL SETUP ---
    # Calculate how many chunks fit in 0.5 seconds
    chunks_per_sec = native_sr / chunk_size
    pre_roll_len = int(PRE_ROLL_DURATION * chunks_per_sec)
    pre_roll_buffer = deque(maxlen=pre_roll_len)

    def audio_callback(indata, frames, time, status):
        audio_chunk = indata[:, 0]
        vol = np.sqrt(np.mean(audio_chunk**2)) * 10
        state["current_vol"] = vol

        # LOGIC:
        # If NOT started: Keep adding to pre-roll (oldest chunks auto-delete).
        # If TRIGGERED: Dump pre-roll into main recording, then switch flag.

        if not state["is_started"]:
            if vol > threshold:
                state["is_started"] = True
                # Dump the past ~0.5s into the recording first
                state["recording"].extend(pre_roll_buffer)
                # Add the current chunk that triggered it
                state["recording"].append(audio_chunk.copy())
            else:
                # Still waiting, just update pre-roll
                pre_roll_buffer.append(audio_chunk.copy())
            return

        # If STARTED: Just record normally
        state["recording"].append(audio_chunk.copy())

        if vol < threshold:
            state["silence_chunks"] += 1
        else:
            state["silence_chunks"] = 0

        if state["silence_chunks"] > silence_limit:
            raise sd.CallbackStop

    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=native_sr,
                            blocksize=chunk_size, callback=audio_callback):
            while True:
                if state["silence_chunks"] > silence_limit: break

                vol = state["current_vol"]
                bar_len = int(min(vol, 0.5) * 50)
                bar = "█" * bar_len
                spaces = " " * (50 - bar_len)

                status = "REC    " if state["is_started"] else "WAITING"
                color = "\033[91m" if state["is_started"] else "\033[93m"

                sys.stdout.write(f"\r   [{color}{status}\033[0m] Level: |{color}{bar}{spaces}\033[0m| {vol:.3f}")
                sys.stdout.flush()
                time.sleep(0.05)

    except Exception: return None

    sys.stdout.write("\n")
    if not state["recording"]: return None

    # Process
    audio_data = np.concatenate(state["recording"], axis=0)
    audio_data = apply_dsp_cleaning(audio_data, native_sr, noise_profile, console)

    # Resample
    resampled_audio_data = resampled(audio_data, native_sr, TARGET_SR)

    return resampled_audio_data

def resampled(generated_audio, source_sample_rate: int, target_sample_rate: int):
    if source_sample_rate != target_sample_rate:
        num_samples = int(len(generated_audio) * target_sample_rate / source_sample_rate)
        resampled_audio = scipy.signal.resample(generated_audio, num_samples)
    else:
        resampled_audio = generated_audio
    return resampled_audio