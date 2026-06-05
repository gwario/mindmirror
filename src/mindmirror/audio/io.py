import sys
import time
import numpy as np
import sounddevice as sd
from collections import deque

from mindmirror.audio.dsp import apply_dsp_cleaning, resampled
from mindmirror.audio.devices import safe_open_stream
from mindmirror.ui.meters import create_volume_meter

from mindmirror import config

def calibrate_noise_floor(device_id, native_sr, console):
    console.print("\n[bold yellow]🎙️  CALIBRATING (Stay Silent)...[/bold yellow]")
    console.print("[dim]   Learning room noise profile...[/dim]")

    noise_buffer = []

    def callback(indata, frames, time, status):
        noise_buffer.append(indata.copy())

    try:
        with safe_open_stream(device_id, native_sr, callback=callback):
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

def create_preroll_buffer(sample_rate, chunk_duration):
    """Create a preroll buffer with appropriate size"""
    chunks_in_preroll = int(config.PRE_ROLL_DURATION / chunk_duration)
    return deque(maxlen=chunks_in_preroll)

def record_clip(device_id, native_sr, threshold, noise_profile, console):
    """Record a single clip with preroll"""

    # Keyboard Buffer
    sys.stdout.write("\r   \033[90m...getting ready (shhh)...\033[0m")
    sys.stdout.flush()
    time.sleep(config.KEYBOARD_BUFFER)
    sys.stdout.write("\r" + " " * 50 + "\r")

    state = {"recording": [], "is_started": False, "silence_chunks": 0, "current_vol": 0.0}
    chunk_size = 1024
    chunk_duration = chunk_size / native_sr
    silence_limit = int((config.SILENCE_DURATION * native_sr) / chunk_size)

    # Shared preroll setup
    pre_roll_buffer = create_preroll_buffer(native_sr, chunk_duration)

    def audio_callback(indata, frames, time, status):
        audio_chunk = indata[:, 0]
        vol = np.sqrt(np.mean(audio_chunk**2)) * 10
        state["current_vol"] = vol

        if not state["is_started"]:
            if vol > threshold:
                state["is_started"] = True
                # Dump preroll into recording
                state["recording"].extend(pre_roll_buffer)
                state["recording"].append(audio_chunk.copy())
            else:
                # Keep filling preroll buffer
                pre_roll_buffer.append(audio_chunk.copy())
            return

        # Recording mode
        state["recording"].append(audio_chunk.copy())

        if vol < threshold:
            state["silence_chunks"] += 1
        else:
            state["silence_chunks"] = 0

        if state["silence_chunks"] > silence_limit:
            raise sd.CallbackStop

    try:
        with safe_open_stream(device_id, native_sr, blocksize=chunk_size, callback=audio_callback):
            while True:
                if state["silence_chunks"] > silence_limit:
                    break

                vol = state["current_vol"]
                meter = create_volume_meter(vol, threshold, is_recording=state["is_started"])
                sys.stdout.write(f"\r{meter}")
                sys.stdout.flush()
                time.sleep(0.05)

    except Exception:
        return None

    sys.stdout.write("\n")
    if not state["recording"]:
        return None

    # Process
    audio_data = np.concatenate(state["recording"], axis=0)
    audio_data = apply_dsp_cleaning(audio_data, native_sr, noise_profile, console)

    # Resample
    resampled_audio_data = resampled(audio_data, native_sr, config.TARGET_SR)

    return resampled_audio_data
