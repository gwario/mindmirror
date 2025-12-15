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

def select_audio_devices():
    """Select separate input and output devices"""

    print("\n" + "🎤 " + "="*58)
    print("SELECT INPUT DEVICE (Microphone)")
    print("="*60)

    input_device, input_sr = select_audio_device(device_type='input')

    print("\n" + "🔊 " + "="*58)
    print("SELECT OUTPUT DEVICE (Speaker)")
    print("="*60)

    output_device, output_sr = select_audio_device(device_type='output')

    return input_device, input_sr, output_device, output_sr


def select_audio_device(device_type='input'):
    """Select audio device with type filter (input or output)"""

    while True:
        print("\n" + "="*60)
        print(f"AVAILABLE AUDIO {device_type.upper()} DEVICES:")
        print("="*60)

        devices = sd.query_devices()
        valid_devices = []

        for i, device in enumerate(devices):
            if device_type == 'input':
                channel_count = device['max_input_channels']
            else:
                channel_count = device['max_output_channels']

            if channel_count > 0:
                valid_devices.append((i, device))

                if device_type == 'input':
                    default_marker = " [DEFAULT]" if i == sd.default.device else ""
                else:
                    default_marker = " [DEFAULT]" if i == sd.default.device else ""

                print(f"  {i}: {device['name']}{default_marker}")
                print(f"      Channels: {channel_count}, "
                      f"Sample Rate: {device['default_samplerate']:.0f} Hz")

        print("="*60)

        if not valid_devices:
            print(f"⚠️  No {device_type} devices found!")
            retry = input("Press Enter to refresh, or 'q' to quit: ").strip().lower()
            if retry == 'q':
                return None, None
            continue

        device_input = input(f"\nEnter {device_type} device ID (Enter for default, 'r' to refresh): ").strip().lower()

        if device_input == 'r':
            print("🔄 Refreshing device list...")
            continue

        if device_input == "":
            # Get the actual default device ID and its sample rate
            if device_type == 'input':
                default_id = sd.default.device
            else:
                default_id = sd.default.device

            default_device = sd.query_devices(default_id)
            default_sr = int(default_device['default_samplerate'])
            print(f"✅ Using default {device_type}: {default_device['name']} @ {default_sr}Hz")
            return default_id, default_sr  # Return the actual ID, not None

        try:
            device_id = int(device_input)

            if device_id < 0 or device_id >= len(devices):
                print(f"⚠️  Device {device_id} doesn't exist. Please try again.")
                input("Press Enter to continue...")
                continue

            selected_device = sd.query_devices(device_id)

            if device_type == 'input':
                channels = selected_device['max_input_channels']
            else:
                channels = selected_device['max_output_channels']

            if channels == 0:
                print(f"⚠️  Device {device_id} has no {device_type} channels.")
                input("Press Enter to continue...")
                continue

            device_sr = int(selected_device['default_samplerate'])
            print(f"✅ Selected {device_type}: {selected_device['name']} @ {device_sr} Hz")

            confirm = input("Confirm? (y/n, Enter=yes): ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return device_id, device_sr
            else:
                continue

        except ValueError:
            print(f"⚠️  Invalid input. Please enter a number.")
            input("Press Enter to continue...")
            continue
        except Exception as e:
            print(f"⚠️  Error: {e}")
            input("Press Enter to continue...")
            continue

def get_device_by_name(pattern):
    """Find device by name pattern (fallback if index changes)"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if (pattern.lower() in device['name'].lower() and
                device['max_input_channels'] > 0):
            return i, int(device['default_samplerate'])
    return None, None

def safe_open_stream(device_id, sample_rate, channels=1, **kwargs):
    """Open audio stream with automatic fallback if device disappeared"""
    try:
        # Try with specified device
        stream = sd.InputStream(
            device=device_id,
            samplerate=sample_rate,
            channels=channels,
            **kwargs
        )
        return stream

    except sd.PortAudioError as e:
        print(f"⚠️  Device {device_id} unavailable: {e}")
        print("🔄 Falling back to default device...")

        # Fallback to default
        default_device = sd.query_devices(sd.default.device)
        default_sr = int(default_device['default_samplerate'])

        stream = sd.InputStream(
            device=None,  # Use default
            samplerate=default_sr,
            channels=channels,
            **kwargs
        )
        return stream

def get_valid_samplerate(device_id):
    try:
        with safe_open_stream(device_id, PREFERRED_SR):
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

# --- SHARED CONSTANTS ---
KEYBOARD_BUFFER = 0.5  # adjust as needed
SILENCE_DURATION = 2.0
PRE_ROLL_DURATION = 0.5
TARGET_SR = 16000  # or whatever you need

# --- SHARED HELPER FUNCTIONS ---

def create_preroll_buffer(sample_rate, chunk_duration):
    """Create a preroll buffer with appropriate size"""
    chunks_in_preroll = int(PRE_ROLL_DURATION / chunk_duration)
    return deque(maxlen=chunks_in_preroll)

def create_volume_meter(volume, threshold, width=50, is_recording=False):
    """Create a visual volume meter for terminal display"""
    bar_len = int(min(volume, 0.5) * width)
    bar = "█" * bar_len
    spaces = " " * (width - bar_len)

    status = "REC    " if is_recording else "WAITING"
    color = "\033[91m" if is_recording else "\033[93m"

    return f"   [{color}{status}\033[0m] Level: |{color}{bar}{spaces}\033[0m| {volume:.3f}"

def create_volume_meter_rich(volume, noise_floor, silence_thresh, speech_thresh, width=40):
    """Create a visual volume meter for Rich console - SINGLE LINE ONLY"""
    normalized = min(volume, 1.0)
    filled = int(normalized * width)

    bar = "█" * filled + "░" * (width - filled)

    # Color and status
    if volume > speech_thresh:
        color = "green"
        status = "🎤 SPEAKING"
    elif volume > silence_thresh:
        color = "yellow"
        status = "🔊 SOUND"
    else:
        color = "dim"
        status = "🔇 SILENT"

    return f"[{color}]{bar}[/{color}] {volume:.3f} | {status}"

# --- YOUR REFACTORED FUNCTIONS ---

def record_clip(device_id, native_sr, threshold, noise_profile, console):
    """Record a single clip with preroll"""

    # Keyboard Buffer
    sys.stdout.write("\r   \033[90m...getting ready (shhh)...\033[0m")
    sys.stdout.flush()
    time.sleep(KEYBOARD_BUFFER)
    sys.stdout.write("\r" + " " * 50 + "\r")

    state = {"recording": [], "is_started": False, "silence_chunks": 0, "current_vol": 0.0}
    chunk_size = 1024
    chunk_duration = chunk_size / native_sr
    silence_limit = int((SILENCE_DURATION * native_sr) / chunk_size)

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
    resampled_audio_data = resampled(audio_data, native_sr, TARGET_SR)

    return resampled_audio_data


def resampled(generated_audio, source_sample_rate: int, target_sample_rate: int):
    if source_sample_rate != target_sample_rate:
        num_samples = int(len(generated_audio) * target_sample_rate / source_sample_rate)
        resampled_audio = scipy.signal.resample(generated_audio, num_samples)
    else:
        resampled_audio = generated_audio
    return resampled_audio