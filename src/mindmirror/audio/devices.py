import sys
import time
import numpy as np
import sounddevice as sd
from mindmirror.ui.meters import create_volume_meter

from mindmirror import config

def select_audio_devices():
    """Select separate input and output devices"""

    print("\n" + "🎤 " + "="*58)
    print("SELECT INPUT DEVICE (Microphone)")
    print("="*60)

    while True:
        input_device, input_sr = select_audio_device(device_type='input')
        if input_device is None: 
            break
        
        test_input_device(input_device, input_sr)
        
        confirm = input("\nUse this input device? (y/n, Enter=yes): ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            break
        print("\nLet's try selecting again...")

    print("\n" + "🔊 " + "="*58)
    print("SELECT OUTPUT DEVICE (Speaker)")
    print("="*60)

    while True:
        output_device, output_sr = select_audio_device(device_type='output')
        if output_device is None: 
            break
        
        test_output_device(output_device, output_sr)
        
        confirm = input("\nUse this output device? (y/n, Enter=yes): ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            break
        print("\nLet's try selecting again...")

    return input_device, input_sr, output_device, output_sr


def test_input_device(device_id, samplerate):
    """Test the selected input device with a visual meter"""
    print("\n" + "-"*60)
    print(f"🎙️  TESTING INPUT DEVICE ({config.INPUT_DEVICE_TEST_DURATION} seconds)")
    print("    Please speak into the microphone...")
    print("-"*60)

    try:
        def callback(indata, frames, time, status):
            vol = np.linalg.norm(indata) * 10
            meter = create_volume_meter(vol, 0.1, width=40, is_recording=True)
            sys.stdout.write(f"\r{meter}")
            sys.stdout.flush()

        with safe_open_stream(device_id, samplerate, callback=callback):
            time.sleep(config.INPUT_DEVICE_TEST_DURATION)
        print("\n✅ Input test complete.")
    except Exception as e:
        print(f"\n❌ Input test failed: {e}")


def test_output_device(device_id, samplerate):
    """Test the selected output device by playing a tone"""
    print("\n" + "-"*60)
    print(f"🔊 TESTING OUTPUT DEVICE ({config.OUTPUT_DEVICE_TEST_DURATION} seconds)")
    print("    Playing a test tone...")
    print("-"*60)

    try:
        t = np.linspace(0, config.OUTPUT_DEVICE_TEST_DURATION, int(samplerate * config.OUTPUT_DEVICE_TEST_DURATION), False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        sd.play(tone, samplerate, device=device_id, blocking=True)
        print("✅ Output test complete.")
    except Exception as e:
        print(f"❌ Output test failed: {e}")


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
            if device_type == 'input':
                default_id = sd.default.device
            else:
                default_id = sd.default.device

            default_device = sd.query_devices(default_id)
            default_sr = int(default_device['default_samplerate'])
            print(f"✅ Using default {device_type}: {default_device['name']} @ {default_sr}Hz")
            return default_id, default_sr

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
            return device_id, device_sr

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

        default_device = sd.query_devices(sd.default.device)
        default_sr = int(default_device['default_samplerate'])

        stream = sd.InputStream(
            device=None,
            samplerate=default_sr,
            channels=channels,
            **kwargs
        )
        return stream

def get_valid_samplerate(device_id):
    try:
        with safe_open_stream(device_id, config.PREFERRED_SR):
            pass
        return config.PREFERRED_SR
    except Exception:
        if device_id is None:
            return int(sd.query_devices(kind='input')['default_samplerate'])
        return int(sd.query_devices(device_id)['default_samplerate'])
