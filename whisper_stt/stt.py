import os
import time
import queue
import tempfile
from collections import deque
import torch

import numpy as np
import sounddevice as sd
import soundfile as sf

import sound


def continuous_stt_task(log_queue, selected_device, text_queue):
    """Listen and transcribe continuously with preroll"""

    # --- 1. LAZY IMPORT ---
    try:
        import whisper
    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import whisper. Is it installed?"})
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. LOAD MODEL ---
    log_queue.put({'type': 'info', 'text': f"Loading Whisper model on {device} (PID: {os.getpid()})..."})
    try:
        model = whisper.load_model("small", device=device)
        log_queue.put({'type': 'info', 'text': "✅ Whisper Model Loaded."})
    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Failed to load Whisper: {e}"})
        return

    audio_queue = queue.Queue()

    # Settings
    sample_rate = sound.get_valid_samplerate(selected_device)
    chunk_duration = 0.1
    volume_percentile = 75
    min_audio_length = 0.8

    # Calibration
    def calibrate_noise_floor(duration):
        """Record ambient noise to set baseline"""
        log_queue.put({'type': 'info', 'text': f"Calibrating... Stay quiet for {duration} seconds..."})
        noise_samples = []

        def callback(indata, frames, time, status):
            noise_samples.append(np.abs(indata).max())

        with sound.safe_open_stream(selected_device, sample_rate, callback=callback):
            sd.sleep(int(duration * 1000))

        noise_floor = np.percentile(noise_samples, 90)
        log_queue.put({'type': 'info', 'text': f"Noise floor detected: {noise_floor:.3f}"})
        return noise_floor

    # Calibrate
    current_noise_floor = calibrate_noise_floor(3)
    speech_threshold = current_noise_floor * 4.0
    silence_threshold = current_noise_floor * 2.5

    # Initialize buffers
    silence_duration = 2.0
    volume_window = deque(maxlen=int(silence_duration / chunk_duration))
    preroll_buffer = sound.create_preroll_buffer(sample_rate, chunk_duration)

    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    def is_silent():
        """Check if most of the window is silent"""
        if len(volume_window) < volume_window.maxlen:
            return False
        return np.percentile(volume_window, volume_percentile) < silence_threshold

    buffer = []
    is_speaking = False
    last_meter_update = time.time()
    meter_update_interval = 0.2

    with sound.safe_open_stream(selected_device, sample_rate, callback=audio_callback,
                                blocksize=int(sample_rate * chunk_duration)):
        log_queue.put({'type': 'info', 'text': "Whisper is listening... Start speaking!"})
        log_queue.put({'type': 'info', 'text': f"Thresholds → Noise: {current_noise_floor:.3f} | Silence: {silence_threshold:.3f} | Speech: {speech_threshold:.3f}"})

        while True:
            audio_chunk = audio_queue.get()
            volume = np.abs(audio_chunk).max()
            volume_window.append(volume)

            # Always fill preroll buffer
            preroll_buffer.append(audio_chunk)

            # Update volume meter periodically
            current_time = time.time()
            if current_time - last_meter_update > meter_update_interval:
                meter = sound.create_volume_meter_rich(
                    volume,
                    current_noise_floor,
                    silence_threshold,
                    speech_threshold
                )
                log_queue.put({'type': 'meter', 'text': meter})
                last_meter_update = current_time

            if volume > speech_threshold:
                if not is_speaking:
                    # Just started speaking - dump preroll
                    buffer.extend(list(preroll_buffer))
                is_speaking = True
                buffer.append(audio_chunk)
            elif is_speaking and volume > silence_threshold:
                buffer.append(audio_chunk)

            # Check if we should transcribe
            if is_speaking and is_silent() and len(buffer) > 0:
                # Check minimum audio length
                audio_duration = len(buffer) * chunk_duration
                if audio_duration < min_audio_length:
                    log_queue.put({'type': 'status', 'text': f"🚫 Too short ({audio_duration:.1f}s)"})
                    buffer = []
                    is_speaking = False
                    volume_window.clear()
                    continue

                log_queue.put({'type': 'status', 'text': "⏳ Transcribing..."})
                audio_data = np.concatenate(buffer)

                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, audio_data, sample_rate)

                        result = model.transcribe(f.name, language='en', fp16=torch.cuda.is_available())
                        transcribed_text = result['text'].strip()

                        if transcribed_text:
                            log_queue.put({'type': 'user', 'text': transcribed_text})
                            text_queue.put(transcribed_text)
                        else:
                            log_queue.put({'type': 'status', 'text': "🚫 Empty transcription"})

                        try:
                            os.remove(f.name)
                        except:
                            pass

                except Exception as e:
                    log_queue.put({'type': 'error', 'text': f"Transcribe Error: {e}"})

                # Reset
                buffer = []
                is_speaking = False
                volume_window.clear()