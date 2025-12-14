import os
import queue
import tempfile
from collections import deque
import torch

import numpy as np
import sounddevice as sd
import soundfile as sf

import sound


def continuous_stt_task(log_queue, selected_device, text_queue):
    """Listen and transcribe, put results in queue"""

    # --- 1. LAZY IMPORT ---
    # We import whisper here so it initializes CUDA only in this child process
    try:
        import whisper
    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import whisper. Is it installed?"})
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. LOAD MODEL ---
    log_queue.put({'type': 'info', 'text': f"Loading Whisper model on {device} (PID: {os.getpid()})..."})
    try:
        # You can force device="cpu" here if you want to save GPU for TTS,
        # but "cuda" is fine if you have VRAM (approx 2GB for Small).
        model = whisper.load_model("small", device=device)
        log_queue.put({'type': 'info', 'text': "✅ Whisper Model Loaded."})
    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Failed to load Whisper: {e}"})
        return

    audio_queue = queue.Queue()

    # Settings
    sample_rate = sound.get_valid_samplerate(selected_device)
    silence_duration = 2.0
    chunk_duration = 0.1
    volume_percentile = 75
    min_audio_length = 0.8  # Minimum audio length before transcription

    # Calibration
    def calibrate_noise_floor(duration):
        """Record ambient noise to set baseline"""
        log_queue.put({'type': 'info', 'text': f"Calibrating... Stay quiet for {duration} seconds..."})
        noise_samples = []

        def callback(indata, frames, time, status):
            noise_samples.append(np.abs(indata).max())

        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, device=selected_device):
            sd.sleep(int(duration * 1000))

        noise_floor = np.percentile(noise_samples, 90)
        log_queue.put({'type': 'info', 'text': f"Noise floor detected: {noise_floor:.3f}"})
        return noise_floor

    # Calibrate
    current_noise_floor = calibrate_noise_floor(3)
    speech_threshold = current_noise_floor * 4.0
    silence_threshold = current_noise_floor * 2.5

    # Initialize volume window
    volume_window = deque(maxlen=int(silence_duration / chunk_duration))

    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    def is_silent():
        """Check if most of the window is silent"""
        if len(volume_window) < volume_window.maxlen:
            return False
        return np.percentile(volume_window, volume_percentile) < silence_threshold

    buffer = []
    is_speaking = False

    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=sample_rate,
                        device=selected_device,
                        blocksize=int(sample_rate * chunk_duration)):
        log_queue.put({'type': 'info', 'text': "Whisper is listening... Start speaking!"})

        while True:
            audio_chunk = audio_queue.get()
            volume = np.abs(audio_chunk).max()
            volume_window.append(volume)

            if volume > speech_threshold:
                # Using \r for status updates prevents log spam
                # Note: Sending too many updates to a multiprocessing queue can lag it.
                # Consider reducing frequency if UI lags.
                # log_queue.put({'type': 'status', 'text': f"\r🎤 Speaking: {volume:.3f}"})
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

                        # Actual Transcription
                        # fp16=False prevents warnings on some CPUs, change to True if using GPU only
                        result = model.transcribe(f.name, language='en', fp16=torch.cuda.is_available())
                        transcribed_text = result['text'].strip()

                        if transcribed_text:
                            log_queue.put({'type': 'user', 'text': transcribed_text})
                            text_queue.put(transcribed_text)
                        else:
                            log_queue.put({'type': 'status', 'text': "🚫 Empty transcription"})

                        # Cleanup temp file
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