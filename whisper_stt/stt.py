import time
import queue
import os
from collections import deque
import numpy as np
import sys
from pathlib import Path
from .config import (
    LOCK_FILE, CHUNK_DURATION, MIN_AUDIO_LENGTH, SILENCE_DURATION,
    SPEECH_THRESHOLD_MULTIPLIER, SILENCE_THRESHOLD_MULTIPLIER
)
from .vad import calibrate_noise_floor, is_window_silent, get_volume
from .transcriber import load_whisper_model, transcribe_audio
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sound

def stt_task(log_queue, selected_device, text_queue):

    # 1. SETUP MODEL
    model, device = load_whisper_model(log_queue)
    if not model:
        return # Exit if load failed

    sample_rate = sound.get_valid_samplerate(selected_device)

    # 2. CALIBRATION
    noise_floor = calibrate_noise_floor(selected_device, sample_rate, log_queue=log_queue)
    speech_thresh = noise_floor * SPEECH_THRESHOLD_MULTIPLIER
    silence_thresh = noise_floor * SILENCE_THRESHOLD_MULTIPLIER

    # 3. STATE
    audio_queue = queue.Queue()
    preroll_buffer = sound.create_preroll_buffer(sample_rate, CHUNK_DURATION)
    volume_window = deque(maxlen=int(SILENCE_DURATION / CHUNK_DURATION))

    buffer = []
    is_speaking = False
    last_meter_time = 0

    # 4. CALLBACK
    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    # 5. MAIN LOOP
    with sound.safe_open_stream(selected_device, sample_rate, callback=audio_callback,
                                blocksize=int(sample_rate * CHUNK_DURATION)):

        log_queue.put({'type': 'info', 'text': "👂 Listening..."})

        while True:
            # --- A. LOCK CHECK ---
            if os.path.exists(LOCK_FILE):
                with audio_queue.mutex: audio_queue.queue.clear()
                buffer = []; is_speaking = False; volume_window.clear()
                time.sleep(0.1)
                continue

            # --- B. PROCESS AUDIO ---
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            volume = get_volume(chunk)
            volume_window.append(volume)
            preroll_buffer.append(chunk)

            # Update UI Meter
            if time.time() - last_meter_time > 0.2:
                meter = sound.create_volume_meter_rich(volume, noise_floor, silence_thresh, speech_thresh)
                log_queue.put({'type': 'meter', 'text': meter})
                last_meter_time = time.time()

            # --- C. VAD ---
            if volume > speech_thresh:
                if not is_speaking:
                    buffer.extend(list(preroll_buffer))
                is_speaking = True
                buffer.append(chunk)

            elif is_speaking and volume > silence_thresh:
                buffer.append(chunk)

            # --- D. END OF SPEECH ---
            if is_speaking and is_window_silent(volume_window, silence_thresh):
                if len(buffer) > 0:
                    duration = len(buffer) * CHUNK_DURATION

                    if duration < MIN_AUDIO_LENGTH:
                        log_queue.put({'type': 'status', 'text': f"🚫 Noise ({duration:.1f}s)"})
                    else:
                        # --- E. TRANSCRIBE (Functional Call) ---
                        log_queue.put({'type': 'status', 'text': "⏳ Transcribing..."})
                        full_audio = np.concatenate(buffer)

                        # Pass model and device explicitly
                        text = transcribe_audio(model, full_audio, sample_rate, device, log_queue)

                        if text:
                            log_queue.put({'type': 'user', 'text': text})
                            text_queue.put(text)
                        else:
                            log_queue.put({'type': 'status', 'text': "🚫 Empty"})

                    # Reset
                    buffer = []; is_speaking = False; volume_window.clear()