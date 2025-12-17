import time
import queue
import os
import numpy as np
from .config import (
    LOCK_FILE, CHUNK_DURATION, MIN_AUDIO_LENGTH, SILENCE_DURATION,
    COOLDOWN_DURATION, LOOP_SLEEP_TIME, QUEUE_TIMEOUT
)
from .vad import VADEngine
from .transcriber import load_whisper_model, transcribe_audio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sound

def stt_task(log_queue, selected_device, text_queue):

    # 1. SETUP
    model, device = load_whisper_model(log_queue)
    if not model: return
    sample_rate = sound.get_valid_samplerate(selected_device)
    vad = VADEngine()

    audio_queue = queue.Queue()
    preroll_buffer = sound.create_preroll_buffer(sample_rate, CHUNK_DURATION)
    buffer = []

    is_speaking = False
    silence_counter = 0

    # DYNAMIC CALCULATIONS
    # How many chunks = X seconds?
    required_silence_chunks = int(SILENCE_DURATION / CHUNK_DURATION)
    cooldown_limit_chunks = int(COOLDOWN_DURATION / CHUNK_DURATION)

    cooldown_counter = 0
    last_meter_time = 0

    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    with sound.safe_open_stream(selected_device, sample_rate, callback=audio_callback,
                                blocksize=int(sample_rate * CHUNK_DURATION)):

        log_queue.put({'type': 'info', 'text': "👂 Listening..."})

        while True:
            # --- A. SHIELD & COOLDOWN CHECK ---
            if os.path.exists(LOCK_FILE):
                with audio_queue.mutex: audio_queue.queue.clear()
                buffer = []; is_speaking = False; silence_counter = 0

                # Reset cooldown so it starts FRESH when lock is removed
                cooldown_counter = cooldown_limit_chunks

                time.sleep(LOOP_SLEEP_TIME)
                continue

            # Lock is gone, but we wait for reverb to die
            if cooldown_counter > 0:
                try:
                    audio_queue.get(timeout=0.05) # Fast drain
                except queue.Empty:
                    pass
                cooldown_counter -= 1
                continue
            # ----------------------------------

            # --- B. PROCESS AUDIO ---
            try:
                chunk = audio_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            preroll_buffer.append(chunk)

            # --- C. VAD ---
            # Don't adapt noise floor if we think someone is speaking
            is_speech_frame, is_silence_frame, vol, noise_floor = vad.process_chunk(chunk, adapt=not is_speaking)

            # --- D. VISUALIZE ---
            if time.time() - last_meter_time > 0.2:
                s_thresh = noise_floor * 4.0
                meter = sound.create_volume_meter_rich(vol, noise_floor, s_thresh * 0.8, s_thresh)
                log_queue.put({'type': 'meter', 'text': meter})
                last_meter_time = time.time()

            # --- E. STATE MACHINE ---
            if is_speech_frame:
                silence_counter = 0
                if not is_speaking:
                    buffer.extend(list(preroll_buffer))
                is_speaking = True
                buffer.append(chunk)

            elif is_speaking:
                buffer.append(chunk)
                if is_silence_frame:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter > required_silence_chunks:
                    # TRANSCRIBE
                    if len(buffer) * CHUNK_DURATION > MIN_AUDIO_LENGTH:
                        log_queue.put({'type': 'status', 'text': "⏳ Transcribing..."})
                        full_audio = np.concatenate(buffer)
                        text = transcribe_audio(model, full_audio, sample_rate, device, log_queue)
                        if text:
                            log_queue.put({'type': 'user', 'text': text})
                            text_queue.put(text)
                    else:
                        log_queue.put({'type': 'status', 'text': "🚫 Too short"})

                    buffer = []; is_speaking = False; silence_counter = 0