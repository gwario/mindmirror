import time
import queue
import os
import numpy as np
from mindmirror.config import (
    LOCK_FILE, PLAYBACK_LOCK, CHUNK_DURATION, MIN_AUDIO_LENGTH, SILENCE_DURATION,
    COOLDOWN_DURATION, LOOP_SLEEP_TIME, QUEUE_TIMEOUT,
    INTERRUPT_ENERGY_MULTIPLIER, INTERRUPT_BASELINE_WINDOW, 
    INTERRUPT_RECORDING_DURATION, DUCK_VOLUME, INTERRUPT_KEYWORDS
)
from .vad import VADEngine
from .transcriber import load_whisper_model, transcribe_audio
from mindmirror import audio
from mindmirror.ui import meters

def stt_task(log_queue, selected_device, text_queue, control_queue):

    # 1. SETUP
    model, device = load_whisper_model(log_queue)
    if not model: return
    sample_rate = audio.get_valid_samplerate(selected_device)
    vad = VADEngine()

    audio_queue = queue.Queue()
    preroll_buffer = audio.create_preroll_buffer(sample_rate, CHUNK_DURATION)
    buffer = []

    is_speaking = False
    silence_counter = 0

    # DYNAMIC CALCULATIONS
    # How many chunks = X seconds?
    required_silence_chunks = int(SILENCE_DURATION / CHUNK_DURATION)
    cooldown_limit_chunks = int(COOLDOWN_DURATION / CHUNK_DURATION)

    cooldown_counter = 0
    last_meter_time = 0

    # INTERRUPTION DETECTION STATE
    from collections import deque
    playback_baseline_window = deque(maxlen=INTERRUPT_BASELINE_WINDOW)
    playback_baseline = 0.01
    is_ducked = False
    interruption_buffer = []
    interrupt_recording_chunks = int(INTERRUPT_RECORDING_DURATION / CHUNK_DURATION)

    def audio_callback(indata, frames, time, status):
        if status:
            log_queue.put({'type': 'error', 'text': f"Audio Input Error: {status}"})
        audio_queue.put(indata.copy())

    with audio.safe_open_stream(selected_device, sample_rate, callback=audio_callback,
                                blocksize=int(sample_rate * CHUNK_DURATION)):

        log_queue.put({'type': 'info', 'text': "👂 Listening..."})

        while True:
            # --- Continuous Recording (No Lock File) ---

            # --- B. PROCESS AUDIO ---
            try:
                chunk = audio_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            preroll_buffer.append(chunk)

            # --- INTERRUPTION DETECTION ---
            if os.path.exists(PLAYBACK_LOCK):
                # Calculate energy for this chunk
                energy = np.sqrt(np.mean(chunk**2)) * 10
                
                # Update baseline window
                playback_baseline_window.append(energy)
                
                # Calculate baseline (median of recent chunks for robustness)
                if len(playback_baseline_window) >= 3:
                    playback_baseline = np.median(list(playback_baseline_window))
                    playback_baseline = max(playback_baseline, 0.01)  # Floor
                
                # Check for energy spike
                if not is_ducked and energy > playback_baseline * INTERRUPT_ENERGY_MULTIPLIER:
                    # Possible interruption detected!
                    log_queue.put({'type': 'status', 'text': f"📉 Possible interruption (Energy: {energy:.3f} > {playback_baseline * INTERRUPT_ENERGY_MULTIPLIER:.3f})"})
                    control_queue.put({'command': 'volume', 'value': DUCK_VOLUME})
                    is_ducked = True
                    interruption_buffer = [chunk]  # Start recording
                
                # If ducked, keep recording
                elif is_ducked:
                    interruption_buffer.append(chunk)
                    
                    # Check if we have enough audio to transcribe
                    if len(interruption_buffer) >= interrupt_recording_chunks:
                        # Transcribe interruption
                        log_queue.put({'type': 'status', 'text': "🎤 Transcribing interruption..."})
                        interrupt_audio = np.concatenate(interruption_buffer)
                        interrupt_text = transcribe_audio(model, interrupt_audio, sample_rate, device, log_queue)
                        
                        if interrupt_text:
                            log_queue.put({'type': 'debug', 'text': f"Interruption text: '{interrupt_text}'"})
                            
                            # Check for interruption keywords
                            interrupt_text_lower = interrupt_text.lower()
                            has_keyword = any(keyword in interrupt_text_lower for keyword in INTERRUPT_KEYWORDS)
                            
                            if has_keyword:
                                # TRUE interruption - stop playback
                                log_queue.put({'type': 'status', 'text': f"🛑 Interruption confirmed! Stopping playback."})
                                control_queue.put({'command': 'stop'})
                                
                                # Transcribe as user input
                                log_queue.put({'type': 'user', 'text': interrupt_text})
                                text_queue.put(interrupt_text)
                                
                                # Reset state
                                is_ducked = False
                                interruption_buffer = []
                                playback_baseline_window.clear()
                                continue  # Skip normal processing
                            else:
                                # FALSE alarm - restore volume
                                log_queue.put({'type': 'debug', 'text': "❌ No interruption keyword found. Restoring volume."})
                                control_queue.put({'command': 'volume', 'value': 1.0})
                                is_ducked = False
                                interruption_buffer = []
            else:
                # Not in playback - reset interruption state
                if is_ducked or len(interruption_buffer) > 0:
                    is_ducked = False
                    interruption_buffer = []
                    playback_baseline_window.clear()

            # --- C. VAD ---
            # Don't adapt noise floor if we think someone is speaking
            is_speech_frame, is_silence_frame, vol, noise_floor = vad.process_chunk(chunk, adapt=not is_speaking)

            # --- D. VISUALIZE ---
            if time.time() - last_meter_time > 0.2:
                s_thresh = noise_floor * 4.0
                meter = meters.create_volume_meter_rich(vol, noise_floor, s_thresh * 0.8, s_thresh)
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