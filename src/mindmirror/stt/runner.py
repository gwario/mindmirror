import time
import queue
import os
import numpy as np
from mindmirror.config import (
    LOCK_FILE, PLAYBACK_LOCK, CHUNK_DURATION, MIN_AUDIO_LENGTH, SILENCE_DURATION,
    COOLDOWN_DURATION, LOOP_SLEEP_TIME, QUEUE_TIMEOUT,
    INTERRUPT_ENERGY_MULTIPLIER, INTERRUPT_BASELINE_WINDOW, 
    INTERRUPT_RECORDING_DURATION, DUCK_VOLUME, INTERRUPT_KEYWORDS,
    POST_PLAYBACK_COOLDOWN
)
from mindmirror import audio
from mindmirror.ui import meters
from mindmirror.stt.whisper_stt.vad import VADEngine

def run_stt_loop(stt_class, stt_kwargs, log_queue, selected_device, text_queue, control_queue, headphones_mode=False):
    # 1. SETUP ENGINE
    stt_engine = stt_class(**stt_kwargs, log_queue=log_queue)
    try:
        stt_engine.load_model()
    except Exception:
        return

    sample_rate = audio.get_valid_samplerate(selected_device)
    vad = VADEngine()

    audio_queue = queue.Queue()
    preroll_buffer = audio.create_preroll_buffer(sample_rate, CHUNK_DURATION)
    buffer = []

    is_speaking = False
    silence_counter = 0
    last_playback_time = 0.0

    # DYNAMIC CALCULATIONS
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

    last_log_time = [0.0]

    def audio_callback(indata, frames, pa_time, status):
        if status:
            status_str = str(status)
            if "input overflow" in status_str:
                import time as time_module
                now = time_module.time()
                if now - last_log_time[0] > 3.0:
                    log_queue.put({'type': 'debug', 'text': f"Audio Input Warning: {status_str} (throttled)"})
                    last_log_time[0] = now
            else:
                log_queue.put({'type': 'error', 'text': f"Audio Input Error: {status_str}"})
        audio_queue.put(indata.copy())

    with audio.safe_open_stream(selected_device, sample_rate, callback=audio_callback,
                                blocksize=int(sample_rate * CHUNK_DURATION)):

        log_queue.put({'type': 'info', 'text': "👂 Listening..."})

        was_muted = False
        while True:
            # Check Playback and Cooldown
            playback_active = os.path.exists(PLAYBACK_LOCK)
            speaking_active = os.path.exists(LOCK_FILE)
            if playback_active:
                last_playback_time = time.time()

            if not headphones_mode:
                if playback_active or speaking_active or (time.time() - last_playback_time < POST_PLAYBACK_COOLDOWN):
                    # Drain queue to prevent backlog
                    try:
                        while True:
                            audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    preroll_buffer.clear()
                    buffer.clear()
                    is_speaking = False
                    silence_counter = 0
                    was_muted = True
                    
                    time.sleep(LOOP_SLEEP_TIME)
                    continue

            # If transitioning from muted back to listening, perform a final drain
            # of any audio that was queued during the last sleep/cooldown transition.
            if was_muted:
                try:
                    while True:
                        audio_queue.get_nowait()
                except queue.Empty:
                    pass
                preroll_buffer.clear()
                buffer.clear()
                is_speaking = False
                silence_counter = 0
                was_muted = False

            # --- B. PROCESS AUDIO ---
            try:
                chunk = audio_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            preroll_buffer.append(chunk)

            # --- INTERRUPTION DETECTION ---
            if os.path.exists(PLAYBACK_LOCK):
                energy = np.sqrt(np.mean(chunk**2)) * 10
                playback_baseline_window.append(energy)
                
                if len(playback_baseline_window) >= 3:
                    playback_baseline = np.median(list(playback_baseline_window))
                    playback_baseline = max(playback_baseline, 0.01)  # Floor
                
                if not is_ducked and energy > playback_baseline * INTERRUPT_ENERGY_MULTIPLIER:
                    log_queue.put({'type': 'status', 'text': f"📉 Possible interruption (Energy: {energy:.3f} > {playback_baseline * INTERRUPT_ENERGY_MULTIPLIER:.3f})"})
                    control_queue.put({'command': 'volume', 'value': DUCK_VOLUME})
                    is_ducked = True
                    interruption_buffer = [chunk]
                
                elif is_ducked:
                    interruption_buffer.append(chunk)
                    
                    if len(interruption_buffer) >= interrupt_recording_chunks:
                        log_queue.put({'type': 'status', 'text': "🎤 Transcribing interruption..."})
                        interrupt_audio = np.concatenate(interruption_buffer)
                        interrupt_text = stt_engine.transcribe(interrupt_audio, sample_rate)
                        
                        if interrupt_text:
                            log_queue.put({'type': 'debug', 'text': f"Interruption text: '{interrupt_text}'"})
                            
                            interrupt_text_lower = interrupt_text.lower()
                            has_keyword = any(keyword in interrupt_text_lower for keyword in INTERRUPT_KEYWORDS)
                            
                            if has_keyword:
                                log_queue.put({'type': 'status', 'text': f"🛑 Interruption confirmed! Stopping playback."})
                                control_queue.put({'command': 'stop'})
                                
                                log_queue.put({'type': 'user', 'text': interrupt_text})
                                text_queue.put(interrupt_text)
                                
                                is_ducked = False
                                interruption_buffer = []
                                playback_baseline_window.clear()
                                continue
                            else:
                                log_queue.put({'type': 'debug', 'text': "❌ No interruption keyword found. Restoring volume."})
                                control_queue.put({'command': 'volume', 'value': 1.0})
                                is_ducked = False
                                interruption_buffer = []
            else:
                if is_ducked or len(interruption_buffer) > 0:
                    is_ducked = False
                    interruption_buffer = []
                    playback_baseline_window.clear()

            # --- C. VAD ---
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
                    if len(buffer) * CHUNK_DURATION > MIN_AUDIO_LENGTH:
                        log_queue.put({'type': 'status', 'text': "⏳ Transcribing..."})
                        full_audio = np.concatenate(buffer)
                        text = stt_engine.transcribe(full_audio, sample_rate)
                        if text:
                            log_queue.put({'type': 'user', 'text': text})
                            text_queue.put(text)
                    else:
                        log_queue.put({'type': 'status', 'text': "🚫 Too short"})

                    buffer = []; is_speaking = False; silence_counter = 0
