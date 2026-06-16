import sounddevice as sd
from piper import PiperVoice
from mindmirror import audio
import numpy as np
import os
import queue
import time
from pathlib import Path
import sys

from mindmirror import config
from mindmirror.tts.interface import TTSInterface

def set_speaking_lock(active: bool):
    """Creates or removes the lock file to mute the mic."""
    try:
        if active:
            with open(config.LOCK_FILE, "w") as f: f.write("active")
        else:
            if os.path.exists(config.LOCK_FILE):
                os.remove(config.LOCK_FILE)
    except Exception:
        pass

def set_playback_lock(active: bool):
    """Creates or removes the playback lock file to indicate active audio playback."""
    try:
        if active:
            with open(config.PLAYBACK_LOCK, "w") as f: f.write("1")
        else:
            if os.path.exists(config.PLAYBACK_LOCK):
                os.remove(config.PLAYBACK_LOCK)
    except Exception:
        pass

class PiperTTS(TTSInterface):
    """
    PiperVoice implementation of the TTSInterface.
    Uses ONNX for fast CPU speech synthesis.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.PIPER_MODEL_PATH

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        """Process text from AI and send to TTS with interruption support"""
        if not os.path.exists(self.model_path):
            log_queue.put({'type': 'error', 'text': f"Piper Model not found at {self.model_path}"})
            return

        voice = PiperVoice.load(self.model_path, use_cuda=False)
        log_queue.put({'type': 'info', 'text': "PiperVoice ready, waiting for responses..."})

        native_sr = audio.get_valid_samplerate(selected_device)
        BLOCK_SIZE = 2048
        SMOOTHING = 0.1

        while True:
            try:
                item = text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
                
            if item is None: break
            
            if isinstance(item, tuple):
                _, text = item
            else:
                text = item

            if text.strip():
                set_speaking_lock(True)
                log_queue.put({'type': 'status', 'text': f"🔊 Speaking:"})
                log_queue.put({'type': 'status', 'text': text})
                print(f"DEBUG: Piper TTS received text: {text[:20]}...")

                # Create Playback Lock (Audio Output)
                set_playback_lock(True)

                try:
                    with sd.OutputStream(device=selected_device, samplerate=native_sr, channels=1, blocksize=BLOCK_SIZE) as stream:
                        target_vol = 1.0
                        current_vol = 1.0
                        interrupted = False
                        
                        chunk_count = 0
                        
                        for chunk in voice.synthesize(text):
                            if interrupted: break
                            
                            chunk_count += 1
                            log_queue.put({'type': 'status', 'text': f"TTS: Processing chunk {chunk_count}"})

                            audio_data = chunk.audio_float_array
                            original_rate = chunk.sample_rate

                            # Resample
                            resampled = audio.resampled(audio_data, original_rate, native_sr)
                            
                            # Play Loop for this chunk
                            total_samples = len(resampled)
                            idx = 0
                            
                            while idx < total_samples:
                                # Check Control Queue
                                try:
                                    while True:
                                        cmd = control_queue.get_nowait()
                                        if cmd['command'] == 'volume':
                                            target_vol = float(cmd['value'])
                                        elif cmd['command'] == 'stop':
                                            interrupted = True
                                            log_queue.put({'type': 'status', 'text': "🚫 Playback Interrupted"})
                                except queue.Empty:
                                    pass

                                if interrupted:
                                    break
                                
                                # Smoothing Volume
                                if abs(current_vol - target_vol) > 0.01:
                                    current_vol += (target_vol - current_vol) * SMOOTHING
                                else:
                                    current_vol = target_vol

                                # Prepare block
                                end = min(idx + BLOCK_SIZE, total_samples)
                                block = resampled[idx:end]
                                
                                # Apply Volume
                                block = block * current_vol
                                
                                # Write
                                stream.write(block.astype(np.float32))
                                idx = end
                        
                        if interrupted:
                            # Clear text queue
                            while not text_queue.empty():
                                try: text_queue.get_nowait()
                                except: pass
                
                except Exception as e:
                    log_queue.put({'type': 'error', 'text': f"Piper TTS Error: {e}"})
                
                finally:
                    # Remove Playback Lock
                    set_playback_lock(False)
                    set_speaking_lock(False)
                    
                log_queue.put({'type': 'status', 'text': f"TTS: Finished ({chunk_count} chunks)"})
            else:
                log_queue.put({'type': 'status', 'text': "TTS: Empty text, skipping"})
