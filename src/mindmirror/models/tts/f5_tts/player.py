import sounddevice as sd
import numpy as np
import queue
import time
import os
from .utils import set_speaking_lock
from mindmirror.config import PLAYBACK_LOCK

def playback_thread(audio_queue, device_id, log_queue, control_queue, native_sr, stop_event):
    """
    Consumer thread: Plays audio from the queue with volume control and interruption.
    """
    BLOCK_SIZE = 2048
    target_vol = 1.0
    current_vol = 1.0
    SMOOTHING = 0.1

    try:
        with sd.OutputStream(device=device_id, samplerate=native_sr, channels=1, blocksize=BLOCK_SIZE) as stream:
            while True:
                try:
                    item = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check control queue even when idle
                    try:
                        while True:
                            cmd = control_queue.get_nowait()
                            if cmd['command'] == 'volume':
                                target_vol = cmd['value']
                            # processing stop here doesn't make much sense if idle, but safe
                    except queue.Empty:
                        pass
                    continue

                # 1. Stop Signal (Process dying)
                if item is None:
                    break

                # 2. End of Paragraph Signal
                if item == "DONE":
                    set_speaking_lock(False) # Lower shield
                    # Reset volume for next time
                    target_vol = 1.0
                    current_vol = 1.0
                    
                    # Remove Playback Lock
                    if os.path.exists(PLAYBACK_LOCK):
                        try:
                            os.remove(PLAYBACK_LOCK)
                        except: pass
                    continue

                # 3. Play Audio
                # Create Playback Lock if not exists
                if not os.path.exists(PLAYBACK_LOCK):
                    try:
                        with open(PLAYBACK_LOCK, 'w') as f: f.write("1")
                    except: pass

                audio_data, sr = item
                # Resample if needed (should be already resampled but safety first)
                # We assume sr == native_sr here as per design
                
                # Iterate in blocks
                total_samples = len(audio_data)
                idx = 0
                
                interrupted = False

                while idx < total_samples:
                    # Check Control Queue
                    try:
                        while True:
                            cmd = control_queue.get_nowait()
                            if cmd['command'] == 'volume':
                                target_vol = float(cmd['value'])
                            elif cmd['command'] == 'stop':
                                stop_event.set()
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

                    # Prepare chunk
                    end = min(idx + BLOCK_SIZE, total_samples)
                    chunk = audio_data[idx:end]
                    
                    # Apply Volume
                    chunk = chunk * current_vol
                    
                    # Write to Stream
                    stream.write(chunk.astype(np.float32))
                    idx = end

                if interrupted:
                    # Drain queue until DONE
                    while True:
                        try:
                            drained = audio_queue.get()
                            if drained == "DONE":
                                break
                            if drained is None: # Should re-put or handle?
                                # If we drained None, we should probably stop entirely
                                audio_queue.put(None)
                                break
                        except:
                            break
                    set_speaking_lock(False)
                    target_vol = 1.0
                    current_vol = 1.0

    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Playback Error: {e}"})
        # Try to clean up lock if crash
        set_speaking_lock(False)
    finally:
        if os.path.exists(PLAYBACK_LOCK):
            try:
                os.remove(PLAYBACK_LOCK)
            except: pass