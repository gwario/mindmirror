import sounddevice as sd
from .utils import set_speaking_lock

def playback_thread(audio_queue, device_id, log_queue):
    """
    Consumer thread: Plays audio from the queue.
    """
    while True:
        item = audio_queue.get()

        # 1. Stop Signal (Process dying)
        if item is None:
            break

        # 2. End of Paragraph Signal
        if item == "DONE":
            set_speaking_lock(False) # Lower shield
            continue

        # 3. Play Audio
        audio_data, sr = item
        try:
            sd.play(audio_data, sr, device=device_id, blocking=True)
        except Exception as e:
            log_queue.put({'type': 'error', 'text': f"Playback Error: {e}"})