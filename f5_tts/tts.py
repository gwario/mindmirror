import sys
import time
import queue
import threading
import torch
from pathlib import Path
from .config import STYLES, NFE_STEPS
from .utils import split_into_sentences, set_speaking_lock
from .player import playback_thread
from .loader import load_f5_model


def tts_task(log_queue, selected_device_id, text_queue):

    # --- 1. SETUP ---
    # Add root to path for sound.py
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    try:
        import sound
    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import sound.py"})
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    native_sr = sound.get_valid_samplerate(selected_device_id)

    # --- 2. LOAD MODEL ---
    model, vocoder = load_f5_model(log_queue, device)
    if not model:
        return # Exit if load failed

    # Need to import infer_process locally after path setup in loader
    from f5_tts.infer.utils_infer import infer_process

    # --- 3. START PLAYER ---
    audio_queue = queue.Queue()
    player = threading.Thread(
        target=playback_thread,
        args=(audio_queue, selected_device_id, log_queue),
        daemon=True
    )
    player.start()
    log_queue.put({'type': 'success', 'text': "F5-TTS System Ready."})

    # --- 4. MAIN LOOP ---
    while True:
        task = text_queue.get()
        if task is None:
            audio_queue.put(None) # Kill player
            player.join()
            break

        style, raw_text = task if isinstance(task, tuple) else ("neutral", task)
        if not raw_text or not raw_text.strip(): continue

        chunks = split_into_sentences(raw_text)
        config = STYLES.get(style, STYLES["neutral"])

        # Shield Up
        set_speaking_lock(True)

        try:
            for i, chunk in enumerate(chunks):
                log_queue.put({'type': 'status', 'text': f"🔊 Gen ({i+1}/{len(chunks)})..."})
                start_t = time.time()

                # INFERENCE
                generated_audio, sample_rate, _ = infer_process(
                    config["ref_audio"], config["ref_text"], chunk, model, vocoder,
                    nfe_step=NFE_STEPS, speed=config["speed"], cfg_strength=config["cfg"],
                    device=device, mel_spec_type="vocos"
                )

                # SEND TO PLAYER
                resampled = sound.resampled(generated_audio, sample_rate, native_sr)
                audio_queue.put((resampled, native_sr))

                log_queue.put({'type': 'debug', 'text': f"Gen: {time.time() - start_t:.2f}s"})

            audio_queue.put("DONE") # Signal end of paragraph

        except Exception as e:
            log_queue.put({'type': 'error', 'text': f"Gen Error: {e}"})
            set_speaking_lock(False)