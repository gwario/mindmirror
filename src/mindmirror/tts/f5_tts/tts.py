import sys
import time
import queue
import threading
import torch
from pathlib import Path
from mindmirror.config import F5_STYLES as STYLES, F5_NFE_STEPS as NFE_STEPS
from .utils import split_into_sentences, set_speaking_lock
from .player import playback_thread
from .loader import load_f5_model
from mindmirror.tts.interface import TTSInterface

class F5TTS(TTSInterface):
    """
    F5-TTS implementation of the TTSInterface.
    Uses flow-matching diffusion for high quality voice cloning.
    """

    def __init__(self, styles = None, nfe_steps: int = None):
        self.styles = styles or STYLES
        self.nfe_steps = nfe_steps or NFE_STEPS

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        # --- 1. SETUP ---
        try:
            from mindmirror import audio
        except ImportError:
            log_queue.put({'type': 'error', 'text': "Could not import sound.py"})
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        native_sr = audio.get_valid_samplerate(selected_device)

        # --- 2. LOAD MODEL ---
        model, vocoder = load_f5_model(log_queue, device)
        if not model:
            return # Exit if load failed

        # Need to import infer_process locally after path setup in loader
        from f5_tts.infer.utils_infer import infer_process

        # --- 3. START PLAYER ---
        stop_event = threading.Event()
        audio_queue = queue.Queue()
        player = threading.Thread(
            target=playback_thread,
            args=(audio_queue, selected_device, log_queue, control_queue, native_sr, stop_event),
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
            config = self.styles.get(style, self.styles["neutral"])

            # Shield Up
            set_speaking_lock(True)
            stop_event.clear() # Reset stop flag for new task

            try:
                for i, chunk in enumerate(chunks):
                    # Check for stop signal from player
                    if stop_event.is_set():
                        log_queue.put({'type': 'status', 'text': "🛑 Generation Stopped"})
                        break

                    log_queue.put({'type': 'status', 'text': f"🔊 Gen ({i+1}/{len(chunks)})..."})
                    start_t = time.time()

                    # INFERENCE
                    generated_audio, sample_rate, _ = infer_process(
                        config["ref_audio"], config["ref_text"], chunk, model, vocoder,
                        nfe_step=self.nfe_steps, speed=config["speed"], cfg_strength=config["cfg"],
                        device=device, mel_spec_type="vocos"
                    )

                    # SEND TO PLAYER
                    resampled = audio.resampled(generated_audio, sample_rate, native_sr)
                    audio_queue.put((resampled, native_sr))

                    log_queue.put({'type': 'debug', 'text': f"Gen: {time.time() - start_t:.2f}s"})

            except Exception as e:
                log_queue.put({'type': 'error', 'text': f"Gen Error: {e}"})

            finally:
                # Always signal end-of-paragraph so the player releases its locks,
                # even if generation was interrupted by an exception or stop event.
                audio_queue.put("DONE")