import os
import sys
import time
from pathlib import Path

import sounddevice as sd
import torch

# DO NOT import F5-TTS here.
# Importing it here initializes CUDA in the parent process, causing the crash.
VOICE="MyVoice"
F5_PATH = os.path.join(os.getcwd(), "..", "F5-TTS")
CKPT_PATH = os.path.join(F5_PATH, f"ckpts/{VOICE}/model_last.pt")
VOCAB_FILE = os.path.join(F5_PATH, f"data/{VOICE}_char/vocab.txt")
WAVS_DIR = f"data/{VOICE}/wavs"

# Define your Reference Audio (The "Golden Samples")
# Define your BEST recorded samples here
STYLES = {
    "neutral":   {"ref_audio": WAVS_DIR + "/en_A_01.wav", "ref_text": "Here is the summary of your notifications."},
    "serious":   {"ref_audio": WAVS_DIR + "/en_B_01.wav", "ref_text": "I'm sorry, but I cannot complete that request right now."},
    "excited":   {"ref_audio": WAVS_DIR + "/en_C_01.wav", "ref_text": "Wow! That worked perfectly on the first try!"},
    "lazy":      {"ref_audio": WAVS_DIR + "/en_D_01.wav", "ref_text": "Yeah, I think that's... mostly correct, actually."},
    "german":    {"ref_audio": WAVS_DIR + "/de_E_01.wav", "ref_text": "Servus, das ist jetzt ein Test für die deutsche Stimme."}
}

def tts_task(log_queue, selected_device, text_queue):
    """Process text from AI and send to F5-TTS"""

    # --- 1. IMPORTS INSIDE THE PROCESS ---
    # This ensures CUDA is only initialized deep inside this child process,
    # completely separate from Whisper's process.
    sys.path.append(os.path.join(F5_PATH, "src"))
    try:
        from f5_tts.infer.utils_infer import load_model, infer_process, load_vocoder
        from f5_tts.model import DiT
    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import F5-TTS. Check src path."})
        return
    sys.path.append(str(Path(os.getcwd()).parent.parent))
    try:
        import sound
    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import sound.py. Check src path."})
        return

    # --- 2. CONFIGURATION ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(CKPT_PATH):
        print(f"❌ CRITICAL: Checkpoint not found at {CKPT_PATH}")
        return
    native_sr = sound.get_valid_samplerate(selected_device)

    # --- 3. LOADING ---
    log_queue.put({'type': 'info', 'text': f"Loading F5-TTS on {device} (PID: {os.getpid()})..."})
    log_queue.put({'type': 'info', 'text': "1/2 Loading Vocoder..."})
    try:
        vocoder = load_vocoder(is_local=False)
        log_queue.put({'type': 'info', 'text': "Vocoder Loaded."})
    except Exception as e:
        print(f"❌ Vocoder Failed: {e}")
        return

    # 3. Load DiT (The part that hangs)
    log_queue.put({'type': 'info', 'text': "2/2 Loading DiT Model (Please wait 10-20s)..."})
    try:
        start_t = time.time()
        model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=CKPT_PATH,
            mel_spec_type="vocos",
            vocab_file=VOCAB_FILE,
            ode_method="euler",
            use_ema=True,
            device=device
        )
        log_queue.put({'type': 'info', 'text': f"DiT Model Loaded in {time.time() - start_t:.1f}s"})
    except RuntimeError as e:
        print(f"❌ RUNTIME ERROR: {e}")
    except Exception as e:
        print(f"❌ DiT Load Failed: {e}")
        return

    log_queue.put({'type': 'info', 'text': "✅ F5-TTS Model Loaded."})

    # --- 4. MAIN LOOP ---
    while True:
        item = text_queue.get()
        if item is None: break

        if isinstance(item, tuple):
            style, text = item
        else:
            text = item
            style = "neutral"

        if style not in STYLES: style = "neutral"

        # --- DYNAMIC SETTINGS ---
        if style == "excited":
            current_cfg = 2.5
            current_speed = 0.95 # Slightly slower helps stabilize rapid shouting
        elif style == "serious":
            current_cfg = 2.2
            current_speed = 1.2
        elif style == "lazy":
            current_cfg = 2.2
            current_speed = 1.0
        else:
            # Default for Neutral / German
            current_cfg = 2.0
            current_speed = 0.8

        if text.strip():
            log_queue.put({'type': 'status', 'text': f"🔊 Generating: {text[:30]}..."})

            try:
                start_t = time.time()
                # F5-TTS Inference
                generated_audio, sample_rate, _ = infer_process(
                    STYLES[style]["ref_audio"],
                    STYLES[style]["ref_text"],
                    text,
                    model,
                    vocoder,
                    nfe_step=32,
                    speed=current_speed,
                    cfg_strength=current_cfg,
                    device=device,
                    mel_spec_type="vocos",
                )
                log_queue.put({'type': 'info', 'text': f"⏱️ Generation took {time.time() - start_t:.1f}s"})

                # Resampling for SoundDevice (if needed)
                resampled_audio = sound.resampled(generated_audio, sample_rate, native_sr)

                # Playback
                sd.play(resampled_audio, native_sr, blocking=True, device=selected_device)
                log_queue.put({'type': 'status', 'text': "✅ Finished speaking"})

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n💀 FATAL ERROR: GPU Out Of Memory.")
                    print("Try closing your browser/games or use a smaller model.")
                else:
                    print(f"\n❌ RUNTIME ERROR: {e}")
            except Exception as e:
                print(f"\n❌ UNKNOWN ERROR: {e}")
