import os
import sys

import soundfile as sf
import torch

# --- CONFIGURATION ---
VOICE="MyVoice"
F5_PATH = os.path.join(os.getcwd(), "..", "F5-TTS")
CKPT_PATH = os.path.join(F5_PATH, f"ckpts/{VOICE}/model_last.pt")
VOCAB_FILE = os.path.join(F5_PATH, f"data/{VOICE}_char/vocab.txt")
WAVS_DIR = f"data/{VOICE}/wavs"

# Define your BEST recorded samples here
STYLES = {
    "neutral":   {"file": "en_A_01.wav", "ref_text": "Here is the summary of your notifications."},
    "serious":   {"file": "en_B_01.wav", "ref_text": "I'm sorry, but I cannot complete that request right now."},
    "excited":   {"file": "en_C_01.wav", "ref_text": "Wow! That worked perfectly on the first try!"},
    "lazy":      {"file": "en_D_01.wav", "ref_text": "Yeah, I think that's... mostly correct, actually."},
    "german":    {"file": "de_E_01.wav", "ref_text": "Servus, das ist jetzt ein Test für die deutsche Stimme."}
}

# The text the AI will speak
TEST_TEXT_EN = "I cannot believe we finally fixed the system. It is working."
TEST_TEXT_DE = "Das System funktioniert jetzt endlich einwandfrei."

# --- SETUP ---
sys.path.append(os.path.join(F5_PATH, "src"))
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
from f5_tts.model import DiT

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Model on {device}...")
    vocoder = load_vocoder(is_local=False)
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

    print("\n--- STARTING STYLE TEST ---")

    for style_name, data in STYLES.items():
        ref_audio = os.path.join(WAVS_DIR, data['file'])
        ref_text = data['ref_text']
        # Auto-switch text for German test
        gen_text = TEST_TEXT_DE if style_name == "german" else TEST_TEXT_EN

        # --- DYNAMIC SETTINGS ---
        if style_name == "excited":
            # Excited needs LOWER freedom (CFG) to prevent "frying" the audio
            current_cfg = 1.5
            current_speed = 0.95 # Slightly slower helps stabilize rapid shouting
        elif style_name == "lazy":
            # Lazy needs HIGHER freedom to capture the monotone drawl
            current_cfg = 2.5
            current_speed = 1.0
        else:
            # Default for Neutral / Serious / German
            current_cfg = 2.0
            current_speed = 0.95

        print(f"👉 Generating {style_name.upper()} (CFG: {current_cfg}, Speed: {current_speed})...")
        try:
            audio, sr, _ = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                model,
                vocoder,
                mel_spec_type="vocos",
                nfe_step=64,
                speed=current_speed,
                cfg_strength=current_cfg,
                device=device
            )

            # Save file
            out_name = f"test_{style_name}.wav"
            sf.write(out_name, audio, sr)
            print(f"   ✅ Saved {out_name}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\nDone! Check the folder for test_*.wav files.")

if __name__ == "__main__":
    main()