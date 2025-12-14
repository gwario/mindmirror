import sys
import os
import torch

# Point to your F5-TTS src folder
VOICE="MyVoice"
F5_PATH = os.path.join(os.getcwd(), "..", "F5-TTS")
CKPT_PATH = os.path.join(F5_PATH, f"ckpts/{VOICE}/model_last.pt")
VOCAB_FILE = os.path.join(F5_PATH, f"data/{VOICE}_char/vocab.txt")
WAVS_DIR = f"data/{VOICE}/wavs"

sys.path.append(os.path.join(F5_PATH, "src"))

from f5_tts.infer.utils_infer import load_model
from f5_tts.model import DiT

print("1. Starting Test Load...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Attempt to load just like the worker does
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
    print("2. SUCCESS! Model loaded.")
except Exception as e:
    print(f"3. FAILURE: {e}")