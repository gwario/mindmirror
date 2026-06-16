import sys
import os
import torch
from pathlib import Path

# Add src folder to sys.path to allow importing mindmirror config
src_path = str(Path(__file__).resolve().parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from mindmirror import config

F5_PATH = config.F5_LIB_PATH
CKPT_PATH = config.F5_CKPT_PATH
VOCAB_FILE = config.F5_VOCAB_FILE
WAVS_DIR = config.F5_WAVS_DIR

sys.path.append(str(F5_PATH / "src"))

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