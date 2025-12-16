import sys
from .config import F5_LIB_PATH, CKPT_PATH, VOCAB_FILE

def load_f5_model(log_queue, device):
    """
    Sets up system paths and loads the F5-TTS DiT model.
    """
    # 1. Setup Paths
    f5_src = F5_LIB_PATH / "src"
    if not f5_src.exists():
        log_queue.put({'type': 'error', 'text': f"Missing F5-TTS at {f5_src}"})
        return None, None

    sys.path.append(str(f5_src))

    # 2. Import (Lazy import to avoid global pollution)
    try:
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        log_queue.put({'type': 'info', 'text': f"Loading F5-TTS on {device}..."})

        vocoder = load_vocoder(is_local=False)
        model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=str(CKPT_PATH),
            mel_spec_type="vocos",
            vocab_file=str(VOCAB_FILE),
            ode_method="euler",
            use_ema=True,
            device=device
        )
        return model, vocoder

    except ImportError as e:
        log_queue.put({'type': 'error', 'text': f"Import Error: {e}"})
        return None, None
    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Model Load Error: {e}"})
        return None, None