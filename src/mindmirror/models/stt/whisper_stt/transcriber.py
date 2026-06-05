import os
import torch
import tempfile
import soundfile as sf


def load_whisper_model(log_queue):
    """
    Loads and returns the Whisper model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import whisper
        log_queue.put({'type': 'info', 'text': f"Loading Whisper on {device}..."})

        # Load model and return it
        model = whisper.load_model("small", device=device)

        log_queue.put({'type': 'success', 'text': "✅ Whisper Model Loaded."})
        return model, device

    except ImportError:
        log_queue.put({'type': 'error', 'text': "Could not import 'whisper'. Install it first."})
        return None, None
    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Whisper Load Failed: {e}"})
        return None, None

def transcribe_audio(model, audio_data, sample_rate, device, log_queue):
    """
    Runs inference on a numpy audio array.
    """
    if not model: return None

    try:
        # Write to temp file (Whisper requirement)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            filename = f.name
            sf.write(filename, audio_data, sample_rate)

        # Inference
        # fp16=True is faster on GPU, but crashes on CPU
        use_fp16 = (device == "cuda")

        result = model.transcribe(filename, language='en', fp16=use_fp16)
        text = result['text'].strip()

        # Cleanup
        try:
            os.remove(filename)
        except OSError:
            pass

        return text

    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"Transcribe Error: {e}"})
        return None