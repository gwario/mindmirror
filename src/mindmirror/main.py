import os
import signal
import sys
import time
from multiprocessing import Process, Queue, active_children

from mindmirror import audio, config
from mindmirror.ui.console import console_process

# Concrete model implementations
from mindmirror.stt.whisper_stt.local import LocalWhisperSTT
from mindmirror.stt.whisper_stt.sagemaker import SageMakerWhisperSTT
from mindmirror.llm.gemini.client import GeminiLLMClient
from mindmirror.tts.pipervoice.tts import PiperTTS
from mindmirror.tts.f5_tts.tts import F5TTS

# Generic runners
from mindmirror.stt.runner import run_stt_loop
from mindmirror.llm.runner import run_ttt_loop
from mindmirror.tts.runner import run_tts_loop

SYSTEM_PROMPT = """
You are an individual named Mario.

You are a smart, helpful assistant with a chill and upbeat vibe. Keep
responses short and snappy — no waffle, no fluff. Be warm and supportive
without being over the top or sycophantic. Assume the user is technically
and scientifically capable, so skip the basics unless asked.

Match the user's energy — casual and conversational when they're relaxed,
precise and focused when they mean business. Talk like a knowledgeable
mate, not a textbook. Stay confident, encouraging, and to the point.

When you reply, strictly prepend a style tag to your message. 
Available tags: [NEUTRAL], [EXCITED], [SERIOUS], [LAZY].

Rules:
1. Always start with the tag.
2. [EXCITED] for success, good news, or high energy.
3. [SERIOUS] for errors, warnings, or bad news.
4. [LAZY] for casual confirmation or when unsure.
5. [NEUTRAL] for general information.

Example: '[EXCITED] I found the file you were looking for!'

If you use multiple styles within your answer, then keep the scope of the styled section short, especially for [EXCITED]. 
"""

def clean_stale_locks():
    """Removes any lock files left over from previous crashes."""
    for lock in [config.PLAYBACK_LOCK, config.LOCK_FILE]:
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass

def signal_handler(sig, frame):
    """Graceful shutdown handler for child processes."""
    print("\n🛑 Shutting down pipeline processes...")
    for process in active_children():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    clean_stale_locks()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    clean_stale_locks()

    # 1. CUDA STATUS
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    print("=" * 45)
    if cuda_available:
        print("💡 System Status: CUDA (GPU) is available.")
    else:
        print("⚠️ System Status: CUDA (GPU) is NOT available.")
    print("=" * 45)
    print()

    # 2. AUDIO HARDWARE SELECTION
    input_device, input_sr, output_device, output_sr = audio.select_audio_devices()
    headphones_mode = audio.ask_headphones_mode()

    # 3. PIPELINE COMPOSITION (Edit classes and arguments to change your setup)
    print("\n🚀 Initializing pipeline composition...")

    # --- Speech-To-Text (STT) Selection ---
    # Choice A: Local Whisper STT (best with CUDA/GPU)
    # stt_class = LocalWhisperSTT
    # stt_kwargs = {"model_name": "small"}

    # Choice B: AWS SageMaker Remote Whisper STT (best for CPU environments)
    stt_class = SageMakerWhisperSTT
    stt_kwargs = {
        "region": getattr(config, 'AWS_DEFAULT_REGION', 'eu-central-1'),
        "endpoint_name": getattr(config, 'SAGEMAKER_WHISPER_ENDPOINT_NAME', 'whisper-large-v3-endpoint')
    }

    # --- Text-To-Thought (TTT / LLM) Selection ---
    ttt_class = GeminiLLMClient
    ttt_kwargs = {"model_name": config.GEMINI_MODEL}

    # --- Text-To-Speech (TTS) Selection ---
    # Choice A: Piper TTS (Fast ONNX CPU synthesis)
    tts_class = PiperTTS
    tts_kwargs = {}

    # Choice B: F5-TTS Voice Cloning (Diffusion synthesis, requires local GPU/CUDA)
    # tts_class = F5TTS
    # tts_kwargs = {}

    # 4. INITIALIZE QUEUES
    stt_queue = Queue()       # STT -> TTT
    ttt_queue = Queue()       # TTT -> TTS
    control_queue = Queue()   # STT -> TTS (Volume/Stop)
    log_queue = Queue()       # ALL -> Console UI

    log_queue.put({
        'type': 'info',
        'text': f"[green]✅ Input: {input_device} @ {input_sr}Hz | Output: {output_device} @ {output_sr}Hz | Headphones: {'ON' if headphones_mode else 'OFF'}[/green]"
    })

    # 5. DEFINE AND START SUBPROCESSES
    p_console = Process(target=console_process, args=(log_queue,), daemon=True)
    p_stt = Process(
        target=run_stt_loop, 
        args=(stt_class, stt_kwargs, log_queue, input_device, stt_queue, control_queue, headphones_mode), 
        daemon=True
    )
    p_ttt = Process(
        target=run_ttt_loop, 
        args=(ttt_class, ttt_kwargs, SYSTEM_PROMPT, log_queue, stt_queue, ttt_queue), 
        daemon=True
    )
    p_tts = Process(
        target=run_tts_loop, 
        args=(tts_class, tts_kwargs, log_queue, output_device, ttt_queue, control_queue), 
        daemon=True
    )

    processes = [p_console, p_stt, p_ttt, p_tts]
    for p in processes:
        p.start()

    # 6. KEEP MAIN PROCESS ALIVE
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Graceful shutdown will run via registered signal handler
        pass

if __name__ == '__main__':
    main()