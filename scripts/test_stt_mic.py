import sys
import time
import queue
from pathlib import Path

# Add src to sys.path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mindmirror import audio, config
from mindmirror.stt.google.stt import GoogleCloudSTT
from mindmirror.stt.runner import run_stt_loop

def main():
    print("=" * 60)
    print("🧪 Testing STT Microphone Capture & Transcription")
    print("=" * 60)
    
    device_id, sr = audio.select_audio_device("input")
    print("Selected input device:", device_id, "Sample rate:", sr)
    
    log_q = queue.Queue()
    text_q = queue.Queue()
    control_q = queue.Queue()
    
    device_id = audio.get_default_device(is_input=True)
    
    print("\n👂 Initializing GoogleCloudSTT and starting listening loop for 15 seconds...")
    print(">>> PLEASE SPEAK INTO THE MICROPHONE NOW <<<")
    print("-" * 60)

    import threading
    stt_thread = threading.Thread(
        target=run_stt_loop,
        args=(GoogleCloudSTT, {"language_code": config.GOOGLE_STT_LANG, "model": config.GOOGLE_STT_MODEL}, log_q, device_id, text_q, control_q, True),
        daemon=True
    )
    stt_thread.start()

    start_time = time.time()
    while time.time() - start_time < 15.0:
        # Check logs
        try:
            while True:
                msg = log_q.get_nowait()
                mtype = msg.get("type", "")
                text = msg.get("text", "")
                if mtype != "meter":
                    print(f"[{mtype.upper()}] {text}")
        except queue.Empty:
            pass

        # Check transcribed text
        try:
            while True:
                transcribed = text_q.get_nowait()
                print(f"\n🎉 SUCCESSFUL STT TRANSCRIPTION: '{transcribed}'\n")
        except queue.Empty:
            pass

        time.sleep(0.1)

    print("-" * 60)
    print("✅ Test complete!")

if __name__ == "__main__":
    main()
