from f5_tts import tts as f5_tts
# from pipervoice import tts as pipervoice_tts

def tts_task(log_queue, selected_device, text_queue):
    """
    Main TTS entry point used by main.py
    """
    try:
        f5_tts.tts_task(log_queue, selected_device, text_queue)
        # pipervoice_tts.tts_task(log_queue, selected_device, text_queue)
    except KeyboardInterrupt:
        print("TTS shutting down...")
    except Exception as e:
        log_queue.put({'type': 'error', 'text': f"TTS Wrapper Failed: {e}"})
    finally:
        pass