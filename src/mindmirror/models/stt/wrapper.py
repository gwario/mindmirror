from mindmirror.models.stt.whisper_stt import stt as whisper_stt


def stt_task(log_queue, selected_device, text_queue, control_queue):
    try:
        whisper_stt.stt_task(log_queue, selected_device, text_queue, control_queue)
    except KeyboardInterrupt:
        print("STT shutting down...")
    finally:
        pass