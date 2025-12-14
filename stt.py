from whisper_stt import stt as whisper_stt

def stt_task(log_queue, selected_device, text_queue):
    whisper_stt.continuous_stt_task(log_queue, selected_device, text_queue)