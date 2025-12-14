# from pipervoice import tts as pipervoice_tts
from f5_tts import tts as f5_tts

def tts_task(log_queue, selected_device, text_queue):
    f5_tts.tts_task(log_queue, selected_device, text_queue)
    # pipervoice_tts.tts_task(log_queue, selected_device, text_queue)