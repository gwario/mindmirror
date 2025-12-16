import signal
import time
from multiprocessing import Process, Queue

import sound
from ai_ttt import ttt_task
from console import console_process, signal_handler
from stt import stt_task
from tts import tts_task

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)

    input_device, input_sr, output_device, output_sr = sound.select_audio_devices()

    stt_queue = Queue() # STT → AI
    ttt_queue = Queue() # AI → TTS
    log_queue = Queue()

    log_queue.put({
        'type': 'info',
        'text': f"[green]✅ Input: {input_device} @ {input_sr}Hz | Output: {output_device} @ {output_sr}Hz[/green]"
    })

    p_console = Process(target=console_process, args=(log_queue,), daemon=True)
    p_stt = Process(target=stt_task, args=(log_queue, input_device, stt_queue,), daemon=True)
    p_ttt = Process(target=ttt_task, args=(log_queue, stt_queue, ttt_queue), daemon=True)
    p_tts = Process(target=tts_task, args=(log_queue, output_device, ttt_queue,), daemon=True)

    p_console.start()
    p_stt.start()
    p_tts.start()
    p_ttt.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(None, None)