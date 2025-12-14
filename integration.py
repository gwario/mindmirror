from multiprocessing import Process, Queue
from stt import stt_task
from ai_ttt import ttt_task
from tts import tts_task
from console import console_process
from sound import select_audio_device, get_valid_samplerate

if __name__ == '__main__':

    selected_device, native_sr = select_audio_device()

    stt_queue = Queue() # STT → AI
    ttt_queue = Queue() # AI → TTS
    log_queue = Queue()

    log_queue.put({'type': 'info', 'text': f"[green]✅ Using Device: {selected_device} | Rate: {native_sr}Hz[/green]"})

    p_console = Process(target=console_process, args=(log_queue,))
    p_stt = Process(target=stt_task, args=(log_queue, selected_device, stt_queue,))
    p_ttt = Process(target=ttt_task, args=(log_queue, stt_queue, ttt_queue))
    p_tts = Process(target=tts_task, args=(log_queue, selected_device, ttt_queue,))

    p_console.start()
    p_stt.start()
    p_tts.start()
    p_ttt.start()

    p_console.join()
    p_stt.join()
    p_ttt.join()
    p_tts.join()