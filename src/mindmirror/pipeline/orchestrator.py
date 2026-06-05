import os
import signal
import sys
import time
from multiprocessing import Process, Queue, active_children

from mindmirror import audio
from mindmirror import config
from mindmirror.ui.console import console_process
from mindmirror.models.stt.wrapper import stt_task
from mindmirror.models.llm.wrapper import ttt_task
from mindmirror.models.tts.wrapper import tts_task

class VoiceAssistantPipeline:
    def __init__(self):
        self.stt_queue = Queue()       # STT -> LLM
        self.ttt_queue = Queue()       # LLM -> TTS
        self.control_queue = Queue()   # STT -> TTS (Volume/Stop)
        self.log_queue = Queue()       # ALL -> Console
        self.processes = []
        
    def setup_audio_devices(self):
        """Interactive selection of input and output hardware devices."""
        self.input_device, self.input_sr, self.output_device, self.output_sr = audio.select_audio_devices()
        
        # Enqueue startup info to be rendered by the console process
        self.log_queue.put({
            'type': 'info',
            'text': f"[green]✅ Input: {self.input_device} @ {self.input_sr}Hz | Output: {self.output_device} @ {self.output_sr}Hz[/green]"
        })

    def clean_stale_locks(self):
        """Removes any lock files left over from previous crashes."""
        if os.path.exists(config.PLAYBACK_LOCK):
            try: os.remove(config.PLAYBACK_LOCK)
            except OSError: pass
            
        if os.path.exists(config.LOCK_FILE):
            try: os.remove(config.LOCK_FILE)
            except OSError: pass

    def start(self):
        """Starts the main pipeline processes."""
        # Handle graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.clean_stale_locks()
        self.setup_audio_devices()

        # Define all subprocesses
        p_console = Process(target=console_process, args=(self.log_queue,), daemon=True)
        p_stt = Process(target=stt_task, args=(self.log_queue, self.input_device, self.stt_queue, self.control_queue), daemon=True)
        p_ttt = Process(target=ttt_task, args=(self.log_queue, self.stt_queue, self.ttt_queue), daemon=True)
        p_tts = Process(target=tts_task, args=(self.log_queue, self.output_device, self.ttt_queue, self.control_queue), daemon=True)

        self.processes = [p_console, p_stt, p_ttt, p_tts]

        # Ignite!
        for p in self.processes:
            p.start()

        try:
            # Main thread sleeps, keeping the orchestration alive
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Terminates all child processes and cleans up."""
        print("\n🛑 Shutting down pipeline...")
        for process in active_children():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
        self.clean_stale_locks()

    def _signal_handler(self, sig, frame):
        """Catches SIGINT (Ctrl+C)."""
        self.stop()
        sys.exit(0)
