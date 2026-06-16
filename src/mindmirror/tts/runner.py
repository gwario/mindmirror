def run_tts_loop(tts_class, tts_kwargs, log_queue, selected_device, text_queue, control_queue):
    """
    Generic worker that instantiates the given TTS engine inside the child process
    and executes its synthesis/playback loop.
    """
    engine = tts_class(**tts_kwargs)
    engine.tts_task(log_queue, selected_device, text_queue, control_queue)
