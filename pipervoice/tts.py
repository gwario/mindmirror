import sounddevice as sd
from piper import PiperVoice

import sound


def tts_task(log_queue, selected_device, text_queue):
    """Process text from AI and send to TTS"""
    voice = PiperVoice.load("pipervoice/en/semaine/en_GB-semaine-medium.onnx", use_cuda=False)
    log_queue.put({'type': 'info', 'text': "PiperVoice ready, waiting for responses..."})

    native_sr = sound.get_valid_samplerate(selected_device)

    while True:
        item = text_queue.get()
        if item is None: break
        if isinstance(item, tuple):
            _, text = item
        else:
            text = item

        if text.strip():
            log_queue.put({'type': 'status', 'text': f"🔊 Speaking:"})
            log_queue.put({'type': 'status', 'text': text})

            chunk_count = 0
            for chunk in voice.synthesize(text):
                chunk_count += 1
                log_queue.put({'type': 'status', 'text': f"TTS: Processing chunk {chunk_count}"})

                audio_data = chunk.audio_float_array
                original_rate = chunk.sample_rate

                resampled = sound.resampled(audio_data, original_rate, native_sr)

                sd.play(resampled, native_sr, blocking=True, device=selected_device)

            log_queue.put({'type': 'status', 'text': f"TTS: Finished speaking ({chunk_count} chunks)"})
        else:
            log_queue.put({'type': 'status', 'text': "TTS: Empty text, skipping"})
