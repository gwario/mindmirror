import os
import queue
import time
import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

from mindmirror import audio, config
from mindmirror.tts.interface import TTSInterface
from mindmirror.tts.utils import set_speaking_lock, set_playback_lock

class ElevenLabsTTS(TTSInterface):
    """
    ElevenLabs Text-to-Speech implementation of the TTSInterface.
    Uses ElevenLabs Streaming TTS API for ultra-low latency voice synthesis.
    """
    def __init__(self):
        self.api_key = config.ELEVENLABS_API_KEY
        self.voice_id = config.ELEVENLABS_VOICE_ID
        self.model_id = config.ELEVENLABS_MODEL_ID
        self.client = ElevenLabs(api_key=self.api_key)

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        """Process text from AI, stream audio from ElevenLabs, and playback with interruption support."""
        log_queue.put({'type': 'info', 'text': "ElevenLabs TTS ready, waiting for responses..."})

        native_sr = audio.get_valid_samplerate(selected_device)
        BLOCK_SIZE = 2048
        SMOOTHING = 0.1

        # Nearest supported ElevenLabs PCM sample rate
        # Supported on standard tiers: 16000, 22050, 24000. 
        # (pcm_44100 is Pro-tier only, so we exclude it and rely on resampling)
        supported_srs = [16000, 22050, 24000]
        target_sr = min(supported_srs, key=lambda x: abs(x - native_sr))
        output_format = f"pcm_{target_sr}"

        while True:
            try:
                item = text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            style = "neutral"
            if isinstance(item, tuple):
                style, text = item
            else:
                text = item

            if not text.strip():
                log_queue.put({'type': 'status', 'text': "TTS: Empty text, skipping"})
                continue

            set_speaking_lock(True)
            log_queue.put({'type': 'status', 'text': f"🔊 Speaking ({style}):"})
            log_queue.put({'type': 'status', 'text': text})

            set_playback_lock(True)

            try:
                style_config = config.ELEVENLABS_STYLES.get(style.lower(), config.ELEVENLABS_STYLES["neutral"])

                log_queue.put({'type': 'status', 'text': "TTS: Querying ElevenLabs Streaming API..."})
                
                # Fetch audio stream
                audio_stream = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format=output_format,
                    voice_settings=VoiceSettings(
                        stability=style_config["stability"],
                        similarity_boost=style_config["similarity_boost"],
                        style=style_config["style"],
                        use_speaker_boost=style_config["use_speaker_boost"]
                    )
                )

                target_vol = 1.0
                current_vol = 1.0
                interrupted = False
                
                needs_resample = target_sr != native_sr
                
                byte_buffer = bytearray()
                leftover_audio = np.array([], dtype=np.float32)

                # Use sounddevice stream
                with sd.OutputStream(device=selected_device, samplerate=native_sr, channels=1, blocksize=BLOCK_SIZE) as stream:
                    for chunk in audio_stream:
                        # Check Control Queue for interruption
                        try:
                            while True:
                                cmd = control_queue.get_nowait()
                                if cmd['command'] == 'volume':
                                    target_vol = float(cmd['value'])
                                elif cmd['command'] == 'stop':
                                    interrupted = True
                                    log_queue.put({'type': 'status', 'text': "🚫 Playback Interrupted"})
                        except queue.Empty:
                            pass

                        if interrupted:
                            break
                            
                        if chunk:
                            byte_buffer.extend(chunk)
                        
                        # Process in chunks of 4096 bytes (2048 16-bit samples) to minimize resampling artifacts
                        while len(byte_buffer) >= 4096:
                            num_valid_bytes = len(byte_buffer) - (len(byte_buffer) % 2)
                            valid_bytes = bytes(byte_buffer[:num_valid_bytes])
                            del byte_buffer[:num_valid_bytes]
                            
                            raw_data = np.frombuffer(valid_bytes, dtype=np.int16)
                            audio_data = raw_data.astype(np.float32) / 32768.0
                            
                            if needs_resample:
                                resampled = audio.resampled(audio_data, target_sr, native_sr)
                            else:
                                resampled = audio_data
                                
                            current_buffer = np.concatenate((leftover_audio, resampled))
                            idx = 0
                            total_samples = len(current_buffer)
                            
                            while idx + BLOCK_SIZE <= total_samples:
                                block = current_buffer[idx:idx+BLOCK_SIZE]
                                
                                # Volume smoothing
                                if abs(current_vol - target_vol) > 0.01:
                                    vol_curve = np.linspace(current_vol, target_vol, len(block))
                                    current_vol = target_vol
                                    block = block * vol_curve
                                else:
                                    current_vol = target_vol
                                    block = block * current_vol
                                    
                                stream.write(block.astype(np.float32))
                                idx += BLOCK_SIZE
                                
                            leftover_audio = current_buffer[idx:]

                    # Process remaining audio if not interrupted
                    if not interrupted and len(byte_buffer) >= 2:
                        num_valid_bytes = len(byte_buffer) - (len(byte_buffer) % 2)
                        valid_bytes = bytes(byte_buffer[:num_valid_bytes])
                        
                        raw_data = np.frombuffer(valid_bytes, dtype=np.int16)
                        audio_data = raw_data.astype(np.float32) / 32768.0
                        
                        if needs_resample:
                            resampled = audio.resampled(audio_data, target_sr, native_sr)
                        else:
                            resampled = audio_data
                            
                        current_buffer = np.concatenate((leftover_audio, resampled))
                        
                        idx = 0
                        total_samples = len(current_buffer)
                        while idx < total_samples:
                            block_size = min(BLOCK_SIZE, total_samples - idx)
                            block = current_buffer[idx:idx+block_size]
                            
                            if abs(current_vol - target_vol) > 0.01:
                                current_vol += (target_vol - current_vol) * SMOOTHING
                            else:
                                current_vol = target_vol
                                
                            padded_block = np.zeros(BLOCK_SIZE, dtype=np.float32)
                            padded_block[:len(block)] = block * current_vol
                            stream.write(padded_block)
                            idx += block_size

                if interrupted:
                    while not text_queue.empty():
                        try:
                            text_queue.get_nowait()
                        except Exception:
                            pass

            except Exception as e:
                log_queue.put({'type': 'error', 'text': f"ElevenLabs TTS Error: {e}"})

            finally:
                set_playback_lock(False)
                set_speaking_lock(False)

            log_queue.put({'type': 'status', 'text': "TTS: Finished speaking segment"})
