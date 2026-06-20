import os
import queue
import time
import numpy as np
import sounddevice as sd
from google.cloud import texttospeech

from mindmirror import audio, config
from mindmirror.tts.interface import TTSInterface
from mindmirror.tts.utils import set_speaking_lock, set_playback_lock



class GoogleCloudTTS(TTSInterface):
    """
    Google Cloud Text-to-Speech implementation of the TTSInterface.
    Uses Google Cloud TTS API for natural, cloud-based synthesis.
    """
    def __init__(self, voice_name: str = None, language_code: str = None, model_name: str = None):
        self.voice_name = voice_name or config.GOOGLE_TTS_VOICE
        self.language_code = language_code or config.GOOGLE_TTS_LANG
        self.model_name = model_name or config.GOOGLE_TTS_MODEL
        self.client = None

    def _get_client(self):
        if not self.client:
            self.client = texttospeech.TextToSpeechClient()
        return self.client

    def tts_task(self, log_queue, selected_device, text_queue, control_queue) -> None:
        """Process text from AI, fetch audio from Google Cloud TTS, and stream playback with interruption support."""
        log_queue.put({'type': 'info', 'text': "Google Cloud TTS ready, waiting for responses..."})

        native_sr = audio.get_valid_samplerate(selected_device)
        BLOCK_SIZE = 2048
        SMOOTHING = 0.1

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
                # Resolve pitch and speaking rate based on style
                style_config = config.GOOGLE_TTS_STYLES.get(style.lower(), config.GOOGLE_TTS_STYLES["neutral"])
                speaking_rate = style_config.get("speaking_rate", 1.0)
                pitch = style_config.get("pitch", 0.0)
                style_prompt = style_config.get("prompt", None)

                # Initialize request
                client = self._get_client()
                
                if style_prompt:
                    synthesis_input = texttospeech.SynthesisInput(text=text, prompt=style_prompt)
                else:
                    synthesis_input = texttospeech.SynthesisInput(text=text)

                # Normalise language code (e.g. en-AU or en-GB)
                normalized_lang = self.language_code.replace('_', '-')
                lang_parts = normalized_lang.split('-')
                if len(lang_parts) == 2:
                    normalized_lang = f"{lang_parts[0].lower()}-{lang_parts[1].upper()}"
                else:
                    normalized_lang = normalized_lang.lower()

                voice = texttospeech.VoiceSelectionParams(
                    language_code=normalized_lang,
                    name=self.voice_name,
                    model_name=self.model_name
                )

                # Request LINEAR16 PCM at 24000Hz
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,
                    speaking_rate=speaking_rate,
                    pitch=pitch
                )

                log_queue.put({'type': 'status', 'text': "TTS: Querying Google Cloud API..."})
                
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )

                # Parse PCM bytes (16-bit signed int) to float32 numpy array
                raw_data = np.frombuffer(response.audio_content, dtype=np.int16)
                audio_data = raw_data.astype(np.float32) / 32768.0

                # Resample from 24000Hz to native output sample rate
                resampled = audio.resampled(audio_data, 24000, native_sr)

                total_samples = len(resampled)
                idx = 0
                target_vol = 1.0
                current_vol = 1.0
                interrupted = False

                with sd.OutputStream(device=selected_device, samplerate=native_sr, channels=1, blocksize=BLOCK_SIZE) as stream:
                    while idx < total_samples:
                        # Check Control Queue
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

                        # Volume smoothing
                        if abs(current_vol - target_vol) > 0.01:
                            current_vol += (target_vol - current_vol) * SMOOTHING
                        else:
                            current_vol = target_vol

                        # Prepare blocks
                        end = min(idx + BLOCK_SIZE, total_samples)
                        block = resampled[idx:end]
                        block = block * current_vol

                        # Write block to speaker stream
                        stream.write(block.astype(np.float32))
                        idx = end

                if interrupted:
                    # Clear remaining text queue to prevent backlog
                    while not text_queue.empty():
                        try:
                            text_queue.get_nowait()
                        except Exception:
                            pass

            except Exception as e:
                log_queue.put({'type': 'error', 'text': f"Google Cloud TTS Error: {e}"})

            finally:
                set_playback_lock(False)
                set_speaking_lock(False)

            log_queue.put({'type': 'status', 'text': "TTS: Finished speaking segment"})
