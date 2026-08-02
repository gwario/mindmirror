"""
test_google_cloud_tts.py — Test Google Cloud Instant Custom Voice Cloning
========================================================================

Usage:
    cd /path/to/mindmirror
    PYTHONPATH=src \python scripts/test_google_cloud_tts.py

This script:
  1. Loads your generated voice cloning keys from voice_cloning_keys.json
  2. Prompts you to select which emotion to synthesize with
  3. Queries Google Cloud TTS to synthesize a test phrase in your cloned voice using streaming_synthesize
  4. Plays the synthesized audio back on your speakers.
  5. Saves the generated file as test_google_voice.wav.
"""
import os
import sys
import json
from pathlib import Path
import re

import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import texttospeech

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mindmirror import config
from mindmirror.ui.ui import console

# ================= CONFIGURATION =================
VOICE = "ReferenceVoice"
GCTTS_BASE = "data/GoogleCloudTTS"
OUTPUT_BASE = os.path.join(GCTTS_BASE, VOICE)
KEYS_FILE = os.path.join(OUTPUT_BASE, "voice_cloning_keys.json")
OUTPUT_FILE = "test_google_voice.wav"
DEFAULT_TEXT = "Hello! This is a test of my Google Cloud Instant Custom Voice clone. Let's see how natural and expressive it sounds."


def main():
    if not os.path.exists(KEYS_FILE):
        console.print(f"[bold red]❌ Voice cloning keys not found at {KEYS_FILE}[/bold red]")
        console.print("[yellow]Please run scripts/generate_google_cloud_tts_keys.py first to generate your keys.[/yellow]")
        return

    # 1. Load the Voice Cloning Keys
    console.print(f"[blue]📂 Loading voice cloning keys from {KEYS_FILE}...[/blue]")
    with open(KEYS_FILE, "r") as f:
        try:
            keys_map = json.load(f)
        except json.JSONDecodeError:
            console.print("[bold red]❌ Failed to parse voice_cloning_keys.json[/bold red]")
            return

    if not keys_map:
        console.print("[bold red]❌ Voice cloning keys map is empty![/bold red]")
        return

    # 2. Select Emotion
    emotions = list(keys_map.keys())
    console.print("\n[bold]Available Emotions:[/bold]")
    for idx, emotion in enumerate(emotions):
        console.print(f"  [{idx + 1}] {emotion}")
    
    choice = input(f"\nSelect emotion to test [1–{len(emotions)}, default=1]: ").strip()
    try:
        idx = int(choice) - 1 if choice else 0
        selected_emotion = emotions[max(0, min(idx, len(emotions) - 1))]
    except ValueError:
        selected_emotion = emotions[0]

    voice_key = keys_map[selected_emotion]
    console.print(f"[green]✅ Selected Emotion: {selected_emotion}[/green]")

    # 3. Get input text to synthesize
    console.print(f"\n[bold]Enter text to synthesize (or press Enter for default):[/bold]")
    user_text = input("> ").strip()
    text_to_synthesize = user_text if user_text else DEFAULT_TEXT

    # 4. Setup client and request
    console.print("\n[blue]📡 Calling Google Cloud Text-to-Speech API (Streaming)...[/blue]")
    try:
        location = getattr(config, "GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION"))
        client_options = {}
        if location and location.lower() != "global":
            client_options = {"api_endpoint": f"{location}-texttospeech.googleapis.com"}
        client = texttospeech.TextToSpeechClient(client_options=client_options)
        
        voice_clone_params = texttospeech.VoiceCloneParams(voice_cloning_key=voice_key)
        normalized_lang = (getattr(config, "GOOGLE_TTS_LANG", "en-GB")).replace('_', '-')
        lang_parts = normalized_lang.split('-')
        if len(lang_parts) == 2:
            normalized_lang = f"{lang_parts[0].lower()}-{lang_parts[1].upper()}"
        else:
            normalized_lang = normalized_lang.lower()

        # Configure streaming config
        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                language_code=normalized_lang, 
                voice_clone=voice_clone_params
            ),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            ),
        )
        config_request = texttospeech.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )

        def request_generator():
            yield config_request
            # Break text into chunks simulating streamed generation
            chunks = [c.strip() for c in re.split(r'(?<=[.!?]) +', text_to_synthesize) if c.strip()]
            if not chunks:
                chunks = [text_to_synthesize]
            
            for text in chunks:
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=text)
                )

        import io
        audio_buffer = io.BytesIO()
        streaming_responses = client.streaming_synthesize(request_generator())
        
        for response in streaming_responses:
            if response.audio_content:
                console.print(f"[dim]Received audio chunk: {len(response.audio_content)} bytes[/dim]")
                audio_buffer.write(response.audio_content)

        raw_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
        audio_data = raw_data.astype(np.float32) / 32768.0

        # 5. Playback check
        console.print("[green]🔊 Playback synthesized audio...[/green]")
        sd.play(audio_data, 24000)
        sd.wait()

        # 6. Save output file
        sf.write(OUTPUT_FILE, audio_data, 24000)
        console.print(f"[bold green]🎉 Saved successfully to: {OUTPUT_FILE}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Google Cloud TTS Synthesis failed: {e}[/bold red]")


if __name__ == "__main__":
    main()
