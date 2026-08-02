"""
record_google_cloud_tts.py — Chirp 3: Instant Custom Voice recorder
=====================================================================
Docs: https://cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice

OVERVIEW
--------
Google's Chirp 3 Instant Custom Voice API lets you clone a voice in real time
using only two short audio clips — no training required. This script handles:
  1. Recording the mandatory consent statement clip.
  2. Recording ~10s expressive reference audio clips.
  3. (Optional) Submitting both to the API to generate a Voice Cloning Key.

The recording script and texts are loaded from:
    data/GoogleTTS_ReferenceVoice/script.yaml

HOW TO RUN
----------
    cd /path/to/mindmirror
    PYTHONPATH=src python3 scripts/record_google_cloud_tts.py

CONSENT STATEMENT (required, read verbatim)
-------------------------------------------
    "I am the owner of this voice and I consent to Google using this voice
     to create a synthetic voice model."

  - Must be read in the same language as the voice clone target.
  - Must be recorded as a single-channel audio file (LINEAR16, PCM, MP3, or M4A).
  - Must be up to 10 seconds long.
  - You cannot substitute a custom consent script — the above wording is
    mandatory per Google's terms.

REFERENCE AUDIO GUIDELINES (from Google docs)
---------------------------------------------
  - Aim for as close to 10 seconds as possible.
  - Use a single-channel audio file (mono). Any sample rate is accepted.
  - Record in the same environment as the consent statement (same mic, same room).
  - Minimal background noise — the model replicates microphone quality.
  - Be expressive and dynamic — a monotone reading produces a monotone clone.
  - Include natural pauses — the cloned voice mirrors your cadence.
  - Aim for a slightly more excited/energetic delivery than the intended output.

OUTPUT
------
  data/GoogleCloudTTS/
  ├── script.yaml                      ← recording script (shared, like f5-tts)
  └── ReferenceVoice/                  ← voice subdir (customisable via VOICE)
      ├── wavs/
      │   ├── consent_statement.wav    ← mandatory consent clip (consent: true)
      │   └── en_<Category>_<N>.wav    ← reference audio clips
      └── voice_cloning_key.txt        ← API-returned cloning key (if generated)

The voice_cloning_key.txt value is what you pass to the GoogleCloudTTS
`voice_clone.voice_cloning_key` field in synthesis requests.
"""
import os
import sys
import json
import base64
import subprocess
from pathlib import Path

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Importing config triggers load_dotenv() and sets GOOGLE_APPLICATION_CREDENTIALS
# in the environment — exactly as the GoogleCloudTTS module relies on.
from mindmirror import audio, config
from mindmirror.ui.ui import console

# ================= CONFIGURATION =================
VOICE = "ReferenceVoice"
GCTTS_BASE = "data/GoogleCloudTTS"
OUTPUT_BASE = os.path.join(GCTTS_BASE, VOICE)
WAVS_DIR = os.path.join(OUTPUT_BASE, "wavs")
SCRIPT_FILE = os.path.join(GCTTS_BASE, "script.yaml")
CONSENT_FNAME = "consent_statement.wav"





def load_script(yaml_path: str) -> list[dict]:
    if not os.path.exists(yaml_path):
        console.print(f"[bold red]❌ Script not found: {yaml_path}[/bold red]")
        sys.exit(1)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup output folder
    os.makedirs(WAVS_DIR, exist_ok=True)

    # 2. Load script and build playlist
    script_data = load_script(SCRIPT_FILE)
    playlist = []
    consent_fname = None

    for section in script_data:
        is_consent = section.get("consent", False)
        cat_code = section["category"].replace(" ", "")
        for idx, line in enumerate(section["lines"]):
            if is_consent:
                # Consent recording always gets the fixed filename
                fname = CONSENT_FNAME
                consent_fname = CONSENT_FNAME
            else:
                fname = f"{section['lang']}_{cat_code}.wav"
            playlist.append({
                "fname": fname,
                "text": line,
                "cat": section["category"],
                "act": section["acting"],
                "lang": section["lang"],
                "consent": is_consent,
            })

    # 3. Hardware setup
    device_id, _ = audio.select_audio_device()
    native_sr = audio.get_valid_samplerate(device_id)
    console.print(f"[green]✅ Using Device: {device_id} | Rate: {native_sr}Hz[/green]")

    # 4. Noise calibration
    thresh, noise_profile = audio.calibrate_noise_floor(device_id, native_sr, console)

    # 5. Main recording loop
    existing_files = set(os.listdir(WAVS_DIR))

    for i, item in enumerate(playlist):
        if item["fname"] in existing_files:
            console.print(f"[dim]⏭  Skipping {item['fname']} (already recorded)[/dim]")
            continue

        border = "magenta" if item["consent"] else "cyan"
        tag = " *(CONSENT — read verbatim)*" if item["consent"] else ""
        md = f"# {item['cat']}{tag}\n**Acting:** *{item['act']}*\n\n# 🗣️ \"{item['text']}\""
        console.print(Panel(Markdown(md), title=f"Scene {i + 1}/{len(playlist)}", border_style=border))

        while True:
            audio_clip = audio.record_clip(device_id, native_sr, thresh, noise_profile, console)

            if audio_clip is None:
                console.print("[yellow]No audio detected. Try again.[/yellow]")
                continue

            console.print("[blue]▶️  Playback check...[/blue]")
            sd.play(audio_clip, config.TARGET_SR)
            sd.wait()

            choice = input("💾 [Enter] Save | [r]etry | [s]kip: ").strip().lower()
            if choice == "s":
                break
            if choice == "r":
                continue

            path = os.path.join(WAVS_DIR, item["fname"])
            sf.write(path, audio_clip, config.TARGET_SR, subtype='PCM_16')
            console.print(f"[green]✅ Saved {item['fname']}[/green]")
            existing_files.add(item["fname"])
            break

    console.print("[bold green]🎉 Recording session complete![/bold green]")

    console.print("\n[blue]Done. Run scripts/test_google_cloud_tts.py to generate your voice key and test it![/blue]")


if __name__ == "__main__":
    main()
