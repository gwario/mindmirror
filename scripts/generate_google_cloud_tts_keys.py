"""
generate_google_cloud_tts_keys.py — Generate Instant Custom Voice Keys
======================================================================

Usage:
    cd /path/to/mindmirror
    PYTHONPATH=src \python scripts/generate_google_cloud_tts_keys.py

This script reads the recorded reference WAV files for each emotion, 
along with the mandatory consent statement, and generates a voice cloning 
key for each. It saves the resulting mapping to a JSON file.
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
import glob
import soundfile as sf
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mindmirror import config
from mindmirror.ui.ui import console

# ================= CONFIGURATION =================
VOICE = "ReferenceVoice"
GCTTS_BASE = "data/GoogleCloudTTS"
OUTPUT_BASE = os.path.join(GCTTS_BASE, VOICE)
WAVS_DIR = os.path.join(OUTPUT_BASE, "wavs")
KEYS_FILE = os.path.join(OUTPUT_BASE, "voice_cloning_keys.json")
CONSENT_FNAME = "consent_statement.wav"


def _get_gcp_credentials() -> tuple[str, str]:
    import google.auth
    from google.auth.transport.requests import Request
    try:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials, project_id = google.auth.default(scopes=scopes)
        credentials.refresh(Request())
        access_token = credentials.token
    except Exception as e:
        console.print(f"[bold red]❌ Failed to resolve GCP credentials: {e}[/bold red]")
        sys.exit(1)

    project_id = project_id or config.GOOGLE_CLOUD_PROJECT
    if not project_id:
        console.print("[bold red]❌ GCP Project ID could not be determined.[/bold red]")
        sys.exit(1)
    return access_token, project_id


def create_instant_custom_voice_key(access_token, project_id, ref_b64, consent_b64, lang):
    location = getattr(config, "GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION"))
    if location and location.lower() != "global":
        endpoint = f"https://{location}-texttospeech.googleapis.com/v1beta1/voices:generateVoiceCloningKey"
    else:
        endpoint = "https://texttospeech.googleapis.com/v1beta1/voices:generateVoiceCloningKey"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-goog-user-project": project_id,
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "reference_audio": {"audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000}, "content": ref_b64},
        "voice_talent_consent": {"audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000}, "content": consent_b64},
        "consent_script": "I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model.",
        "language_code": lang,
    }
    try:
        response = requests.post(endpoint, headers=headers, json=body)
        response.raise_for_status()
        return response.json().get("voiceCloningKey")
    except requests.exceptions.RequestException as e:
        last_error = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                last_error = f"{e} (Detail: {json.dumps(e.response.json())})"
            except:
                last_error = f"{e} (Body: {e.response.text})"
        console.print(f"[bold red]❌ API request failed: {last_error}[/bold red]")
        if "403" in str(last_error) or "permission" in str(last_error).lower():
            console.print("[yellow]💡 Note: Ensure your service account has the 'Text-to-Speech API Administrator' or 'Project Editor' role.[/yellow]")
        elif "allow" in str(last_error).lower() or "not found" in str(last_error).lower():
            console.print("[yellow]💡 Note: This restricted feature may require whitelisting your GCP project for Chirp 3 Instant Custom Voice.[/yellow]")
        return None


def main():
    console.print(f"[blue]🔍 Looking for recorded WAV files in {WAVS_DIR}...[/blue]")
    consent_path = os.path.join(WAVS_DIR, CONSENT_FNAME)
    
    if not os.path.exists(consent_path):
        console.print(f"[bold red]❌ Consent recording not found ({consent_path}). Please run scripts/record_google_cloud_tts.py first.[/bold red]")
        return

    ref_wavs = [os.path.basename(f) for f in glob.glob(os.path.join(WAVS_DIR, "*.wav")) if os.path.basename(f) != CONSENT_FNAME]
    if not ref_wavs:
        console.print("[bold red]❌ No reference emotion clips recorded yet.[/bold red]")
        return
        
    console.print(f"[green]✅ Found {len(ref_wavs)} reference clip(s).[/green]")

    generate = input("🔑 Generate Instant Custom Voice cloning keys now? [y/N]: ").strip().lower()
    if generate != "y":
        console.print("[blue]Aborted.[/blue]")
        return

    console.print("[blue]🔐 Resolving GCP credentials...[/blue]")
    access_token, project_id = _get_gcp_credentials()

    lang_code = (getattr(config, "GOOGLE_TTS_LANG", "en-GB")).replace('_', '-')
    
    try:
        consent_data, _ = sf.read(consent_path)
        consent_b64 = base64.b64encode((consent_data * 32768.0).astype(np.int16).tobytes()).decode("utf-8")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to read consent audio: {e}[/bold red]")
        return

    keys_map = {}
    success_count = 0

    for fname in ref_wavs:
        # e.g., en_Neutral.wav -> Neutral
        emotion = fname.split("_")[-1].replace(".wav", "")
        ref_path = os.path.join(WAVS_DIR, fname)
        
        console.print(f"\n[bold]🎙️  Processing emotion: {emotion}[/bold]")
        console.print(f"[dim]Reading {fname}...[/dim]")
        
        try:
            ref_data, _ = sf.read(ref_path)
            ref_b64 = base64.b64encode((ref_data * 32768.0).astype(np.int16).tobytes()).decode("utf-8")
        except Exception as e:
            console.print(f"[bold red]❌ Failed to read {fname}: {e}[/bold red]")
            continue

        console.print("[blue]📡 Calling Instant Custom Voice API...[/blue]")
        key = create_instant_custom_voice_key(access_token, project_id, ref_b64, consent_b64, lang_code)
        
        if key:
            console.print(f"[green]✅ Key generated for {emotion}[/green]")
            keys_map[emotion] = key
            success_count += 1
        else:
            console.print(f"[bold red]❌ Failed to generate key for {emotion}[/bold red]")

    if keys_map:
        with open(KEYS_FILE, "w") as f:
            json.dump(keys_map, f, indent=2)
        console.print(f"\n[bold green]🎉 Saved {success_count} keys to: {KEYS_FILE}[/bold green]")
        console.print("[blue]You can now run scripts/test_google_cloud_tts.py to synthesize speech.[/blue]")
    else:
        console.print("\n[bold red]⚠️  No keys were successfully generated.[/bold red]")


if __name__ == "__main__":
    main()
