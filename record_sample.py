import os
import sys

import sounddevice as sd
import soundfile as sf
import yaml
from rich.markdown import Markdown
from rich.panel import Panel

import sound
from ui import console

# ================= CONFIGURATION =================
VOICE="MyVoice"
OUTPUT_BASE = f"data/{VOICE}"
METADATA_FILE = os.path.join(OUTPUT_BASE, "metadata.csv")
WAVS_DIR = os.path.join(OUTPUT_BASE, "wavs")
SCRIPT_FILE = "script.yaml"


def load_script(yaml_path):
    if not os.path.exists(yaml_path):
        console.print(f"[bold red]❌ Error: {yaml_path} not found![/bold red]")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup Folders
    if not os.path.exists(WAVS_DIR):
        os.makedirs(WAVS_DIR)

    # 2. Load Script
    script_data = load_script(SCRIPT_FILE)
    playlist = []
    for section in script_data:
        cat_code = section['category'].split(".")[0].strip().replace(" ", "")
        for idx, line in enumerate(section['lines']):
            fname = f"{section['lang']}_{cat_code}_{idx + 1:02d}.wav"
            playlist.append({
                "fname": fname, "text": line, "cat": section['category'],
                "act": section['acting'], "lang": section['lang']
            })

    # 3. Hardware Setup (Via Engine)
    device_id, _ = sound.select_audio_device()
    native_sr = sound.get_valid_samplerate(device_id)

    console.print(f"[green]✅ Using Device: {device_id} | Rate: {native_sr}Hz[/green]")

    # 4. Calibration (Via Engine)
    thresh, noise_profile = sound.calibrate_noise_floor(device_id, native_sr, console)

    # 5. Main Recording Loop
    existing_files = set(os.listdir(WAVS_DIR))
    f_meta = open(METADATA_FILE, "a", encoding="utf-8")

    for i, item in enumerate(playlist):
        if item['fname'] in existing_files:
            continue

        # UI Display
        md = f"# {item['cat']}\n**Acting:** *{item['act']}*\n\n# 🗣️ \"{item['text']}\""
        console.print(Panel(Markdown(md), title=f"Scene {i + 1}/{len(playlist)}", border_style="cyan"))

        while True:
            # Record (Passes control to audio sound)
            audio = sound.record_clip(device_id, native_sr, thresh, noise_profile, console)

            if audio is None:
                console.print("[yellow]No audio detected. Try again.[/yellow]")
                continue

            console.print("[blue]▶️ Check playback...[/blue]")
            sd.play(audio, sound.TARGET_SR)
            sd.wait()

            choice = input("💾 [Enter] Save | [r]etry | [s]kip: ").lower()
            if choice == 's': break
            if choice == 'r': continue

            # Save
            path = os.path.join(WAVS_DIR, item['fname'])
            sf.write(path, audio, sound.TARGET_SR)

            # Write Metadata
            # Note: F5-TTS usually expects relative paths like 'wavs/file.wav'
            f_meta.write(f"wavs/{item['fname']}|{item['text']}\n")
            f_meta.flush()
            console.print(f"[green]✅ Saved {item['fname']}[/green]")
            break

    f_meta.close()
    console.print("[bold green]🎉 Dataset Recording Complete![/bold green]")


if __name__ == "__main__":
    main()
