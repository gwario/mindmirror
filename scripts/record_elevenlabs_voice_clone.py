import os
import tempfile
import time
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv, set_key
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

def record_audio(duration: float, samplerate: int = 44100, channels: int = 1) -> np.ndarray:
    print(f"Recording for {duration} seconds...")
    print("Speak clearly into your microphone.")
    
    # Record audio
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
    
    # Print a simple progress bar
    for i in range(int(duration)):
        time.sleep(1)
        print(f"[{i+1}/{int(duration)}] seconds recorded...")
        
    sd.wait()
    print("Recording finished!")
    return recording, samplerate

def clone_voice(api_key: str, recording: np.ndarray, samplerate: int, voice_name: str, description: str):
    print("Preparing audio data for ElevenLabs...")
    
    # Write to a temporary WAV file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        sf.write(temp_path, recording, samplerate, format='WAV', subtype='PCM_16')
        
        print("Uploading to ElevenLabs to create Instant Voice Clone...")
        client = ElevenLabs(api_key=api_key)
        
        # Add voice using the file
        with open(temp_path, 'rb') as f:
            voice = client.voices.add(
                name=voice_name,
                description=description,
                files=[f]
            )
            
        return voice.voice_id
    except Exception as e:
        print(f"Error creating voice clone: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Record your voice and create an ElevenLabs voice clone.")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration to record in seconds (default: 30s)")
    parser.add_argument("--name", type=str, default="MindMirror Clone", help="Name of the new voice")
    parser.add_argument("--description", type=str, default="Cloned voice for MindMirror", help="Description for the voice")
    args = parser.parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return

    # Record audio
    recording, samplerate = record_audio(args.duration)
    
    # Playback for verification
    print("Playing back recording for verification...")
    sd.play(recording, samplerate)
    sd.wait()
    
    confirm = input("Do you want to use this recording to create a clone? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Create voice clone
    voice_id = clone_voice(api_key, recording, samplerate, args.name, args.description)
    
    if voice_id:
        print(f"\n✅ Successfully created voice clone!")
        print(f"Voice ID: {voice_id}")
        
        # Optionally update .env file automatically
        update = input("Do you want to automatically update your .env file with this Voice ID? (y/n): ")
        if update.lower() == 'y':
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
            if os.path.exists(env_path):
                set_key(env_path, "ELEVENLABS_VOICE_ID", voice_id)
                print(f"Updated ELEVENLABS_VOICE_ID in {env_path}")
            else:
                print("Could not find .env file to update.")
        
        print("\nMake sure your configuration is set to use ElevenLabsTTS in main.py.")
    
if __name__ == "__main__":
    main()
