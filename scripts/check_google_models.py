from dotenv import load_dotenv
from google import genai
import os
import json
import sys

# Ensure src path is in sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mindmirror import config

load_dotenv()

# Get Google Cloud credentials
key_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
project_id = None
if key_path and os.path.exists(key_path):
    try:
        with open(key_path, "r") as f:
            key_data = json.load(f)
            project_id = key_data.get("project_id")
    except Exception:
        pass

project = project_id or getattr(config, 'GOOGLE_CLOUD_PROJECT', None)
location = getattr(config, 'GOOGLE_CLOUD_LOCATION', None)

print("=========================================================")
print(f"Vertex AI Model Checker")
print(f"Project: {project}")
print(f"Location: {location}")
print("=========================================================\n")

client = genai.Client(
    vertexai=True,
    project=project,
    location=location
)

# Check Speech-to-Text (STT) V2 Configuration
print("=========================================================")
print("Speech-to-Text (STT) V2 Configuration Check")
print("=========================================================")
stt_model = getattr(config, 'GOOGLE_STT_MODEL', None)
stt_lang = getattr(config, 'GOOGLE_STT_LANG', None)

if stt_model and stt_lang:
    print(f"Configured STT Model: {stt_model}")
    print(f"Configured STT Language: {stt_lang}")
    print("🔄 Testing Speech V2 Recognizer setup... ", end="", flush=True)
    try:
        from mindmirror.stt.google import GoogleCloudSTT
        stt = GoogleCloudSTT(
            language_code=stt_lang,
            model=stt_model,
            location=location,
            project_id=project,
            log_queue=None
        )
        stt.load_model()
        print("✅ [SUCCESSFUL] Recognizer config is valid and ready.")
    except Exception as e:
        print(f"❌ [FAILED: {e}]")
else:
    print("  (STT Configuration not found in config/env)")
print("=========================================================\n")

try:
    models = list(client.models.list())
except Exception as e:
    print(f"❌ Failed to list models from Vertex AI: {e}")
    sys.exit(1)

# Filter for standard conversational Gemini chat models and TTS voice models
conversational_candidates = []
tts_candidates = []

for m in models:
    name = m.name
    if name.startswith("publishers/google/models/"):
        name = name.split("/")[-1]
    
    is_chat = (
        "gemini-" in name.lower()
        and not any(x in name.lower() for x in ["embedding", "image", "tts", "computer-use", "live-", "clip"])
    )
    is_tts = (
        "gemini-" in name.lower()
        and "tts" in name.lower()
    )
    
    if is_chat:
        conversational_candidates.append(name)
    elif is_tts:
        tts_candidates.append(name)

# Sort them alphabetically
conversational_candidates = sorted(list(set(conversational_candidates)))
tts_candidates = sorted(list(set(tts_candidates)))

print(f"Found {len(conversational_candidates)} chat models and {len(tts_candidates)} TTS models in catalog.")
print("Testing API accessibility for each...\n")

accessible_chats = []
accessible_tts = []

print("--- Conversational Chat Models ---")
for model_name in conversational_candidates:
    print(f"🔄 Checking {model_name}... ", end="", flush=True)
    try:
        client.models.generate_content(
            model=model_name,
            contents="Hi"
        )
        print("✅ [ACCESSIBLE]")
        accessible_chats.append(model_name)
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "not found" in err_msg.lower():
            print("❌ [NOT ACCESSIBLE / 404]")
        elif "permission" in err_msg.lower() or "403" in err_msg:
            print("❌ [PERMISSION DENIED / 403]")
        else:
            print(f"❌ [FAILED: {err_msg[:60]}...]")

print("\n--- Text-to-Speech (TTS) Models ---")
for model_name in tts_candidates:
    print(f"🔄 Checking {model_name}... ", end="", flush=True)
    try:
        # Generative TTS models might throw error on standard text generation but we check for accessibility
        client.models.generate_content(
            model=model_name,
            contents="Hi"
        )
        print("✅ [ACCESSIBLE]")
        accessible_tts.append(model_name)
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "not found" in err_msg.lower():
            print("❌ [NOT ACCESSIBLE / 404]")
        elif "permission" in err_msg.lower() or "403" in err_msg:
            print("❌ [PERMISSION DENIED / 403]")
        else:
            # If it's a signature/unsupported action error, it's still accessible (i.e. model exists and didn't 404/403)
            print("✅ [ACCESSIBLE]")
            accessible_tts.append(model_name)

print("\n=========================================================")
print("Summary of Accessible Conversational Models:")
print("=========================================================")
if accessible_chats:
    for m in accessible_chats:
        print(f"  ⭐ {m}")
else:
    print("  (None found)")

print("\n=========================================================")
print("Summary of Accessible Text-to-Speech Models:")
print("=========================================================")
if accessible_tts:
    for m in accessible_tts:
        print(f"  ⭐ {m}")
else:
    print("  (None found)")
print("=========================================================")