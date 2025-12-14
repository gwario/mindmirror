from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

models = list(client.models.list())

print(f"Available models ({len(models)}):")
for m in models:
    if 'generateContent' in m.supported_actions:
        print(f"{m.display_name}({m.name})")

decision = input(f"Show model details? [y/N]")
if decision.strip().lower() == 'y':
    for m in models:
        if 'generateContent' in m.supported_actions:
            print(f"\n{m.name}")
            print(f"  {m.display_name}")

            # Print all attributes
            for attr in dir(m):
                if not attr.startswith('_') and not attr in ["model_fields", "model_config", "model_computed_fields", "model_fields_set"]:
                    try:
                        value = getattr(m, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass