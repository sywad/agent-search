"""Independent Gemini API latency test."""
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

MODELS = [
    ('gemini-2.0-flash', None),
    ('gemini-2.5-flash', {'thinkingBudget': 0}),
    ('gemini-2.5-flash', None),  # default thinking
    ('gemini-3-flash-preview', {'thinkingLevel': 'LOW'}),
]

PROMPT = "Say hello in one sentence."

print(f"Testing Gemini API latency with prompt: \"{PROMPT}\"")
print(f"{'='*60}")

for model_id, thinking_config in MODELS:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={API_KEY}"
    gen_config = {"temperature": 0}
    if thinking_config:
        gen_config["thinkingConfig"] = thinking_config
    body = {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": gen_config
    }

    label = model_id
    if thinking_config:
        label += f" ({thinking_config})"

    t0 = time.time()
    try:
        resp = requests.post(url, json=body, timeout=120)
        latency = round(time.time() - t0, 2)
        data = resp.json()

        if 'error' in data:
            print(f"{label}: ERROR in {latency}s — {data['error']['message'][:100]}")
            continue

        text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'NO TEXT')[:100]
        usage = data.get('usageMetadata', {})
        tokens = f"prompt={usage.get('promptTokenCount', '?')} output={usage.get('candidatesTokenCount', '?')} thinking={usage.get('thoughtsTokenCount', 0)}"
        print(f"{label}: {latency}s — {tokens} — \"{text}\"")

    except requests.exceptions.Timeout:
        print(f"{label}: TIMEOUT after 120s")
    except Exception as e:
        print(f"{label}: ERROR after {round(time.time() - t0, 2)}s — {e}")

print(f"{'='*60}")
print("Done.")
