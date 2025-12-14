import re
import time

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions

SYSTEM_PROMPT = """
You are a voice assistant. 
When you reply, strictly prepend a style tag to your message. 
Available tags: [NEUTRAL], [EXCITED], [SERIOUS], [LAZY].

Rules:
1. Always start with the tag.
2. [EXCITED] for success, good news, or high energy.
3. [SERIOUS] for errors, warnings, or bad news.
4. [LAZY] for casual confirmation or when unsure.
5. [NEUTRAL] for general information.

Example: '[EXCITED] I found the file you were looking for!'
"""

load_dotenv()

client = genai.Client()

def strip_markdown(text):
    # Your regex stripping function
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^]]+)]\([^)]+\)', r'\1', text)
    return text.strip()

import re

def parse_llm_response(response_text):
    """
    Parses the LLM output to separate the style tag from the spoken text.

    Args:
        response_text (str): Raw output like "[EXCITED] I found the file!"

    Returns:
        tuple: (style_string, clean_text_string)
               e.g. ("excited", "I found the file!")
    """

    # Regex breakdown:
    # ^\s* -> Start of line, ignore leading whitespace
    # \[        -> Look for opening bracket '['
    # (...)     -> Capture the word inside (NEUTRAL, EXCITED, etc.)
    # \]        -> Look for closing bracket ']'
    # \s* -> Ignore any whitespace after the tag
    # (.*)      -> Capture everything else as the message content
    pattern = r"^\s*\[(NEUTRAL|EXCITED|SERIOUS|LAZY)\]\s*(.*)"

    # re.DOTALL allows the text to contain newlines
    # re.IGNORECASE allows [Excited] or [excited] to work too
    match = re.match(pattern, response_text, re.IGNORECASE | re.DOTALL)

    if match:
        style = match.group(1).lower()      # Extract style (e.g., "excited")
        clean_text = match.group(2).strip() # Extract message
        return style, clean_text
    else:
        # Fallback: If model forgets the tag, treat as neutral
        return response_text.strip(), "neutral"

def conversation_ttt_task(model_name, log_queue, text_queue, response_queue):
    """Process text from STT and send to Gemini"""
    chat = client.chats.create(
        model=model_name,
        config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        history=[]
    )

    log_queue.put({'type': 'info', 'text': "Gemini ready, waiting for input..."})

    last_request_time = 0
    min_interval = 5.0  # 5 seconds between requests to be safe

    while True:
        text = text_queue.get()

        if not text.strip():
            continue

        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            log_queue.put({'type': 'status', 'text': f"⏱️  Rate limiting: waiting {wait_time:.1f}s..."})
            time.sleep(wait_time)

        # Retry logic
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = chat.send_message(text)
                log_queue.put({'type': 'ai', 'text': response.text})
                clean_text = strip_markdown(response.text)
                clean_text, emotion = parse_llm_response(clean_text)
                response_queue.put((clean_text, emotion,))
                last_request_time = time.time()
                break  # Success, exit retry loop

            except exceptions.ResourceExhausted as e:
                retry_count += 1
                wait_time = 60 * retry_count  # Exponential backoff
                log_queue.put({'type': 'status', 'text': f"❌ Rate limit hit! Retry {retry_count}/{max_retries}"})
                log_queue.put({'type': 'status', 'text': f"⏱️  Waiting {wait_time}s before retrying..."})
                time.sleep(wait_time)

            except Exception as e:
                log_queue.put({'type': 'status', 'text': f"❌ Error with Gemini: {e}"})
                log_queue.put({'type': 'status', 'text': "Skipping this message..."})
                break

        if retry_count >= max_retries:
            log_queue.put({'type': 'status', 'text': f"❌ Failed after {max_retries} retries, skipping message"})
