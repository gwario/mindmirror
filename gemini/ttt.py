import re
import time

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions

SYSTEM_PROMPT = """
You are an individual named Mario.
Your job is to develop software. 
When you reply, strictly prepend a style tag to your message. 
Available tags: [NEUTRAL], [EXCITED], [SERIOUS], [LAZY].

Rules:
1. Always start with the tag.
2. [EXCITED] for success, good news, or high energy.
3. [SERIOUS] for errors, warnings, or bad news.
4. [LAZY] for casual confirmation or when unsure.
5. [NEUTRAL] for general information.

Example: '[EXCITED] I found the file you were looking for!'

If you use multiple styles within your answer, then keep the scope of the styled section short, especially for [EXCITED]. 
"""

load_dotenv()

client = genai.Client()

def strip_markdown(text):
    """Remove markdown formatting but preserve all text content including code"""

    # Code blocks - remove backticks and language identifier but keep the code
    text = re.sub(r'```[\w]*\n?', '', text)  # Remove opening ```python or ```
    text = re.sub(r'```', '', text)           # Remove closing ```

    # Inline code - remove backticks but keep content
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Bold - remove ** but keep text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # Italic - remove * but keep text
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Headers - remove # but keep text
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Links - keep link text only
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Lists - remove bullets/numbers but keep text
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    return text.strip()

def parse_llm_response(response_text):
    """
    Parses the LLM output to separate style tags from spoken text.
    Handles multiple style tags in one response.

    Args:
        response_text (str): Raw output like "[EXCITED] I found it! [NEUTRAL] Here's the info."

    Returns:
        list of tuples: [(style_string, clean_text_string), ...]
                       e.g. [("excited", "I found it!"), ("neutral", "Here's the info.")]
    """

    # Pattern to match [TAG] followed by text until the next [TAG] or end of string
    # (?=\[(?:NEUTRAL|EXCITED|SERIOUS|LAZY)\]|$) is a lookahead for next tag or end
    pattern = r'\[(NEUTRAL|EXCITED|SERIOUS|LAZY)\]\s*((?:(?!\[(?:NEUTRAL|EXCITED|SERIOUS|LAZY)\]).)+)'

    matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)

    if matches:
        # Return list of (style, text) tuples
        result = [(style.lower(), text.strip()) for style, text in matches]
        return result
    else:
        # Fallback: If model forgets tags, treat entire response as neutral
        return [("neutral", response_text.strip())]

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

                # Parse response - now returns a list of (style, text) tuples
                segments = parse_llm_response(response.text)

                # Strip markdown from each segment and send to response queue
                for style, segment_text in segments:
                    clean_text = strip_markdown(segment_text)
                    response_queue.put((style, clean_text))

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