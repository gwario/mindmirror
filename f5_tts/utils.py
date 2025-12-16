import os
import re
from .config import LOCK_FILE, MIN_CHUNK_LENGTH



def split_into_sentences(text):
    """
    Splits text into sentences but merges short ones to maintain flow.

    Logic:
    1. "Okay." (Too short) -> Buffer: "Okay."
    2. "I found the file." (Long enough) -> Buffer: "Okay. I found the file." -> Release!
    """
    # 1. Initial rough split by punctuation
    # (?<=[.?!]) keeps the punctuation attached to the sentence
    raw_sentences = re.split(r'(?<=[.?!])\s+', text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    final_chunks = []
    current_buffer = ""

    for sentence in raw_sentences:
        # Add space if buffer is not empty
        if current_buffer:
            current_buffer += " " + sentence
        else:
            current_buffer = sentence

        # 2. Check if buffer is "substantial" enough to speak
        # We check character length (fastest) or word count
        if len(current_buffer) >= MIN_CHUNK_LENGTH:
            final_chunks.append(current_buffer)
            current_buffer = ""

    # 3. Handle the leftovers (The last few words)
    if current_buffer:
        if final_chunks:
            # If we have previous chunks, append the leftover to the last one
            # to avoid generating a tiny 2-word audio at the very end.
            final_chunks[-1] += " " + current_buffer
        else:
            # If the entire text was short (e.g. "Yes."), just take it.
            final_chunks.append(current_buffer)

    return final_chunks

def set_speaking_lock(active: bool):
    """Creates or removes the lock file to mute the mic."""
    try:
        if active:
            with open(LOCK_FILE, "w") as f: f.write("active")
        else:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
    except Exception:
        pass