import os
from mindmirror.config import LOCK_FILE, PLAYBACK_LOCK


def set_speaking_lock(active: bool) -> None:
    """Creates or removes the speaking lock file to mute the mic during TTS playback."""
    try:
        if active:
            with open(LOCK_FILE, "w") as f:
                f.write("active")
        else:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
    except Exception:
        pass


def set_playback_lock(active: bool) -> None:
    """Creates or removes the playback lock file to signal active audio output."""
    try:
        if active:
            with open(PLAYBACK_LOCK, "w") as f:
                f.write("1")
        else:
            if os.path.exists(PLAYBACK_LOCK):
                os.remove(PLAYBACK_LOCK)
    except Exception:
        pass
