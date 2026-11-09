"""TTS coordination utilities.

File-based locks (speaking.lock, playback.lock) are superseded by PlaybackState
and message bus control topics. This module delegates to PlaybackState while
cleaning up any legacy lock files left on disk.
"""

import os
from mindmirror.audio.playback import PlaybackState
from mindmirror.config import LOCK_FILE, PLAYBACK_LOCK


def set_speaking_lock(active: bool) -> None:
    """Updates mic mute state via PlaybackState and cross-process lock file."""
    try:
        state = PlaybackState.get_instance()
        state._mic_muted = active

        if active:
            with open(LOCK_FILE, "w") as f:
                f.write("active")
        else:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
    except Exception:
        pass


def set_playback_lock(active: bool) -> None:
    """Updates playback state via PlaybackState and cross-process lock file."""
    try:
        state = PlaybackState.get_instance()
        state._playback_active = active

        if active:
            with open(PLAYBACK_LOCK, "w") as f:
                f.write("1")
        else:
            if os.path.exists(PLAYBACK_LOCK):
                os.remove(PLAYBACK_LOCK)
    except Exception:
        pass
