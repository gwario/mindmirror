def create_volume_meter(volume, threshold, width=50, is_recording=False):
    """Create a visual volume meter for terminal display"""
    bar_len = int(min(volume, 0.5) * width)
    bar = "█" * bar_len
    spaces = " " * (width - bar_len)

    status = "REC    " if is_recording else "WAITING"
    color = "\033[91m" if is_recording else "\033[93m"

    return f"   [{color}{status}\033[0m] Level: |{color}{bar}{spaces}\033[0m| {volume:.3f}"

def create_volume_meter_rich(volume, noise_floor, silence_thresh, speech_thresh, width=40):
    """Create a visual volume meter for Rich console - SINGLE LINE ONLY"""
    normalized = min(volume, 1.0)
    filled = int(normalized * width)

    bar = "█" * filled + "░" * (width - filled)

    # Color and status
    if volume > speech_thresh:
        color = "green"
        status = "🎤 SPEAKING"
    elif volume > silence_thresh:
        color = "yellow"
        status = "🔊 SOUND"
    else:
        color = "dim"
        status = "🔇 SILENT"

    return f"[{color}]{bar}[/{color}] {volume:.3f} | {status}"
