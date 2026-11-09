"""Unit tests for FileAudioSource and FileAudioSink (hardware-free audio testing)."""

import asyncio
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from mindmirror.audio.file_io import FileAudioSink, FileAudioSource


@pytest.mark.asyncio
async def test_file_audio_source_and_sink_roundtrip(tmp_path: Path):
    """Validates that audio files can be read, streamed into chunks, and saved to a WAV sink."""
    # 1. Generate a synthetic 44.1kHz stereo test WAV to verify resampling and mono downmixing
    input_wav = tmp_path / "test_input.wav"
    t = np.linspace(0, 0.2, int(44100 * 0.2), endpoint=False)
    sine = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    stereo = np.column_stack([sine, sine])
    sf.write(str(input_wav), stereo, 44100)

    # 2. Stream using FileAudioSource (target 16kHz mono, fast simulation)
    source = FileAudioSource(
        file_path=input_wav,
        target_sample_rate=16000,
        chunk_size=512,
        realtime=False,
        loop=False,
    )

    chunks = []
    async for chunk in source.stream_frames():
        chunks.append(chunk)

    assert len(chunks) > 0
    # Each chunk should be 512 samples * 2 bytes = 1024 bytes
    assert len(chunks[0]) == 1024

    # 3. Buffer through FileAudioSink and save to disk
    output_wav = tmp_path / "test_output.wav"
    sink = FileAudioSink(output_path=output_wav, sample_rate=16000)

    for chunk in chunks:
        sink.play_chunk(chunk)

    saved_path = sink.save()
    assert saved_path.exists()

    # 4. Verify saved WAV properties
    read_data, sr = sf.read(str(output_wav))
    assert sr == 16000
    assert len(read_data) > 0
    # Approximately 0.2 seconds * 16000 = ~3200 samples
    assert abs(len(read_data) - 3200) < 100
