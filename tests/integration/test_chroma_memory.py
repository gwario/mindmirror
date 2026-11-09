"""Integration tests for ChromaMemoryManager retroactive tagging and SpeakerProfileStore."""

import asyncio
from pathlib import Path
import pytest

from mindmirror.memory.chroma_store import ChromaMemoryManager
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore


@pytest.mark.asyncio
async def test_chroma_memory_add_and_retroactive_tagging(tmp_path: Path):
    """Verifies storing memory with 'unresolved' speaker and updating metadata in place."""
    db_dir = tmp_path / "test_chroma_retro"
    mgr = ChromaMemoryManager(db_dir=db_dir, collection_name="test_retro_mem")

    # 1. Add memory with unresolved speaker
    mem_id = mgr.add_memory("Mario is a CS MSc software engineer specializing in AI", category="personal")
    assert mem_id != ""
    assert mem_id in mgr.get_unresolved_memory_ids()

    # Verify initial stored metadata
    initial = mgr.get_all_memories()
    assert len(initial) == 1
    assert initial[0]["id"] == mem_id
    assert initial[0]["speaker_id"] == "unresolved"

    # 2. Retroactively tag memory once speaker is identified
    updated_count = mgr.retroactive_tag_memories(speaker_id="mario")
    assert updated_count == 1
    assert len(mgr.get_unresolved_memory_ids()) == 0

    # 3. Verify metadata has updated in-place in ChromaDB
    updated = mgr.get_all_memories()
    assert len(updated) == 1
    assert updated[0]["id"] == mem_id
    assert updated[0]["speaker_id"] == "mario"

    # Verify direct collection metadata
    direct_res = mgr.collection.get(ids=[mem_id])
    assert direct_res["metadatas"][0]["speaker_id"] == "mario"
    assert direct_res["metadatas"][0]["is_retroactive"] is True


@pytest.mark.asyncio
async def test_chroma_memory_speaker_filtered_recall(tmp_path: Path):
    """Verifies vector recall filtering by speaker_id."""
    db_dir = tmp_path / "test_chroma_filter"
    mgr = ChromaMemoryManager(db_dir=db_dir, collection_name="test_filter_mem")

    mgr.add_memory("Mario loves espresso and dark chocolate", category="preference", speaker_id="mario")
    mgr.add_memory("Alice prefers Earl Grey tea with lemon", category="preference", speaker_id="alice")
    mgr.add_memory("System prompt guidelines for all participants", category="general", speaker_id="general")

    # Recall for Mario
    mario_recalled = mgr.recall_memories("beverage preferences", speaker_id="mario")
    assert len(mario_recalled) > 0
    facts = [m["fact"] for m in mario_recalled]
    speakers = [m["speaker_id"] for m in mario_recalled]
    assert any("espresso" in f for f in facts)
    assert "alice" not in speakers

    # Recall for Alice
    alice_recalled = mgr.recall_memories("tea drinks", speaker_id="alice")
    assert len(alice_recalled) > 0
    alice_facts = [m["fact"] for m in alice_recalled]
    alice_speakers = [m["speaker_id"] for m in alice_recalled]
    assert any("Earl Grey" in f for f in alice_facts)
    assert "mario" not in alice_speakers


@pytest.mark.asyncio
async def test_chroma_memory_auto_tag_on_bus_event(tmp_path: Path):
    """Verifies that ChromaMemoryManager auto-tags pending memories upon 'speaker.identified' bus event."""
    db_dir = tmp_path / "test_chroma_bus"
    mgr = ChromaMemoryManager(db_dir=db_dir, collection_name="test_bus_mem")
    bus = LocalBus()

    await mgr.attach_bus(bus)

    # Store 2 memories while speaker is resolving
    m1 = mgr.add_memory("Prefers British English spelling", category="preference")
    m2 = mgr.add_memory("Lives in London", category="personal")
    assert len(mgr.get_unresolved_memory_ids()) == 2

    # Emit speaker.identified event
    await bus.publish(
        "speaker.identified",
        PipelineMessage(
            channel="speaker.identified",
            phase=StreamPhase.CONTROL,
            payload={"speaker_id": "mario", "confidence": 0.95},
            speaker_id="mario",
        ),
    )

    # Allow event loop to process subscription
    for _ in range(20):
        if len(mgr.get_unresolved_memory_ids()) == 0:
            break
        await asyncio.sleep(0.02)

    assert len(mgr.get_unresolved_memory_ids()) == 0
    memories = mgr.get_all_memories()
    assert all(m["speaker_id"] == "mario" for m in memories)

    await mgr.close()
    await bus.close()


@pytest.mark.asyncio
async def test_speaker_profile_store_crud(tmp_path: Path):
    """Verifies SpeakerProfileStore persistence, retrieval, and updates."""
    db_dir = tmp_path / "test_profiles_crud"
    store = SpeakerProfileStore(db_dir=db_dir, collection_name="test_profiles")

    profile_mario = SpeakerProfile(
        speaker_id="mario",
        name="Mario",
        vad_silence_duration=0.9,
        tts_voice_id="en_GB-alan-medium",
        tts_pitch=1.05,
        tts_speaking_rate=1.0,
        ttt_reasoning_effort="high",
        prompt_context="CS MSc AI engineer; prefers lighthearted British tone",
        metadata={"interests": ["AI", "photography", "design"]},
    )

    # 1. Save profile
    store.save_profile(profile_mario)

    # 2. Retrieve profile
    retrieved = store.get_profile("mario")
    assert retrieved is not None
    assert retrieved.speaker_id == "mario"
    assert retrieved.name == "Mario"
    assert retrieved.vad_silence_duration == 0.9
    assert retrieved.tts_voice_id == "en_GB-alan-medium"
    assert retrieved.tts_pitch == 1.05
    assert retrieved.ttt_reasoning_effort == "high"
    assert retrieved.prompt_context == "CS MSc AI engineer; prefers lighthearted British tone"
    assert retrieved.metadata.get("interests") == ["AI", "photography", "design"]

    # 3. List profiles
    profiles = store.list_profiles()
    assert len(profiles) == 1
    assert profiles[0].speaker_id == "mario"

    # 4. Update profile
    profile_mario.vad_silence_duration = 1.2
    store.save_profile(profile_mario)
    updated = store.get_profile("mario")
    assert updated is not None
    assert updated.vad_silence_duration == 1.2

    # 5. Delete profile
    deleted = store.delete_profile("mario")
    assert deleted is True
    assert store.get_profile("mario") is None
    assert len(store.list_profiles()) == 0
