"""SpeakerProfile model and persistent ChromaDB store for per-speaker runtime tuning."""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Runtime tuning parameters and personalization profile for an identified speaker."""

    speaker_id: str
    name: str = ""
    vad_silence_duration: float = 0.5       # Custom pause threshold (e.g. 1.2s for slow speakers vs 0.5s for fast)
    tts_voice_id: Optional[str] = None      # Specific voice model / voice ID override
    tts_pitch: float = 1.0                  # Voice pitch multiplier (1.0 = normal)
    tts_speaking_rate: float = 1.0          # Voice speaking rate (1.0 = normal)
    ttt_reasoning_effort: str = "high"      # Reasoning effort level ("high" vs "low")
    prompt_context: str = ""                # Personalized user bio / instructions
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Converts profile into flat ChromaDB metadata dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "name": self.name or self.speaker_id,
            "vad_silence_duration": float(self.vad_silence_duration),
            "tts_voice_id": self.tts_voice_id or "",
            "tts_pitch": float(self.tts_pitch),
            "tts_speaking_rate": float(self.tts_speaking_rate),
            "ttt_reasoning_effort": self.ttt_reasoning_effort,
            "prompt_context": self.prompt_context,
            "extra_json": json.dumps(self.metadata),
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> "SpeakerProfile":
        """Reconstructs SpeakerProfile from ChromaDB metadata dictionary."""
        extra_json = meta.get("extra_json", "{}")
        try:
            extra = json.loads(extra_json) if isinstance(extra_json, str) else {}
        except Exception:
            extra = {}

        return cls(
            speaker_id=str(meta.get("speaker_id", "")),
            name=str(meta.get("name", "")),
            vad_silence_duration=float(meta.get("vad_silence_duration", 0.5)),
            tts_voice_id=meta.get("tts_voice_id") or None,
            tts_pitch=float(meta.get("tts_pitch", 1.0)),
            tts_speaking_rate=float(meta.get("tts_speaking_rate", 1.0)),
            ttt_reasoning_effort=str(meta.get("ttt_reasoning_effort", "high")),
            prompt_context=str(meta.get("prompt_context", "")),
            metadata=extra,
        )


class SpeakerProfileStore:
    """ChromaDB-backed persistent store for SpeakerProfile configurations."""

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        collection_name: str = "speaker_profiles",
        client: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        self.collection_name = collection_name
        from mindmirror.memory.chroma_client import create_chroma_client
        self.client = create_chroma_client(db_dir=db_dir, client=client)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Speaker profile runtime tuning store"},
        )

    def save_profile(self, profile: SpeakerProfile) -> None:
        """Saves or updates a speaker profile in the store."""
        summary = f"Speaker profile for {profile.name or profile.speaker_id}: VAD pause {profile.vad_silence_duration}s"
        self.collection.upsert(
            ids=[profile.speaker_id],
            documents=[summary],
            metadatas=[profile.to_metadata()],
        )
        logger.info("Saved profile for speaker '%s'.", profile.speaker_id)

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Retrieves a speaker profile by speaker_id."""
        if not speaker_id:
            return None
        res = self.collection.get(ids=[speaker_id])
        if res and res.get("metadatas") and len(res["metadatas"]) > 0:
            return SpeakerProfile.from_metadata(res["metadatas"][0])
        return None

    def list_profiles(self) -> list[SpeakerProfile]:
        """Lists all registered speaker profiles."""
        if self.collection.count() == 0:
            return []
        res = self.collection.get()
        profiles = []
        if res and res.get("metadatas"):
            for meta in res["metadatas"]:
                profiles.append(SpeakerProfile.from_metadata(meta))
        return profiles

    def delete_profile(self, speaker_id: str) -> bool:
        """Deletes a profile by speaker_id."""
        try:
            self.collection.delete(ids=[speaker_id])
            logger.info("Deleted profile for speaker '%s'.", speaker_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete profile '%s': %s", speaker_id, e)
            return False
