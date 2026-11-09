"""Parallel Background Speaker Identifier and Voice Embedding Registry."""

import asyncio
import logging
from pathlib import Path
import time
from typing import Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np

from mindmirror.pipeline.component import PipelineComponent
from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus
from mindmirror.speaker.context import SpeakerContext
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore

logger = logging.getLogger(__name__)


class AcousticVoiceEmbedder:
    """Computes deterministic 128-dimensional acoustic voice embeddings from raw audio.

    Uses MFCCs, delta features, and spectral moments with L2 normalization.
    """

    def __init__(self, sample_rate: int = 16000, embedding_dim: int = 128) -> None:
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim

    def extract_embedding(self, audio_data: bytes | np.ndarray) -> list[float]:
        """Extracts normalized voice embedding vector from raw PCM bytes or float array."""
        try:
            if isinstance(audio_data, bytes):
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype in (np.float32, np.float64):
                    samples = audio_data.astype(np.float32)
                else:
                    samples = audio_data.astype(np.float32) / 32768.0
            else:
                return [0.0] * self.embedding_dim

            if len(samples) < 512:
                # Pad short samples with zero
                samples = np.pad(samples, (0, 512 - len(samples)))

            import librosa

            # Compute acoustic feature matrix
            mfcc = librosa.feature.mfcc(y=samples, sr=self.sample_rate, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            centroid = librosa.feature.spectral_centroid(y=samples, sr=self.sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=samples, sr=self.sample_rate)

            features = np.vstack([mfcc, delta, centroid, rolloff])
            mean = np.mean(features, axis=1)
            std = np.std(features, axis=1)
            p25 = np.percentile(features, 25, axis=1)
            p75 = np.percentile(features, 75, axis=1)

            vec = np.concatenate([mean, std, p25, p75])
            vec = vec[: self.embedding_dim]
            if len(vec) < self.embedding_dim:
                vec = np.pad(vec, (0, self.embedding_dim - len(vec)))

            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec = vec / norm
            return vec.tolist()

        except Exception as e:
            logger.warning("Error extracting voice embedding: %s", e)
            # Deterministic fallback pseudo-vector from sample statistics
            pseudo = np.zeros(self.embedding_dim, dtype=np.float32)
            if len(samples) > 0:
                pseudo[0] = float(np.mean(samples))
                pseudo[1] = float(np.std(samples))
            return pseudo.tolist()


class SpeakerRegistry:
    """ChromaDB-backed vector registry for enrolled voice embeddings using cosine similarity."""

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        collection_name: str = "speaker_embeddings",
        client: Optional[chromadb.ClientAPI] = None,
        embedder: Optional[AcousticVoiceEmbedder] = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder or AcousticVoiceEmbedder()

        from mindmirror.memory.chroma_client import create_chroma_client
        self.client = create_chroma_client(db_dir=db_dir, client=client)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Speaker voice embeddings registry"},
        )

    def enroll_speaker(
        self,
        speaker_id: str,
        embedding: list[float],
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enrolls a speaker voice embedding vector into the registry."""
        meta = {"speaker_id": speaker_id, "name": name or speaker_id}
        if metadata:
            meta.update(metadata)

        doc = f"Voice embedding profile for speaker {name or speaker_id}"
        self.collection.upsert(
            ids=[f"emb_{speaker_id}"],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[meta],
        )
        logger.info("Enrolled voice embedding for speaker '%s'.", speaker_id)

    def enroll_speaker_audio(
        self,
        speaker_id: str,
        audio_data: bytes | np.ndarray,
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        """Extracts voice embedding from audio data and enrolls speaker."""
        embedding = self.embedder.extract_embedding(audio_data)
        self.enroll_speaker(speaker_id, embedding, name=name, metadata=metadata)
        return embedding

    def identify_speaker(
        self,
        embedding: list[float],
        similarity_threshold: float = 0.70,
    ) -> tuple[str, float]:
        """Matches a query voice embedding against enrolled speakers.

        Returns (matched_speaker_id, confidence). If below threshold, returns ('guest', confidence).
        """
        if self.collection.count() == 0:
            return "guest", 0.0

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=1,
            )
            if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                return "guest", 0.0

            distance = float(results["distances"][0][0])
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            matched_meta = results["metadatas"][0][0]
            speaker_id = matched_meta.get("speaker_id", "guest")

            if similarity >= similarity_threshold:
                return speaker_id, similarity
            else:
                logger.debug(
                    "Best match '%s' similarity %.2f below threshold %.2f.",
                    speaker_id,
                    similarity,
                    similarity_threshold,
                )
                return "guest", similarity

        except Exception as e:
            logger.error("Error matching speaker embedding: %s", e)
            return "guest", 0.0


class SpeakerIdentifierComponent(PipelineComponent):
    """Supervised component running in parallel with STT to identify speakers from vad.speech_audio."""

    def __init__(
        self,
        bus: MessageBus,
        registry: SpeakerRegistry,
        profile_store: Optional[SpeakerProfileStore] = None,
        speaker_context: Optional[SpeakerContext] = None,
        name: str = "speaker_identifier",
        similarity_threshold: float = 0.70,
        simulated_delay_s: float = 0.0,
    ) -> None:
        super().__init__(name=name, bus=bus)
        self.registry = registry
        self.profile_store = profile_store
        self.speaker_context = speaker_context
        self.similarity_threshold = similarity_threshold
        self.simulated_delay_s = simulated_delay_s
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await super().start()
        self.create_subscription_task("vad.speech_audio", self._handle_speech_audio)
        await asyncio.sleep(0.01)
        logger.info("SpeakerIdentifierComponent '%s' started, listening to 'vad.speech_audio'.", self.name)

    async def stop(self) -> None:
        await super().stop()
        logger.info("SpeakerIdentifierComponent '%s' stopped.", self.name)

    async def _handle_speech_audio(self, message: PipelineMessage) -> None:
        """Processes segmented speech audio asynchronously and emits speaker.identified."""
        async with self._lock:
            try:
                audio_payload = message.payload
                if not audio_payload:
                    return

                # Simulated processing delay (e.g. for non-blocking / parallel diarization tests)
                if self.simulated_delay_s > 0:
                    await asyncio.sleep(self.simulated_delay_s)

                # 1. Extract voice embedding in thread pool to prevent event loop starvation
                embedding = await asyncio.to_thread(
                    self.registry.embedder.extract_embedding, audio_payload
                )

                # 2. Match against speaker registry
                speaker_id, confidence = await asyncio.to_thread(
                    self.registry.identify_speaker,
                    embedding,
                    self.similarity_threshold,
                )

                # 3. Resolve profile
                profile = None
                if self.profile_store:
                    profile = self.profile_store.get_profile(speaker_id)

                logger.info(
                    "Speaker identified: '%s' (conf=%.2f, trace=%s)",
                    speaker_id,
                    confidence,
                    message.trace_id,
                )

                # 4. Update SpeakerContext shared state
                if self.speaker_context:
                    await self.speaker_context.update_speaker(
                        speaker_id=speaker_id,
                        confidence=confidence,
                        profile=profile,
                    )

                # 5. Publish speaker.identified event to message bus
                await self.bus.publish(
                    "speaker.identified",
                    PipelineMessage(
                        channel="speaker.identified",
                        phase=StreamPhase.CONTROL,
                        payload={
                            "speaker_id": speaker_id,
                            "confidence": confidence,
                            "is_known": speaker_id not in ("guest", "unresolved"),
                            "profile": profile.to_metadata() if profile else None,
                        },
                        speaker_id=speaker_id,
                        trace_id=message.trace_id,
                    ),
                )

            except Exception as e:
                self.record_error(e)
                logger.error("Error in SpeakerIdentifierComponent: %s", e)
