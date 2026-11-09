"""Persistent Vector Memory Store backed by ChromaDB and langchain-chroma with retroactive speaker tagging."""

import asyncio
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from mindmirror import config
from mindmirror.pipeline.message import PipelineMessage

logger = logging.getLogger(__name__)


class ChromaMemoryManager:
    """Persistent Vector Memory Store backed by ChromaDB and langchain-chroma.

    Manages long-term memories, user facts, retroactive speaker tagging, and context retrieval.
    """

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        collection_name: str = "mindmirror_memories",
        client: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        self.collection_name = collection_name
        self._unresolved_memory_ids: list[str] = []
        self._bus_sub_task: Optional[asyncio.Task] = None

        from mindmirror.memory.chroma_client import create_chroma_client
        self.client = create_chroma_client(db_dir=db_dir, client=client)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Long-term persistent memories for MindMirror voice assistant"},
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
        )

    def as_retriever(self, **kwargs):
        """Returns a LangChain VectorStoreRetriever instance."""
        return self.vector_store.as_retriever(**kwargs)

    def add_memory(
        self,
        fact: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        speaker_id: Optional[str] = None,
    ) -> str:
        """Stores a fact into the persistent memory vector store.

        If speaker_id is None or 'unresolved', it is recorded as 'unresolved' and buffered
        for retroactive tagging when speaker.identified fires.
        Returns the unique memory ID.
        """
        if not fact or not fact.strip():
            return ""

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        resolved_speaker = speaker_id.strip() if speaker_id and speaker_id.strip() else "unresolved"

        meta: Dict[str, Any] = {
            "id": memory_id,
            "timestamp": timestamp,
            "category": category,
            "speaker_id": resolved_speaker,
        }
        if metadata:
            meta.update(metadata)

        doc = Document(page_content=fact.strip(), metadata=meta)
        self.vector_store.add_documents([doc], ids=[memory_id])

        if resolved_speaker == "unresolved":
            self._unresolved_memory_ids.append(memory_id)
            logger.debug("Memory '%s' stored with speaker_id='unresolved' (buffered for retro-tagging).", memory_id)
        else:
            logger.debug("Memory '%s' stored for speaker '%s'.", memory_id, resolved_speaker)

        return memory_id

    def retroactive_tag_memories(
        self,
        speaker_id: str,
        memory_ids: Optional[List[str]] = None,
    ) -> int:
        """Asynchronously updates metadata of unresolved memories to confirmed speaker_id.

        Args:
            speaker_id: The identified speaker ID to tag memories with.
            memory_ids: Specific memory IDs to update. If None, tags all buffered unresolved memories.

        Returns:
            Count of successfully updated memory entries.
        """
        if not speaker_id or speaker_id in ("unresolved", "guest"):
            return 0

        target_ids = list(memory_ids) if memory_ids is not None else list(self._unresolved_memory_ids)
        if not target_ids:
            return 0

        updated_count = 0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        for mid in target_ids:
            try:
                res = self.collection.get(ids=[mid])
                if res and res.get("metadatas") and len(res["metadatas"]) > 0:
                    meta = dict(res["metadatas"][0])
                    meta["speaker_id"] = speaker_id
                    meta["retroactive_tagged_at"] = timestamp
                    meta["is_retroactive"] = True

                    self.collection.update(
                        ids=[mid],
                        metadatas=[meta],
                    )
                    updated_count += 1
                    logger.info("Retroactively tagged memory '%s' with speaker_id='%s'.", mid, speaker_id)
            except Exception as e:
                logger.warning("Failed to retroactively tag memory '%s': %s", mid, e)

        # Clear tagged IDs from buffer
        if memory_ids is None:
            self._unresolved_memory_ids.clear()
        else:
            self._unresolved_memory_ids = [m for m in self._unresolved_memory_ids if m not in target_ids]

        return updated_count

    def get_unresolved_memory_ids(self) -> List[str]:
        """Returns the list of memory IDs currently pending retroactive speaker tagging."""
        return list(self._unresolved_memory_ids)

    def recall_memories(
        self,
        query: str,
        limit: int = 5,
        speaker_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries langchain-chroma vector store for relevant stored memories matching the query context.

        Args:
            query: The search text query.
            limit: Maximum items to return.
            speaker_id: Optional speaker ID to filter memories by.

        Returns:
            List of dicts with keys: 'id', 'fact', 'timestamp', 'category', 'speaker_id', 'distance'.
        """
        if not query or not query.strip() or self.collection.count() == 0:
            return []

        search_kwargs: Dict[str, Any] = {"k": min(limit * 2 if speaker_id else limit, self.collection.count())}
        if speaker_id and speaker_id != "unresolved":
            search_kwargs["filter"] = {"speaker_id": speaker_id}

        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query.strip(),
                **search_kwargs,
            )
        except Exception:
            # Fallback if filter has 0 matches
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query.strip(),
                k=min(limit, self.collection.count()),
            )

        recalled = []
        for doc, score in docs_and_scores:
            doc_speaker = doc.metadata.get("speaker_id", "unresolved")
            if speaker_id and speaker_id != "unresolved" and doc_speaker not in (speaker_id, "general", "all"):
                continue

            recalled.append({
                "id": doc.metadata.get("id", ""),
                "fact": doc.page_content,
                "timestamp": doc.metadata.get("timestamp", ""),
                "category": doc.metadata.get("category", "general"),
                "speaker_id": doc_speaker,
                "distance": float(score),
            })
            if len(recalled) >= limit:
                break

        return recalled

    def get_all_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves up to `limit` stored memories from the collection."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(limit=limit)
        items = []
        if results and results.get("documents"):
            for doc, mid, meta in zip(results["documents"], results["ids"], results["metadatas"]):
                items.append({
                    "id": mid,
                    "fact": doc,
                    "timestamp": meta.get("timestamp", ""),
                    "category": meta.get("category", "general"),
                    "speaker_id": meta.get("speaker_id", "unresolved"),
                })
        return items

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory item by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            if memory_id in self._unresolved_memory_ids:
                self._unresolved_memory_ids.remove(memory_id)
            return True
        except Exception:
            return False

    async def attach_bus(self, bus) -> None:
        """Attaches message bus to automatically listen for speaker.identified and retro-tag memories."""
        if self._bus_sub_task and not self._bus_sub_task.done():
            self._bus_sub_task.cancel()
        ready_event = asyncio.Event()
        self._bus_sub_task = asyncio.create_task(self._listen_for_speaker_identified(bus, ready_event))
        await ready_event.wait()

    def _process_speaker_identified(self, message: PipelineMessage) -> None:
        payload = message.payload
        if isinstance(payload, dict):
            speaker_id = payload.get("speaker_id", message.speaker_id)
            if speaker_id and speaker_id not in ("unresolved", "guest"):
                tagged = self.retroactive_tag_memories(speaker_id)
                if tagged > 0:
                    logger.info(
                        "Auto-retroactively tagged %d memories on 'speaker.identified' event for '%s'.",
                        tagged,
                        speaker_id,
                    )

    async def _listen_for_speaker_identified(self, bus, ready_event: Optional[asyncio.Event] = None) -> None:
        try:
            generator = bus.subscribe("speaker.identified")
            first_step_task = asyncio.create_task(anext(generator))
            await asyncio.sleep(0.01)
            if ready_event:
                ready_event.set()

            first_msg = await first_step_task
            self._process_speaker_identified(first_msg)

            async for message in generator:
                self._process_speaker_identified(message)
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        except Exception as e:
            logger.error("Error in ChromaMemoryManager bus subscriber: %s", e)

    async def close(self) -> None:
        """Closes background subscription tasks."""
        if self._bus_sub_task and not self._bus_sub_task.done():
            self._bus_sub_task.cancel()
            try:
                await self._bus_sub_task
            except asyncio.CancelledError:
                pass
            self._bus_sub_task = None
