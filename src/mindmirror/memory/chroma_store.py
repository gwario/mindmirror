import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from mindmirror import config

class ChromaMemoryManager:
    """
    Persistent Vector Memory Store backed by ChromaDB and langchain-chroma.
    Manages long-term memories, user facts, and context retrieval.
    """

    def __init__(self, db_dir: Optional[Path] = None, collection_name: str = "mindmirror_memories"):
        self.db_dir = db_dir or getattr(config, 'CHROMA_DB_DIR', Path("data/chroma_db"))
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Long-term persistent memories for MindMirror voice assistant"}
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name
        )

    def as_retriever(self, **kwargs):
        """Returns a LangChain VectorStoreRetriever instance."""
        return self.vector_store.as_retriever(**kwargs)

    def add_memory(self, fact: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores a fact into the persistent memory vector store.
        Returns the unique memory ID.
        """
        if not fact or not fact.strip():
            return ""

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        meta = {
            "timestamp": timestamp,
            "category": category
        }
        if metadata:
            meta.update(metadata)

        doc = Document(page_content=fact.strip(), metadata=meta)
        self.vector_store.add_documents([doc], ids=[memory_id])
        return memory_id

    def recall_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries langchain-chroma vector store for relevant stored memories matching the query context.
        Returns a list of dicts with keys: 'id', 'fact', 'timestamp', 'category', 'distance'.
        """
        if not query or not query.strip() or self.collection.count() == 0:
            return []

        docs_and_scores = self.vector_store.similarity_search_with_score(
            query=query.strip(),
            k=min(limit, self.collection.count())
        )

        recalled = []
        for doc, score in docs_and_scores:
            recalled.append({
                "id": doc.metadata.get("id", ""),
                "fact": doc.page_content,
                "timestamp": doc.metadata.get("timestamp", ""),
                "category": doc.metadata.get("category", "general"),
                "distance": float(score)
            })

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
                    "category": meta.get("category", "general")
                })
        return items

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory item by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
