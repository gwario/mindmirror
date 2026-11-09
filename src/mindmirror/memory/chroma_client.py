"""ChromaDB Client Factory supporting local SQLite persistence and remote HTTP container connections."""

import logging
import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


def create_chroma_client(
    db_dir: Optional[Path] = None,
    client: Optional[chromadb.ClientAPI] = None,
) -> chromadb.ClientAPI:
    """Creates a ChromaDB Client instance.

    Priority:
    1. Explicit `client` instance (e.g. mock or custom instance).
    2. Explicit `db_dir` (e.g. pytest `tmp_path` hermetic directory -> PersistentClient).
    3. Remote `CHROMA_HOST` environment variable (e.g. Docker container -> HttpClient).
    4. Default local directory `data/chroma_db` (-> PersistentClient).
    """
    if client is not None:
        return client

    # If an explicit test directory is provided, honour it with a hermetic local client
    if db_dir is not None:
        db_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    # If running in a containerized environment with remote ChromaDB server
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        try:
            logger.info("Connecting to remote ChromaDB service at http://%s:%d", chroma_host, chroma_port)
            http_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
            http_client.heartbeat()
            return http_client
        except Exception as e:
            logger.warning(
                "Could not connect to remote ChromaDB at %s:%d (%s); falling back to local persistent directory.",
                chroma_host,
                chroma_port,
                e,
            )

    # Fallback to local persistent directory
    from mindmirror import config
    default_path = getattr(config, "CHROMA_DB_DIR", Path("data/chroma_db"))
    default_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(default_path),
        settings=Settings(anonymized_telemetry=False),
    )
