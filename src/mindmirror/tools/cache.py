"""ToolResultCache supporting local in-memory LRU and distributed Valkey/Redis caching."""

from collections import OrderedDict
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolResultCache:
    """Two-tier ToolResultCache: L1 in-memory LRU + L2 distributed Valkey/Redis cache.

    Deduplicates tool calls across speculative reasoning branches and distributed
    reasoning microservice replicas. Prevents duplicate network calls, web searches,
    or disk reads when multiple MCTS / Tree-of-Thoughts candidate branches request
    identical queries.
    """

    def __init__(
        self,
        max_entries: int = 128,
        valkey_url: Optional[str] = None,
        ttl_seconds: int = 300,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.hit_count: int = 0
        self.miss_count: int = 0

        # Optional Valkey L2 connection
        self._valkey: Any = None
        target_url = valkey_url or os.getenv("VALKEY_URL")
        if target_url:
            try:
                import redis
                self._valkey = redis.Redis.from_url(target_url, decode_responses=True)
                # Quick ping to verify connectivity
                self._valkey.ping()
                logger.info("ToolResultCache connected to distributed Valkey cache at %s", target_url)
            except Exception as e:
                logger.warning("Could not connect to Valkey cache at %s (%s); operating in L1 memory-only mode.", target_url, e)
                self._valkey = None

    def _make_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Generates a canonical serialized string key for a tool invocation."""
        try:
            sorted_args = json.dumps(arguments, sort_keys=True, default=str)
        except Exception:
            sorted_args = str(sorted(arguments.items()))
        return f"{tool_name}:{sorted_args}"

    def get(self, tool_name: str, arguments: dict[str, Any]) -> Any | None:
        """Retrieves cached result if present in L1 or L2, recording cache hits."""
        key = self._make_key(tool_name, arguments)

        # 1. Check L1 in-memory LRU
        if key in self._cache:
            self.hit_count += 1
            self._cache.move_to_end(key)
            logger.debug("ToolResultCache L1 HIT for '%s' (hits=%d)", tool_name, self.hit_count)
            return self._cache[key]

        # 2. Check L2 distributed Valkey cache if configured
        if self._valkey:
            try:
                raw = self._valkey.get(f"mm:tool_cache:{key}")
                if raw is not None:
                    result = json.loads(raw)
                    # Backfill L1
                    self.set(tool_name, arguments, result, sync_valkey=False)
                    self.hit_count += 1
                    logger.debug("ToolResultCache L2 Valkey HIT for '%s' (hits=%d)", tool_name, self.hit_count)
                    return result
            except Exception as e:
                logger.warning("Valkey read error: %s", e)

        self.miss_count += 1
        return None

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        sync_valkey: bool = True,
    ) -> None:
        """Stores result in L1 cache and syncs to L2 Valkey with TTL."""
        key = self._make_key(tool_name, arguments)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = result
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        if sync_valkey and self._valkey:
            try:
                serialized = json.dumps(result, default=str)
                self._valkey.setex(f"mm:tool_cache:{key}", self.ttl_seconds, serialized)
            except Exception as e:
                logger.warning("Valkey write error: %s", e)

    def has(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Checks if a key exists in cache without incrementing hit/miss counters."""
        key = self._make_key(tool_name, arguments)
        if key in self._cache:
            return True
        if self._valkey:
            try:
                return bool(self._valkey.exists(f"mm:tool_cache:{key}"))
            except Exception:
                pass
        return False

    def clear(self) -> None:
        """Clears all cached entries and resets statistics."""
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        if self._valkey:
            try:
                keys = self._valkey.keys("mm:tool_cache:*")
                if keys:
                    self._valkey.delete(*keys)
            except Exception as e:
                logger.warning("Valkey clear error: %s", e)
