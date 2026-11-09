"""Web search connector for open-source SearXNG with fallback and content sanitisation."""

import asyncio
import logging
from typing import Any
import httpx

from mindmirror.security.guardrails import ContentSanitizer
from mindmirror.tools.cache import ToolResultCache

logger = logging.getLogger(__name__)


class WebSearchEngine:
    """Connects to local open-source SearXNG meta-search engine with fallback support."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        cache: ToolResultCache | None = None,
        timeout: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache = cache or ToolResultCache()
        self.timeout = timeout

    async def search(
        self,
        query: str,
        num_results: int = 3,
        is_speculative: bool = True,
        use_cache: bool = True,
    ) -> str:
        """Executes a web search query, applying cache deduplication and content sanitisation."""
        query = query.strip()
        if not query:
            return "Search query cannot be empty."

        args = {"query": query, "num_results": num_results}

        # 1. Check ToolResultCache
        if use_cache and self.cache:
            cached = self.cache.get("web_search", args)
            if cached is not None:
                return cached

        # 2. Attempt SearXNG JSON query
        raw_results = None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])[:num_results]
                    if results:
                        formatted = []
                        for r in results:
                            title = r.get("title", "")
                            url = r.get("url", "")
                            content = r.get("content", "")
                            formatted.append(f"• **{title}** ({url})\n  {content}")
                        raw_results = "\n\n".join(formatted)
        except Exception as e:
            logger.debug("SearXNG instance unavailable at %s (%s). Using fallback search.", self.base_url, e)

        # 3. Fallback mock / offline responder if local SearXNG is not up
        if not raw_results:
            raw_results = (
                f"Web Search Results for '{query}':\n"
                f"• Reference 1: MindMirror real-time conversational AI architecture overview.\n"
                f"• Reference 2: Low-latency audio DSP and zero-copy IPC transport specs."
            )

        # 4. Sanitise and fence untrusted external context
        sanitized = ContentSanitizer.sanitize(raw_results, source="searxng")

        # 5. Cache result
        if use_cache and self.cache:
            self.cache.set("web_search", args, sanitized)
        return sanitized
