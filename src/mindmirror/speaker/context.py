"""Thread-safe, async shared state holder for current speaker identification and profile."""

import asyncio
from collections.abc import Awaitable, Callable
import logging
from typing import Optional

from mindmirror.pipeline.message import PipelineMessage
from mindmirror.pipeline.transport.interface import MessageBus
from mindmirror.speaker.profile import SpeakerProfile, SpeakerProfileStore

logger = logging.getLogger(__name__)


class SpeakerContext:
    """Thread-safe, non-blocking shared state tracking active speaker and resolved parameters.

    Allows optimistic pipeline execution: STT and Reasoning can start while speaker
    identification is actively running in parallel, reading fallback defaults until
    `speaker.identified` fires.
    """

    _instance: Optional["SpeakerContext"] = None

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        profile_store: Optional[SpeakerProfileStore] = None,
        valkey_url: Optional[str] = None,
    ) -> None:
        import os
        self.bus = bus
        self.profile_store = profile_store
        self._valkey_url = valkey_url or os.getenv("VALKEY_URL")
        self._speaker_id: str = "unresolved"
        self._confidence: float = 0.0
        self._profile: SpeakerProfile = SpeakerProfile(speaker_id="unresolved", name="Unresolved Speaker")
        self._speaker_event: asyncio.Event = asyncio.Event()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._listeners: list[Callable[[str, SpeakerProfile], Awaitable[None] | None]] = []
        self._sub_task: Optional[asyncio.Task] = None

        if bus:
            self.attach_bus(bus)

        SpeakerContext._instance = self

    @classmethod
    def get_instance(cls, bus: Optional[MessageBus] = None) -> "SpeakerContext":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = SpeakerContext(bus=bus)
        elif bus and not cls._instance.bus:
            cls._instance.attach_bus(bus)
        return cls._instance

    @property
    def current_speaker_id(self) -> str:
        """Returns the current speaker ID (may be 'unresolved' if still identifying)."""
        return self._speaker_id

    @property
    def current_profile(self) -> SpeakerProfile:
        """Returns the current resolved SpeakerProfile."""
        return self._profile

    @property
    def confidence(self) -> float:
        """Returns the speaker identification confidence score (0.0 to 1.0)."""
        return self._confidence

    @property
    def is_resolved(self) -> bool:
        """True if the current speaker ID is confirmed (not 'unresolved')."""
        return self._speaker_id not in ("unresolved", "", "guest_unresolved")

    def add_listener(self, listener: Callable[[str, SpeakerProfile], Awaitable[None] | None]) -> None:
        """Registers a callback that fires when speaker ID or profile updates."""
        self._listeners.append(listener)

    def attach_bus(self, bus: MessageBus) -> None:
        """Attaches message bus to automatically listen for speaker.identified events."""
        self.bus = bus
        if self._sub_task and not self._sub_task.done():
            self._sub_task.cancel()
        self._sub_task = asyncio.create_task(self._listen_for_speaker_events())

    async def _listen_for_speaker_events(self) -> None:
        """Consumes speaker.identified messages from the message bus."""
        if not self.bus:
            return
        try:
            async for message in self.bus.subscribe("speaker.identified"):
                payload = message.payload
                if isinstance(payload, dict):
                    speaker_id = payload.get("speaker_id", message.speaker_id or "unresolved")
                    confidence = float(payload.get("confidence", 1.0))
                    await self.update_speaker(speaker_id=speaker_id, confidence=confidence)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in SpeakerContext event listener: %s", e)

    async def update_speaker(
        self,
        speaker_id: str,
        confidence: float = 1.0,
        profile: Optional[SpeakerProfile] = None,
    ) -> None:
        """Updates active speaker ID and dynamically loads associated SpeakerProfile."""
        async with self._lock:
            self._speaker_id = speaker_id
            self._confidence = confidence

            # Resolve profile
            if profile:
                self._profile = profile
            elif self.profile_store:
                stored = self.profile_store.get_profile(speaker_id)
                if stored:
                    self._profile = stored
                else:
                    self._profile = SpeakerProfile(speaker_id=speaker_id, name=speaker_id)
            else:
                self._profile = SpeakerProfile(speaker_id=speaker_id, name=speaker_id)

            self._speaker_event.set()
            logger.info(
                "SpeakerContext updated: speaker_id='%s' (conf=%.2f, VAD pause=%.2fs)",
                self._speaker_id,
                self._confidence,
                self._profile.vad_silence_duration,
            )

        # Notify listeners
        for listener in list(self._listeners):
            try:
                res = listener(self._speaker_id, self._profile)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.warning("Error in SpeakerContext listener: %s", e)

        # Optional distributed sync via Valkey
        if self._valkey_url:
            try:
                import json
                import redis.asyncio as aioredis
                r = aioredis.from_url(self._valkey_url)
                val = json.dumps({
                    "speaker_id": self._speaker_id,
                    "confidence": self._confidence,
                    "profile": self._profile.to_metadata(),
                })
                await r.set("mm:speaker:active", val, ex=3600)
                await r.aclose()
            except Exception as e:
                logger.debug("Optional Valkey speaker sync error: %s", e)

    async def wait_for_speaker(self, timeout: float = 1.0) -> str:
        """Waits for speaker identification to resolve with timeout.

        Returns confirmed speaker_id or 'unresolved' upon timeout.
        """
        if self.is_resolved:
            return self._speaker_id

        try:
            await asyncio.wait_for(self._speaker_event.wait(), timeout=timeout)
            return self._speaker_id
        except asyncio.TimeoutError:
            logger.debug("Speaker identification wait timed out after %.2fs; proceeding optimistically.", timeout)
            return self._speaker_id

    def reset(self) -> None:
        """Resets speaker state to unresolved for a new turn."""
        self._speaker_id = "unresolved"
        self._confidence = 0.0
        self._profile = SpeakerProfile(speaker_id="unresolved", name="Unresolved Speaker")
        self._speaker_event.clear()

    async def close(self) -> None:
        """Cancels background bus subscription task."""
        if self._sub_task and not self._sub_task.done():
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass
            self._sub_task = None
