"""OpenTelemetry-aligned latency tracking and distributed tracing telemetry."""

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import Any

from mindmirror.pipeline.message import PipelineMessage, StreamPhase
from mindmirror.pipeline.transport.interface import MessageBus

logger = logging.getLogger(__name__)


@dataclass
class LatencyReport:
    """End-to-end glass-to-glass latency metrics for a single conversational turn."""
    trace_id: str
    speaker_id: str | None = None
    vad_to_transcript_ms: float = 0.0
    time_to_first_thought_ms: float = 0.0
    total_reasoning_duration_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tts_synthesis_ms: float = 0.0
    total_glass_to_glass_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "speaker_id": self.speaker_id,
            "vad_to_transcript_ms": round(self.vad_to_transcript_ms, 2),
            "time_to_first_thought_ms": round(self.time_to_first_thought_ms, 2),
            "total_reasoning_duration_ms": round(self.total_reasoning_duration_ms, 2),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "tts_synthesis_ms": round(self.tts_synthesis_ms, 2),
            "total_glass_to_glass_ms": round(self.total_glass_to_glass_ms, 2),
            "metadata": self.metadata,
        }


class TurnSpanTracker:
    """Measures and records pipeline stage latencies across a conversational turn trace."""

    def __init__(self, trace_id: str, bus: MessageBus | None = None, speaker_id: str | None = None) -> None:
        self.trace_id = trace_id
        self.bus = bus
        self.speaker_id = speaker_id
        self.report = LatencyReport(trace_id=trace_id, speaker_id=speaker_id)
        self._start_time: float = time.perf_counter()
        self._spans: dict[str, float] = {}

    def start_span(self, span_name: str) -> None:
        self._spans[span_name] = time.perf_counter()

    def end_span(self, span_name: str) -> float:
        if span_name not in self._spans:
            return 0.0
        duration_ms = (time.perf_counter() - self._spans[span_name]) * 1000.0

        if span_name == "vad_to_transcript":
            self.report.vad_to_transcript_ms = duration_ms
        elif span_name == "first_thought":
            self.report.time_to_first_thought_ms = duration_ms
        elif span_name == "reasoning":
            self.report.total_reasoning_duration_ms = duration_ms
        elif span_name == "first_token":
            self.report.time_to_first_token_ms = duration_ms
        elif span_name == "tts_synthesis":
            self.report.tts_synthesis_ms = duration_ms

        return duration_ms

    async def finalize_and_publish(self) -> LatencyReport:
        """Finalizes total turn duration and publishes LatencyReport to metrics.latency."""
        self.report.total_glass_to_glass_ms = (time.perf_counter() - self._start_time) * 1000.0

        logger.info(
            "Turn [trace=%s] Latency: Glass-to-Glass=%.2fms (Reasoning=%.2fms, TTS=%.2fms)",
            self.trace_id,
            self.report.total_glass_to_glass_ms,
            self.report.total_reasoning_duration_ms,
            self.report.tts_synthesis_ms,
        )

        if self.bus:
            await self.bus.publish(
                "metrics.latency",
                PipelineMessage(
                    channel="metrics.latency",
                    phase=StreamPhase.TELEMETRY,
                    payload=self.report.to_dict(),
                    trace_id=self.trace_id,
                    speaker_id=self.speaker_id,
                ),
            )

        # Optional export to Jaeger via OpenTelemetry OTLP HTTP
        await self._export_to_jaeger()

        return self.report

    async def _export_to_jaeger(self) -> None:
        """Asynchronously exports turn latency spans to Jaeger via OTLP HTTP if configured."""
        import os
        endpoint = os.getenv("JAEGER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return

        try:
            import httpx
            # Construct OpenTelemetry-compatible resourceSpans payload
            trace_hex = self.trace_id.replace("-", "").ljust(32, "0")[:32]
            start_nanos = int(self._start_time * 1e9)
            end_nanos = start_nanos + int(self.report.total_glass_to_glass_ms * 1e6)

            otlp_payload = {
                "resourceSpans": [{
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "mindmirror"}},
                            {"key": "speaker.id", "value": {"stringValue": str(self.speaker_id or "unresolved")}},
                        ]
                    },
                    "scopeSpans": [{
                        "scope": {"name": "mindmirror.turn_tracker"},
                        "spans": [{
                            "traceId": trace_hex,
                            "spanId": trace_hex[:16],
                            "name": "conversational_turn",
                            "kind": 1,
                            "startTimeUnixNano": str(start_nanos),
                            "endTimeUnixNano": str(end_nanos),
                            "attributes": [
                                {"key": "vad_to_transcript_ms", "value": {"doubleValue": self.report.vad_to_transcript_ms}},
                                {"key": "reasoning_ms", "value": {"doubleValue": self.report.total_reasoning_duration_ms}},
                                {"key": "tts_ms", "value": {"doubleValue": self.report.tts_synthesis_ms}},
                                {"key": "glass_to_glass_ms", "value": {"doubleValue": self.report.total_glass_to_glass_ms}},
                            ],
                        }]
                    }]
                }]
            }

            async with httpx.AsyncClient(timeout=0.5) as client:
                await client.post(endpoint, json=otlp_payload)
        except Exception as e:
            logger.debug("Optional Jaeger OTLP export error: %s", e)
