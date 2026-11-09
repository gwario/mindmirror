# 🗺️ MindMirror Master Implementation Plan

> **Synthesising Architecture Reviews v1, v2, v3, and v4 + External Tools, Security & Observability**  
> This master document consolidates every architectural requirement discussed across all iterations into an actionable, phased engineering blueprint.

---

## 1. Synthesis of Architectural Evolution (v1 ➔ v4 + Tooling & Guardrails)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ARCHITECTURAL SYNTHESIS                                          │
├───────────────────────────────────┬────────────────────────────────────────────────────────────────────┤
│ Review Iteration                  │ Core Contribution Carried Forward                                  │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ v1: Core Codebase Analysis        │ • Kill file-based locks (speaking.lock, playback.lock)              │
│                                   │ • Decompose run_stt_loop god function                               │
│                                   │ • Unify ABC interfaces across STT, TTT, TTS                        │
│                                   │ • Fix provider-specific exception leaks                            │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ v2: Scale & Parallelism           │ • Parallel background speaker identification                       │
│                                   │ • Retroactive memory tagging in ChromaDB                           │
│                                   │ • Context window management & reasoning-aware compression          │
│                                   │ • Component lifecycle (start, drain, stop, health)                 │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ v3: Pruning & Reliability         │ • Shift from model fine-tuning to per-speaker runtime params        │
│                                   │ • Erlang-style ComponentSupervisor (crash recovery & backoff)      │
│                                   │ • StreamPhase token demuxing (thinking vs responding)              │
│                                   │ • Replace Redis-for-audio with NATS / ZeroMQ                       │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ v4: Reasoning & Open Source       │ • Pluggable Reasoning Engine (custom MCTS/ToT vs native API)       │
│                                   │ • Complete open-source software stack for every component          │
│                                   │ • Concrete OSS service connectors (NATS, Valkey, ZeroMQ)           │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ ✨ Extended: Tools & Ops          │ • External Info: Model Context Protocol (MCP) + SearXNG Web Search │
│                                   │ • Security: Tool side-effect fencing & untrusted input sanitisation │
│                                   │ • Observability: OpenTelemetry tracing & glass-to-glass latency    │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ 🧪 Extended: Testability & QA     │ • Glass-to-Glass PipelineRegressionHarness (zero hardware needed) │
│                                   │ • Hermetic simulation: SyntheticAudioSource & VirtualAudioSink     │
│                                   │ • Deterministic golden scenarios (barge-in, style tags, chaos)    │
│                                   │ • Turnkey single-command test runner (run_regression_suite.py)     │
└───────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

---

## 2. Target High-Level System Architecture

```mermaid
graph TB
    subgraph "Infrastructure & Transport Layer"
        BUS["MessageBus (ABC)<br/>[LocalBus | ZmqBus | NatsBus | ValkeyBus]"]
        SUP["ComponentSupervisor<br/>[Crash Recovery | Health Checks | Redeploy]"]
        SC["SpeakerContext (Shared State)<br/>[Active Speaker | Profile Cache]"]
        OTEL["Observability & Tracing<br/>[OpenTelemetry Spans | Latency Telemetry]"]
    end

    subgraph "Audio & Perception Pipeline"
        MIC["🎙️ AudioCapture<br/>(PortAudio / PipeWire)"]
        VAD["Silero VAD (OSS)<br/>(Dynamic Per-Speaker Silence Threshold)"]
        STT["STT Engine<br/>(Faster-Whisper OSS / Google)"]
        SID["Speaker Identifier (Parallel OSS)<br/>(pyannote.audio / ECAPA-TDNN)"]
    end

    subgraph "Cognitive & Reasoning Layer"
        CTX["ContextManager<br/>(Thinking Pruning & Summarisation)"]
        PROMPT["PromptBuilder<br/>(Speaker Profiles + Chroma Memory)"]
        REASON["ReasoningEngine (Strategy Pattern)<br/>[Custom ToT / MCTS / Reflexion] OR<br/>[Provider Native: Gemini / Claude / DeepSeek]"]
        SEC["Security & Guardrails<br/>[Tool Side-Effect Fence | Prompt Sanitiser]"]
    end

    subgraph "External Information & Tooling"
        MCP["MCPClientManager (OSS)<br/>[Stdio & SSE Tool Servers]"]
        SEARCH["WebSearchEngine (OSS)<br/>[SearXNG Meta-Search / Local Crawler]"]
        CACHE["ToolResultCache<br/>(In-Memory Deduplication)"]
    end

    subgraph "Synthesis & Output Pipeline"
        TTS["TTS Engine<br/>(Piper OSS / F5-TTS / Google)"]
        PLAY["PlaybackLoop<br/>(Audio Device Sink)"]
    end

    subgraph "Automated Testing & Simulation Harness"
        HARNESS["🧪 PipelineRegressionHarness<br/>[Deterministic Event Clock | Probe Recorder]"]
        SYNAUDIO["SyntheticAudioSource<br/>(Golden WAVs / PCM Injection)"]
        VIRTSINK["VirtualAudioSink<br/>(Buffer Inspector | Zero Audio Hardware)"]
        MOCKENG["DeterministicReasoningEngine<br/>(Canned Scenarios | Style Verification)"]
    end

    subgraph "Persistence Layer"
        CHROMA[("ChromaDB Vector Store")]
        MEM["Conversation Memories<br/>(Speaker-Partitioned)"]
        PROF["Speaker Profiles<br/>(VAD, TTS & Reasoning Overrides)"]
        EMB["Voice Embeddings"]
    end

    %% Audio & Perception Flow
    MIC -->|"audio.raw"| BUS
    BUS --> VAD
    VAD -->|"vad.speech_audio"| BUS
    BUS --> STT
    BUS --> SID

    %% Test Harness Injections & Inspections
    HARNESS -.->|"controls & records"| BUS
    SYNAUDIO -->|"audio.raw (simulated)"| BUS
    BUS -->|"audio.playback (captured)"| VIRTSINK
    HARNESS -.->|"asserts pipeline contracts"| VIRTSINK
    HARNESS -.->|"substitutes live LLM"| MOCKENG

    %% Speaker Context & Memory Flow
    SID -.->|"speaker.identified"| SC
    SC -.->|"profile lookup"| PROF
    SC -.->|"retroactive tag"| MEM
    SID --> EMB

    %% Cognitive Flow
    STT -->|"stt.transcript"| BUS
    BUS --> CTX
    CTX --> PROMPT
    SC -.->|"speaker params"| PROMPT
    MEM -.->|"relevant memories"| PROMPT
    PROMPT --> REASON

    %% External Tooling Flow
    REASON <--> SEC
    SEC <--> MCP
    SEC <--> SEARCH
    MCP <--> CACHE
    SEARCH <--> CACHE

    %% Reasoning Telemetry vs Audio Output
    REASON -->|"reasoning.trace (MCTS steps)"| BUS
    REASON -->|"ttt.thinking (internal deliberation)"| BUS
    REASON -->|"ttt.response (final [STYLE] speech)"| BUS

    %% Speech Output Flow
    BUS --> TTS
    SC -.->|"per-speaker voice/pitch"| TTS
    TTS --> PLAY

    %% Observability Tracking
    BUS -.->|"trace propagation"| OTEL

    %% Storage Associations
    CHROMA --- MEM
    CHROMA --- PROF
    CHROMA --- EMB

    %% Supervisor
    SUP -.->|"monitor/restart"| MIC
    SUP -.->|"monitor/restart"| VAD
    SUP -.->|"monitor/restart"| STT
    SUP -.->|"monitor/restart"| SID
    SUP -.->|"monitor/restart"| REASON
    SUP -.->|"monitor/restart"| TTS
    SUP -.->|"monitor/restart"| PLAY
```

---

## 3. External Information Sources & MCP Tooling

When exploring unconstrained custom reasoning algorithms (e.g. MCTS or Tree-of-Thoughts), external information gathering is not just a linear callback — it is an **action step** within the reasoning graph.

### 3.1 Architecture of External Tool Execution

```
                     Candidate Thought Node (Reasoning Graph)
                                       │
                      Does hypothesis require external data?
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
        [1. MCP Tool Invocation]                [2. Web Search Query]
       (Filesystem, GitHub, Git,               (Open-source SearXNG instance
        Local Shell, System APIs)               or privacy-respecting search)
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       ▼
                         ToolResultCache (In-Memory)
                (Prevents redundant queries during branch rollouts)
                                       ▼
                           Security Guardrails Fence
               (Classifies Read-Only vs Mutating; Sanitises Content)
                                       ▼
                         Enriched Reasoning Scratchpad
```

### 3.2 Tool Integration Engine

1. **Model Context Protocol (MCP)**:
   - MindMirror's [MCPClientManager](src/mindmirror/mcp_client/client.py) connects to standard MCP servers over `stdio` and `sse`.
   - The Reasoning Engine dynamically queries `mcp_client.get_all_tools()` and exposes them to reasoning branch generators.
2. **Open-Source Web Search (SearXNG)**:
   - Default OSS Implementation: Self-hosted [SearXNG](https://github.com/searxng/searxng) running locally via Docker (`http://127.0.0.1:8080/search?format=json`).
   - Fallback connectors: DuckDuckGo / Tavily / Brave Search API.
   - Summarisation / Extraction: Fetched web results are parsed through a lightweight markdown extractor (e.g. `trafilatura` or `readabilipy`) before feeding into the reasoning scratchpad.
3. **Speculative Branch Tool Caching**:
   - Because tree-search algorithms evaluate multiple variations of a problem, branches often request the same underlying data (e.g. weather forecast, git commit log).
   - An ephemeral LRU `ToolResultCache` deduplicates queries within the conversation turn.

---

## 4. Security & Guardrails Architecture

Even if security isn't the primary day-one objective, building with **clean security boundaries** prevents painful rewrites later:

### 4.1 Tool Side-Effect Fencing (Speculative Execution Safety)

In custom reasoning algorithms with unconstrained token budgets (like MCTS or beam search), the engine speculatively explores hypotheses. **Speculative branches must never trigger external side-effects.**

```python
# src/mindmirror/security/guardrails.py

from enum import Enum
from typing import Any

class ToolPurity(Enum):
    READ_ONLY = "read_only"       # Safe for speculative tree exploration (e.g. web search, file read)
    MUTATING = "mutating"         # Forbidden during speculative exploration (e.g. delete file, send email)

class SecurityGuardrail:
    """Enforces execution boundaries and prevents side-effect leakage during reasoning."""

    def __init__(self, tool_registry: dict[str, ToolPurity]):
        self.tool_registry = tool_registry

    def validate_tool_call(self, tool_name: str, is_speculative: bool) -> bool:
        purity = self.tool_registry.get(tool_name, ToolPurity.MUTATING) # Default safe: assume mutating
        if is_speculative and purity == ToolPurity.MUTATING:
            raise PermissionError(
                f"Speculative branch attempted to execute mutating tool '{tool_name}'. "
                f"Mutating actions are only permitted on the confirmed winning execution path."
            )
        return True
```

### 4.2 Untrusted Content Sanitisation & Injection Defence
- External web search results, scraped pages, and third-party MCP outputs are untrusted.
- Tool outputs are sanitized:
  1. Strip dangerous terminal control characters and zero-width unicode spaces.
  2. Truncate outputs to a strict character limit (e.g. 4,000 characters).
  3. Wrap the content in strict delimiter XML tags:
     ```xml
     <external_untrusted_context source="searxng">
     ... sanitized text ...
     </external_untrusted_context>
     ```
  4. System prompts explicitly instruct the model to treat content within `<external_untrusted_context>` strictly as passive reference data, never as system instructions.

### 4.3 Local Socket & Biometric Security
- **UNIX Domain Sockets**: ZeroMQ IPC sockets created in `/tmp/mindmirror_*.ipc` are chmodded to `0600` (read/write by owner only), preventing local unprivileged users from sniffing raw PCM audio streams.
- **Biometric & Memory Privacy**: Voice embeddings (`pyannote` vectors) and personal ChromaDB facts reside on local disk (`data/chroma_db/`), never transmitted to external cloud endpoints.

---

## 5. Observability & Telemetry Architecture

Voice interactions require sub-second responsiveness. Observability is the only way to diagnose why a response was slow: was it STT transcription, reasoning branch search, MCP tool delay, or TTS synthesis?

### 5.1 OpenTelemetry (OTel) Distributed Tracing

Every turn receives a unique `trace_id` propagated through `PipelineMessage.metadata`.

```
[Trace: turn-8f12a]
├─ Span: AudioCapture (PCM chunks) ─── [20ms]
├─ Span: SileroVAD (Speech detected) ─── [15ms]
├─ Span: STT_FasterWhisper (Transcribing audio) ─── [180ms]
├─ Span: ReasoningEngine (TreeOfThoughts) ─── [720ms]
│   ├─ Span: CandidateGeneration (Branch A, B, C) ─── [240ms]
│   ├─ Span: MCP_ToolCall (web_search: "ZeroMQ vs NATS") ─── [160ms]
│   ├─ Span: SelfCritique & Scoring ─── [180ms]
│   └─ Span: FinalSpeechSynthesisPrompt ─── [140ms] (Time to First Token: 80ms)
├─ Span: TTS_PiperVoice (Synthesizing audio) ─── [45ms]
└─ Span: PlaybackLoop (First sound played) ─── [10ms]
─────────────────────────────────────────────────────────────────────────────
Total Glass-to-Glass Latency: 990ms
```

### 5.2 Real-Time Latency Metrics on the MessageBus

Every stage measures and publishes latency events to `metrics.latency`:

```python
# src/mindmirror/pipeline/telemetry.py

@dataclass
class LatencyReport:
    trace_id: str
    speaker_id: str | None
    vad_to_transcript_ms: float
    time_to_first_thought_ms: float
    total_reasoning_duration_ms: float
    time_to_first_token_ms: float
    tts_synthesis_ms: float
    total_glass_to_glass_ms: float
```

### 5.3 Live Reasoning Visualisation
- Custom reasoning algorithms publish branch events to `reasoning.trace`.
- An optional web UI or terminal dashboard subscribes to `reasoning.trace` to render the live thought graph/DAG as it unfolds.

---

## 6. Testability, Synthetic Simulation & Automated Regression Architecture

Voice AI architectures are notoriously difficult to test because they combine **hardware I/O** (microphones, soundcards), **real-time asynchronous multiprocessing**, **probabilistic cloud LLMs**, and **stateful persistent memory**. Without a dedicated simulation harness, regression testing is impossible without sitting in front of a live microphone.

### 6.1 The Voice AI Testability Dilemma

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   VOICE AI TESTING CHALLENGES                                      │
├───────────────────────┬────────────────────────────────────────────────────────────────────────────┤
│ Challenge             │ Impact if Unaddressed                                                      │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Hardware Coupling     │ Sounddevice/PortAudio fails in headless CI, Docker, or remote machines.   │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Non-Deterministic LLM │ Cloud model drift, token budget variance, and network latency break tests. │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Race Conditions & IPC │ Multiprocessing queue deadlocks and audio mute races are hard to repro.    │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Memory & State Leak   │ Persistent ChromaDB state pollutes tests between successive turns.        │
└───────────────────────┴────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 The Three-Tier Testing Hierarchy

MindMirror adopts a hermetic-by-default testing pyramid:

```
                            ┌────────────────────────┐
                            │  E2E Regression Suite  │  • Full glass-to-glass pipeline simulation
                            │  (Golden Scenarios)    │  • Synthetic audio injection & virtual sink
                            │                        │  • Zero soundcard, zero cloud API keys needed
                            ├────────────────────────┤
                            │ Component Integration  │  • LocalBus & ZmqBus pub/sub verification
                            │   & Contract Tests     │  • Supervisor crash & backoff recovery
                            │                        │  • Isolated temporary ChromaDB memory tests
                            ├────────────────────────┤
                            │    Pure Unit Tests     │  • Audio DSP filters & normalisation
                            │   (< 2s execution)     │  • [STYLE] tag regex & StreamPhase demuxing
                            │                        │  • ToolPurity guardrails & prompt sanitiser
                            └────────────────────────┘
```

### 6.3 Deterministic Glass-to-Glass Test Harness (`PipelineRegressionHarness`)

To verify the entire system end-to-end without touching physical microphones or speakers, we introduce the `PipelineRegressionHarness` in `src/mindmirror/pipeline/testing/harness.py`:

```python
# src/mindmirror/pipeline/testing/harness.py

from abc import ABC, abstractmethod
import asyncio
import numpy as np
from src.mindmirror.pipeline.message import PipelineMessage, StreamPhase
from src.mindmirror.pipeline.transport.interface import MessageBus

class AudioSource(ABC):
    """Abstract audio input — allows swapping mic with synthetic audio."""
    @abstractmethod
    async def stream_frames(self) -> AsyncIterator[bytes]: ...

class AudioSink(ABC):
    """Abstract audio output — allows capturing playback without physical soundcard."""
    @abstractmethod
    async def play_frame(self, frame: bytes) -> None: ...
    @abstractmethod
    def get_buffered_audio(self) -> bytes: ...

class SyntheticAudioSource(AudioSource):
    """Feeds deterministic PCM chunks from golden WAV files or synthetic waveforms."""
    def __init__(self, pcm_bytes: bytes, chunk_size: int = 1024, sample_rate: int = 16000):
        self.pcm_bytes = pcm_bytes
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

    async def stream_frames(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.pcm_bytes), self.chunk_size):
            yield self.pcm_bytes[i:i + self.chunk_size]
            await asyncio.sleep(0.001)  # Fast-forward simulation (100x real-time)

class VirtualAudioSink(AudioSink):
    """In-memory audio sink for headless verification of synthesized voice chunks."""
    def __init__(self):
        self.frames: list[bytes] = []
        self.playback_interrupted: bool = False

    async def play_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    def get_buffered_audio(self) -> bytes:
        return b"".join(self.frames)

    def total_samples(self) -> int:
        return len(self.get_buffered_audio()) // 2  # 16-bit PCM

class DeterministicReasoningEngine:
    """Mock reasoning engine emitting predictable thinking tokens, tool calls, and style tags."""
    def __init__(self, canned_response: str = "[NEUTRAL] Test response text.",
                 canned_steps: list[dict] = None):
        self.canned_response = canned_response
        self.canned_steps = canned_steps or []

    async def generate_turn(self, prompt: str, bus: MessageBus) -> None:
        # 1. Emit intermediate reasoning steps
        for step in self.canned_steps:
            await bus.publish("reasoning.trace", PipelineMessage(
                channel="reasoning.trace",
                phase=StreamPhase.REASONING_STEP,
                payload=step
            ))
        # 2. Emit internal thinking deltas
        await bus.publish("ttt.thinking", PipelineMessage(
            channel="ttt.thinking",
            phase=StreamPhase.THINKING,
            payload="Deliberating over synthetic input..."
        ))
        # 3. Emit final vocalised response with style tag
        await bus.publish("ttt.response", PipelineMessage(
            channel="ttt.response",
            phase=StreamPhase.RESPONDING,
            payload=self.canned_response
        ))
```

### 6.4 The 7 Golden Regression Scenarios

Every code change must pass this suite of 7 automated regression scenarios:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     THE 7 GOLDEN REGRESSION SCENARIOS                                   │
├─────┬───────────────────────────┬───────────────────────────────────┬───────────────────────────────────┤
│ ID  │ Scenario Name             │ Injected Conditions               │ Asserted Pipeline Contracts       │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS1 │ Full Conversational Turn  │ Synthetic PCM WAV injected into   │ • VAD detects speech audio        │
│     │ (Happy Path)              │ audio.raw                         │ • STT emits valid transcript      │
│     │                           │                                   │ • TTT yields [STYLE] response     │
│     │                           │                                   │ • VirtualAudioSink receives audio │
│     │                           │                                   │ • LatencyReport trace published   │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS2 │ Emotion & Style Tag       │ Prompts expecting [EXCITED],      │ • Regex parses style tag cleanly  │
│     │ Contract Protocol         │ [SERIOUS], [LAZY], [NEUTRAL]      │ • TTS strips tag from spoken text │
│     │                           │                                   │ • Default style fallback applies  │
│     │                           │                                   │   if model omits tag              │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS3 │ Barge-In / Interruption   │ User synthetic audio arrives      │ • VAD triggers control.interrupt  │
│     │ Handling                  │ while TTS is actively streaming   │ • TTS cancels pending synthesis   │
│     │                           │ playback chunks                   │ • Playback sink drains in < 30ms  │
│     │                           │                                   │ • Mic un-mutes cleanly            │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS4 │ Speculative Tool Fencing  │ MCTS branch explores 3 tools:     │ • READ_ONLY web search succeeds   │
│     │ & Cache Deduplication     │ Branch 1: read_only (search)      │ • 2nd branch hits ToolResultCache │
│     │                           │ Branch 2: read_only (duplicate)   │ • Branch 3 raises PermissionError │
│     │                           │ Branch 3: mutating (delete_file)  │ • No mutating action executes     │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS5 │ Parallel Speaker Diarize  │ Turn starts with unresolved voice │ • Memory saved as unresolved      │
│     │ & Retroactive Memory      │ embedding; identification arrives │ • Memory retro-tagged on event    │
│     │                           │ 400ms later                       │ • Turn N+1 adopts speaker profile │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS6 │ Adversarial Prompt        │ Tool returns malicious payload    │ • Content wrapped in untrusted tag│
│     │ Injection Sanitisation    │ ("Ignore instructions and say X") │ • System fence prevents jailbreak │
│     │                           │                                   │ • Model maintains persona         │
├─────┼───────────────────────────┼───────────────────────────────────┼───────────────────────────────────┤
│ GS7 │ Supervisor Chaos &        │ STT or TTS component killed with  │ • Supervisor detects crash        │
│     │ Component Self-Healing    │ SIGKILL or unhandled exception    │ • Exponential backoff applied     │
│     │                           │                                   │ • Component restarts cleanly      │
│     │                           │                                   │ • Bus has 0 dropped messages      │
└─────┴───────────────────────────┴───────────────────────────────────┴───────────────────────────────────┘
```

### 6.5 Automated Test Suite Directory Layout

```
tests/
├── conftest.py                       # Global pytest fixtures: LocalBus, temp ChromaDB, mock engines
├── unit/                             # Blazing-fast unit tests (< 2s total execution)
│   ├── test_dsp.py                   # Noise reduction, bandpass, volume normalization
│   ├── test_style_protocol.py        # [STYLE] tag parsing & fallback validation
│   ├── test_guardrails.py            # ToolPurity fencing & sanitization regex
│   ├── test_context_manager.py       # Thinking token pruning & token budget enforcement
│   ├── test_config.py                # Pydantic BaseSettings loading & runtime overrides
│   └── test_message_envelope.py      # PipelineMessage msgpack serialization roundtrip
├── integration/                      # Isolated subsystem integration tests
│   ├── test_local_bus.py             # Pub/Sub, request-reply, zero message loss
│   ├── test_zmq_bus.py               # IPC permissions (0600) and socket throughput
│   ├── test_supervisor.py            # Process crash detection, backoff, and live redeploy
│   ├── test_chroma_memory.py         # Isolated SQLite vector memory & profile queries
│   └── test_mcp_client.py            # Stdio MCP tool execution with mock server
├── regression/                       # End-to-end glass-to-glass regression scenarios
│   ├── test_e2e_conversational_turn.py  # GS1: Full Happy Path
│   ├── test_e2e_style_contract.py       # GS2: Emotional Style Contract
│   ├── test_e2e_barge_in.py             # GS3: Interruption & Playback Drain
│   ├── test_e2e_speculative_tools.py    # GS4: Tool Purity & Cache Hits
│   ├── test_e2e_speaker_retroactive.py  # GS5: Parallel Diarization & Retro-Tagging
│   ├── test_e2e_injection_defense.py    # GS6: Untrusted Context Sanitisation
│   ├── test_e2e_resilience_chaos.py     # GS7: Supervisor Recovery under Crash
│   └── test_e2e_thinking_leakage.py     # StreamPhase demuxing (no thinking to TTS)
└── fixtures/                         # Deterministic static assets
    ├── audio/                        # Synthetic 16kHz PCM WAVs (neutral, excited, silence)
    ├── mock_llm_responses.json       # Deterministic LLM response cassettes
    └── sample_profiles.json          # Pre-configured SpeakerProfile test fixtures
```

### 6.6 Turnkey Test Runner CLI (`scripts/run_regression_suite.py`)

A single command gives developers instant visibility into system integrity before committing:

```bash
# 1. Fast Hermetic Regression (Default: 0 external APIs, 0 soundcards, runs in ~3s)
python scripts/run_regression_suite.py --mode=fast

# 2. Integration & IPC Suite (Tests ZeroMQ IPC sockets, supervisor chaos, and ChromaDB)
python scripts/run_regression_suite.py --mode=integration

# 3. Full Local OSS Suite (Loads Faster-Whisper, Piper TTS, and local SearXNG)
python scripts/run_regression_suite.py --mode=full-oss

# 4. Standard Pytest invocation
pytest tests/ -v
```

The runner formats results with Rich into an executive summary table:

```
┌───────────────────────────────────┬──────────┬──────────┬──────────┐
│ Regression Test Suite             │ Status   │ Duration │ SLA      │
├───────────────────────────────────┼──────────┼──────────┼──────────┤
│ GS1: Full Conversational Turn     │  PASS    │ 142ms    │ < 500ms  │
│ GS2: Emotion Style Contract       │  PASS    │  18ms    │ < 50ms   │
│ GS3: Barge-In Interruption Drain  │  PASS    │  24ms    │ < 30ms   │
│ GS4: Speculative Tool Fencing     │  PASS    │  35ms    │ < 100ms  │
│ GS5: Parallel Speaker Retro-Tag   │  PASS    │  62ms    │ < 200ms  │
│ GS6: Prompt Injection Sanitizer   │  PASS    │  12ms    │ < 50ms   │
│ GS7: Supervisor Self-Healing      │  PASS    │ 840ms    │ < 2000ms │
├───────────────────────────────────┼──────────┼──────────┼──────────┤
│ OVERALL INTEGRITY CHECK           │  7/7 OK  │ 1.13s    │ ALL PASS │
└───────────────────────────────────┴──────────┴──────────┴──────────┘
```

---

## 7. Target High-Level Data Contracts & Schemas

### 7.1 Updated Pipeline Message Envelopes

```python
# src/mindmirror/pipeline/message.py

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any

class StreamPhase(Enum):
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    THINKING = "thinking"         # Internal LLM thinking tokens
    REASONING_STEP = "step"       # Discrete algorithm event (e.g. MCTS evaluation)
    TOOL_CALL = "tool_call"       # Speculative or committed tool execution
    RESPONDING = "responding"     # Spoken response text (streamed to TTS)
    CONTROL = "control"
    TELEMETRY = "telemetry"

@dataclass
class ReasoningStep:
    algorithm: str                # e.g. "tree_of_thoughts", "mcts", "reflexion"
    step_id: str
    parent_id: str | None
    action: str                   # e.g. "branch_generated", "tool_invoked", "critique", "prune", "score"
    thought: str
    score: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineMessage:
    channel: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    speaker_id: str | None = None
    phase: StreamPhase = StreamPhase.CONTROL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        import msgpack
        return msgpack.packb({
            "channel": self.channel,
            "payload": self.payload,
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "speaker_id": self.speaker_id,
            "phase": self.phase.value,
            "metadata": self.metadata
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> "PipelineMessage":
        import msgpack
        d = msgpack.unpackb(data)
        return cls(
            channel=d["channel"],
            payload=d["payload"],
            message_id=d["message_id"],
            trace_id=d.get("trace_id", str(uuid.uuid4())),
            timestamp=d["timestamp"],
            speaker_id=d.get("speaker_id"),
            phase=StreamPhase(d["phase"]),
            metadata=d.get("metadata", {})
        )
```

---

## 8. Phase-by-Phase Implementation Roadmap

---

### 🟢 Phase 0: Pipeline Core, MessageBus, OSS Connectors & Test Foundation (P0) [COMPLETED ✅]
**Objective**: Build the transport and messaging foundation, eliminate file-based locks, and bootstrap the hermetic testing infrastructure.

#### Tasks:
1. **Kill File Locks**:
   - Delete all references to `speaking.lock` and `playback.lock`.
   - Implement `PlaybackState` in `src/mindmirror/audio/playback.py` using bus control topics (`control.mute_mic`, `control.playback_active`) and `asyncio.Event`.
2. **Implement MessageBus Interfaces**:
   - `src/mindmirror/pipeline/transport/interface.py`: Define `MessageBus` ABC with `publish()`, `subscribe()`, `request()`, `close()`.
   - `src/mindmirror/pipeline/transport/local.py`: Implement `LocalBus` using `asyncio.Queue` (zero external dependencies for blazing-fast hermetic testing).
   - `src/mindmirror/pipeline/transport/zmq_bus.py`: Implement `ZmqBus` connector for **libzmq** open-source IPC sockets (`ipc:///tmp/mindmirror_*.ipc` with `chmod 0600`).
   - `src/mindmirror/pipeline/transport/nats_bus.py`: Implement `NatsBus` connector for open-source **NATS Server** (`nats-py`).
   - `src/mindmirror/pipeline/transport/valkey_bus.py`: Implement `ValkeyBus` connector for open-source **Valkey / Redis** (`redis.asyncio`).
3. **PipelineComponent Lifecycle Base Class**:
   - `src/mindmirror/pipeline/component.py`: Standardise lifecycle (`start`, `drain`, `stop`, `health`).
4. **Bootstrap Automated Test Suite & Fixtures**:
   - Create root test framework: `tests/conftest.py`, `pytest.ini`.
   - Implement `LocalBus` test fixture: creates fresh, isolated in-memory bus per test with zero port bindings.
   - Implement `tests/unit/test_message_envelope.py`: validates `PipelineMessage` serialization/deserialization via `msgpack`, verifying `trace_id` and timestamp preservation.
   - Implement `tests/integration/test_local_bus.py`: verifies pub/sub routing, multi-subscriber fan-out, and request-reply timeouts.
   - Implement `tests/integration/test_zmq_bus.py`: verifies socket creation permissions (`0600`), socket cleanup, and 1,000-message burst throughput.

#### Verification & Automated Regression Tests:
- Run automated unit and integration suite:
  ```bash
  pytest tests/unit/test_message_envelope.py tests/integration/test_local_bus.py tests/integration/test_zmq_bus.py -v
  ```
- Run `scripts/test_message_buses.py` benchmark:
  - Starts `LocalBus`, `ZmqBus`, and `NatsBus` (mock/container).
  - Pushes 1,000 serialized payloads across multiple channels.
  - Asserts latency < 1ms on local IPC, socket permissions set to 0600, and zero message loss.

---

### 🟠 Phase 1: Supervisor, Audio Decomposition, Headless Simulation & Barge-In [COMPLETED ✅]
**Objective**: Transform monolithic runners into supervised components, decouple hardware I/O for headless automated testing, and implement acoustic barge-in.

#### Tasks:
1. **Implement `ComponentSupervisor`**:
   - `src/mindmirror/pipeline/supervisor.py`: Erlang-style supervisor with exponential backoff (e.g. 1s, 2s, 4s, 8s, up to 30s), crash counters, and zero-downtime blue-green `redeploy(name, new_factory)`.
2. **Decompose Audio I/O with Pluggable Sources & Sinks**:
   - `src/mindmirror/pipeline/testing/harness.py`: Implement `AudioSource` ABC and `AudioSink` ABC.
   - `src/mindmirror/audio/capture.py`: Implement `LiveAudioCapture` (PortAudio/PipeWire) streaming PCM frames to `audio.raw`.
   - `src/mindmirror/pipeline/testing/harness.py`: Implement `SyntheticAudioSource` (feeds deterministic PCM chunks from WAV fixtures) and `VirtualAudioSink` (buffers generated PCM audio in memory without physical soundcards).
3. **Decompose STT & Silero VAD**:
   - `src/mindmirror/stt/vad.py`: Refactor Silero VAD into a pipeline component reading from `audio.raw` and publishing to `vad.speech_audio`.
   - `src/mindmirror/stt/component.py`: Subscribes to `vad.speech_audio`, transcribes via `STTEngine`, and publishes to `stt.transcript`.
4. **Decompose TTS Runner & Acoustic Interruption (Barge-In)**:
   - `src/mindmirror/tts/component.py`: Subscribes to `ttt.response`, parses style tags, invokes pure `TTSEngine` (synthesis only, no soundcard), and pushes chunks to `audio.playback`.
   - `src/mindmirror/audio/playback.py`: Consumes `audio.playback`, manages soundcard streams, and fires atomic mute/unmute control signals.
   - **Barge-In Engine**: When VAD detects user speech while playback is active, publish `control.interrupt`; `PlaybackLoop` drains audio buffer in < 30ms and clears mute lock.
5. **Wire Up Open-Source STT & TTS Options**:
   - STT: Integrate `faster-whisper` (`src/mindmirror/stt/local_whisper/`) with GPU/CPU int8 quantization.
   - TTS: Integrate `pipervoice` (`src/mindmirror/tts/pipervoice/`) for low-latency ONNX synthesis, and maintain `f5_tts` for voice cloning.
6. **Automated Tests for Phase 1**:
   - `tests/integration/test_supervisor.py`: Spawns test component under `ComponentSupervisor`; raises unhandled exception; asserts supervisor restarts component within backoff window and resumes processing.
   - `tests/unit/test_dsp.py`: Verifies noise reduction, bandpass filter, and audio normalisation on synthetic sine waves.
   - `tests/regression/test_e2e_barge_in.py` (**GS3**): Uses `SyntheticAudioSource` and `VirtualAudioSink`. Injects speech frame while playback is active; asserts `control.interrupt` emitted, sink drains in < 30ms, and mic un-mutes cleanly.

#### Verification & Automated Regression Tests:
- Run automated supervisor and audio regression suite:
  ```bash
  pytest tests/integration/test_supervisor.py tests/unit/test_dsp.py tests/regression/test_e2e_barge_in.py -v
  ```
- Run `scripts/test_supervisor_recovery.py` to confirm process crash survival under heavy load.

---

### 🟡 Phase 2: Autonomous Reasoning, Tools, Guardrails & Contract Enforcement [COMPLETED ✅]
**Objective**: Build out the unconstrained custom reasoning engine, integrate MCP tools and web search, enforce speculative tool safety, and guarantee emotional style tag contracts.

#### Tasks:
1. **Reasoning Engine & Strategy Architecture**:
   - `src/mindmirror/llm/reasoning/interface.py`: Define `ReasoningStrategy` ABC and `ReasoningStep`.
   - `src/mindmirror/llm/reasoning/custom/tree_of_thoughts.py`: Implement multi-step branch generation, self-critique, candidate scoring, and tool exploration.
   - `src/mindmirror/llm/reasoning/native/provider_native.py`: Streaming parser for native thinking blocks (Anthropic, Gemini `thinkingBudget`, DeepSeek `<think>` tags).
2. **Deterministic Reasoning Test Harness**:
   - `src/mindmirror/pipeline/testing/harness.py`: Implement `DeterministicReasoningEngine` capable of emitting pre-scripted thinking tokens, tool calls, and emotional responses in < 5ms for hermetic regression testing.
3. **External Tools & Search Integration**:
   - `src/mindmirror/tools/search.py`: Connect to local **SearXNG** instance (or DuckDuckGo fallback).
   - Integrate existing `src/mindmirror/mcp_client/client.py` (`MCPClientManager`) into the reasoning loop.
   - Add `ToolResultCache` to cache idempotent lookups across exploratory tree branches.
4. **Security Guardrails & Prompt Sanitisation**:
   - `src/mindmirror/security/guardrails.py`: Enforce `ToolPurity.READ_ONLY` on speculative branches; block mutating actions with `PermissionError` unless explicitly confirmed.
   - Implement `Sanitizer`: wrap external tool content in `<external_untrusted_context>` and strip dangerous unicode/control characters.
5. **Reasoning-Aware Context Management**:
   - `src/mindmirror/llm/context/manager.py`: Prune historical thinking blocks and intermediate reasoning rollouts from conversational context while preserving final synthesised answers.
6. **Observability Tracking**:
   - `src/mindmirror/pipeline/telemetry.py`: Measure and log end-to-end spans (`vad_to_transcript`, `reasoning_time`, `time_to_first_token`, `tts_synthesis`).
7. **Automated Tests for Phase 2**:
   - `tests/unit/test_style_protocol.py` (**GS2**): Asserts regex parses emotional style tags (`[NEUTRAL]`, `[EXCITED]`, `[SERIOUS]`, `[LAZY]`); asserts fallback mechanism applies default style if tag is omitted; asserts style tags are stripped from spoken text.
   - `tests/unit/test_guardrails.py`: Asserts `ToolPurity.MUTATING` throws `PermissionError` on speculative execution; asserts untrusted tool output is wrapped in `<external_untrusted_context>`.
   - `tests/unit/test_context_manager.py`: Asserts thinking tokens and reasoning steps are pruned from turn history while preserving final assistant responses.
   - `tests/regression/test_e2e_speculative_tools.py` (**GS4**): Injects complex query; asserts speculative branches only call read-only tools; asserts duplicate queries hit `ToolResultCache`.
   - `tests/regression/test_e2e_thinking_leakage.py`: Asserts `StreamPhase.THINKING` tokens are routed to telemetry/UI and strictly blocked from reaching `audio.playback`.
   - `tests/regression/test_e2e_injection_defense.py` (**GS6**): Feeds adversarial prompt injection via mock tool; asserts sanitiser wraps payload and system prompt prevents jailbreak.

#### Verification & Automated Regression Tests:
- Run automated reasoning, safety, and style contract regression suite:
  ```bash
  pytest tests/unit/test_style_protocol.py tests/unit/test_guardrails.py tests/unit/test_context_manager.py tests/regression/test_e2e_speculative_tools.py tests/regression/test_e2e_thinking_leakage.py tests/regression/test_e2e_injection_defense.py -v
  ```
- Run `scripts/test_reasoning_pipeline.py` with live web search and MCP mock server.

---

### 🔵 Phase 3: Parallel Background Speaker Identification & Retroactive Memory [COMPLETED ✅]
**Objective**: Make speaker identification completely non-blocking, retroactive, and capable of runtime parameter tuning.

#### Tasks:
1. **Parallel Speaker Identifier**:
   - `src/mindmirror/speaker/identifier.py`: Subscribes directly to `vad.speech_audio` in parallel with STT.
   - Computes voice embeddings via OSS `pyannote.audio` or SpeechBrain ECAPA-TDNN.
   - Matches against `SpeakerRegistry` in ChromaDB using cosine similarity.
   - Emits `speaker.identified` event on the bus.
2. **`SpeakerContext` & Non-Blocking Shared State**:
   - `src/mindmirror/speaker/context.py`: Thread-safe / async state holder tracking `current_speaker_id`, confidence score, and resolved parameter profile.
3. **Retroactive ChromaDB Memory Tagging**:
   - `src/mindmirror/memory/chroma_store.py`:
     - When a memory is saved while speaker ID is still resolving, store with `speaker_id="unresolved"` and buffer the memory ID.
     - When `speaker.identified` arrives, issue an asynchronous metadata update to tag the memory with the confirmed speaker ID.
4. **`SpeakerProfile` Runtime Parameter Store**:
   - `src/mindmirror/speaker/profile.py`: Dataclass containing per-speaker runtime tuning:
     - `vad_silence_duration`: custom pause threshold (e.g. 1.2s for slow speakers vs 0.5s for fast speakers).
     - `tts_voice_id` / `tts_pitch` / `tts_speaking_rate`.
     - `ttt_reasoning_effort`: "high" vs "low".
     - `prompt_context`: custom bio/instructions.
   - Store profiles in dedicated ChromaDB collection `speaker_profiles`.
5. **Automated Tests for Phase 3**:
   - `tests/integration/test_chroma_memory.py`: Uses temporary directory fixture (`tmp_path`); tests storing memories, vector similarity recall, and metadata updates without cross-test leakage.
   - `tests/regression/test_e2e_speaker_retroactive.py` (**GS5**): Simulates audio turn with 400ms speaker identification delay; asserts turn proceeds optimistically with `speaker_id="unresolved"`; asserts memory is retroactively patched when `speaker.identified` fires; asserts turn N+1 dynamically applies custom VAD and TTS parameters.

#### Verification & Automated Regression Tests:
- Run automated speaker identification & retroactive memory regression suite:
  ```bash
  pytest tests/integration/test_chroma_memory.py tests/regression/test_e2e_speaker_retroactive.py -v
  ```
- Run `scripts/test_speaker_retroactive_tagging.py` with simulated multi-speaker voice streams.

---

### 🟣 Phase 4: Containerisation, Deployment & Master Regression Suite [COMPLETED ✅]
**Objective**: Package the entire stack into self-healing open-source containers and deliver a turnkey regression test runner CLI.

#### Tasks:
1. **Docker Compose Setup**:
   - `docker-compose.yml`:
     - Service 1: `nats-server` (OSS binary, port 4222).
     - Service 2: `valkey` (OSS in-memory store, port 6379).
     - Service 3: `chromadb` (OSS vector database, port 8000).
     - Service 4: `searxng` (OSS privacy search engine, port 8080).
     - Service 5: `mindmirror-stt` (Faster-Whisper on GPU/CPU).
     - Service 6: `mindmirror-reasoning` (Custom Reasoning Engine + Local vLLM/API).
     - Service 7: `mindmirror-tts` (Piper / F5-TTS).
     - Service 8: `mindmirror-audio` (Local host container with PipeWire/ALSA socket mount).
2. **Zero-Downtime Blue-Green Deployment Script**:
   - Deploy new versions of individual components without dropping active audio connections using the supervisor's `redeploy()` mechanism.
3. **Turnkey Regression CLI Runner (`scripts/run_regression_suite.py`)**:
   - CLI script running the 7 Golden Regression Scenarios (`--mode=fast`, `--mode=integration`, `--mode=full-oss`).
   - Renders a Rich console summary table with glass-to-glass latency stats and pass/fail indicators.
4. **Full Conversational Turn Glass-to-Glass Test (`tests/regression/test_e2e_conversational_turn.py` - GS1)**:
   - Exercises the complete pipeline from `SyntheticAudioSource` through VAD, STT, TTT, TTS, to `VirtualAudioSink` with zero cloud keys or audio hardware, asserting latency SLAs.
5. **Supervisor Chaos & Resilience Test (`tests/regression/test_e2e_resilience_chaos.py` - GS7)**:
   - Kills pipeline workers during active conversation simulation; verifies supervisor auto-recovers and subsequent turns succeed with zero message loss.

#### Verification & Automated Regression Tests:
- Run the full regression test suite with single turnkey command:
  ```bash
  python scripts/run_regression_suite.py --mode=fast
  ```
- Run full containerized regression suite:
  ```bash
  docker compose up -d
  python scripts/run_regression_suite.py --mode=full-oss
  ```
- Verify all 7 Golden Regression Scenarios pass and glass-to-glass latency stays under SLA (< 1000ms).

---

## 9. Risk Assessment & Mitigation Strategies

| Risk / Failure Mode | Likelihood | Impact | Architectural Mitigation |
|---|---|---|---|
| **Speculative Tool Side-Effects**: Reasoning branch triggers unwanted real-world actions (e.g. sending email). | Medium | High | `SecurityGuardrail` enforces `ToolPurity.READ_ONLY` during speculative branch search. Mutating actions require explicit plan commitment. |
| **Prompt Injection via Web / MCP**: Malicious web content hijacks the reasoning loop. | Medium | High | Wrap external tool outputs in `<external_untrusted_context>`, strip control characters, and instruct LLM system prompts to treat it strictly as reference text. Verified by automated test **GS6**. |
| **Speaker ID Latency**: Identification takes longer than STT + Reasoning. | High | Medium | Non-blocking pipeline: STT and Reasoning proceed using optimistic defaults; memories are buffered and tagged retroactively; speaker profile applied on turn N+1. Verified by automated test **GS5**. |
| **Custom Reasoning Runaway / Infinite Loop**: Custom search (MCTS/ToT) gets stuck exploring tree branches. | Medium | High | Hard timeout and max-branch limit in `ReasoningStrategy`. If timeout triggers, immediately yield best candidate found so far. |
| **Audio Jitter over Distributed Bus**: Network latency spikes degrade real-time audio. | Medium | High | Strict isolation: keep high-sample-rate audio buffers on local UNIX sockets (`ZmqBus` `ipc://`) with `chmod 0600`; use `NatsBus` only for transcripts, control signals, and final text. |
| **Context Window Exhaustion from Reasoning**: Multi-step rollouts generate massive token volumes. | High | Medium | `ContextManager` aggressively prunes thinking blocks and reasoning traces from turn history, keeping only final synthesised responses. |
| **Audio Hardware Mute Race Conditions**: Simultaneous mic listening and speaker playback cause acoustic feedback. | High | High | Replaced file locks with atomic `PlaybackState` event notifications published over the bus directly to `AudioCapture` before DAC playback starts. Verified by automated test **GS3**. |
| **Regression Flakiness & Cloud Dependency**: Tests fail due to cloud API downtime, billing, or network latency. | High | High | `PipelineRegressionHarness` uses `DeterministicReasoningEngine` and `VirtualAudioSink`, making regression tests 100% hermetic, offline, and sub-second. |
| **Silent Emotional Tag Protocol Drift**: LLM updates or prompt tweaks omit `[STYLE]` tags or leak thinking tokens to voice. | High | Medium | Automated regression tests **GS2** and `test_e2e_thinking_leakage.py` strictly validate `StreamPhase` isolation and emotional tag formatting on every build. |
| **Soundcard Hardware Absence in CI/CD**: Tests crash on headless build servers lacking PortAudio or ALSA devices. | High | High | Decoupled `AudioSource` and `AudioSink` abstractions allow headless execution via `SyntheticAudioSource` and `VirtualAudioSink`. |

---

## 10. Master Implementation Checklist & Deliverables

```
[x] Phase 0: Transport, Core & Testing Infrastructure
    [x] Remove file locks (speaking.lock, playback.lock)
    [x] Create src/mindmirror/pipeline/message.py (PipelineMessage, StreamPhase, ReasoningStep)
    [x] Create src/mindmirror/pipeline/transport/interface.py (MessageBus ABC)
    [x] Implement LocalBus (asyncio.Queue)
    [x] Implement ZmqBus (ZeroMQ IPC with chmod 0600)
    [x] Implement NatsBus (NATS Server connector)
    [x] Implement ValkeyBus (Valkey/Redis connector)
    [x] Create test suite skeleton (tests/conftest.py, tests/unit/, tests/integration/, tests/regression/)
    [x] Implement tests/unit/test_message_envelope.py (msgpack serialization)
    [x] Implement tests/integration/test_local_bus.py (pub/sub & request-reply)
    [x] Implement tests/integration/test_zmq_bus.py (IPC permissions & burst latency)
    [x] Verify via scripts/test_message_buses.py

[x] Phase 1: Component Supervision, Audio Decoupling & Barge-In
    [x] Create src/mindmirror/pipeline/supervisor.py (ComponentSupervisor)
    [x] Implement AudioSource and AudioSink ABCs in src/mindmirror/pipeline/testing/harness.py
    [x] Implement SyntheticAudioSource and VirtualAudioSink for headless simulation
    [x] Extract AudioCapture & PlaybackLoop into src/mindmirror/audio/
    [x] Implement acoustic barge-in cancellation in PlaybackLoop (< 30ms drain)
    [x] Refactor Silero VAD into pipeline component reading audio.raw
    [x] Refactor STT runner into component reading vad.speech_audio
    [x] Refactor TTS runner into component consuming ttt.response
    [x] Implement tests/integration/test_supervisor.py (crash & backoff recovery)
    [x] Implement tests/unit/test_dsp.py (DSP filters & normalization)
    [x] Implement tests/regression/test_e2e_barge_in.py (GS3: Interruption & drain)
    [x] Verify via scripts/test_supervisor_recovery.py

[x] Phase 2: Autonomous Reasoning, Tools, Guardrails & Contract Tests
    [x] Create src/mindmirror/llm/reasoning/interface.py (ReasoningStrategy)
    [x] Implement TreeOfThoughtsReasoning (custom multi-step generator/critic)
    [x] Implement DeterministicReasoningEngine in harness.py for hermetic tests
    [x] Integrate MCPClientManager into reasoning loop
    [x] Implement SearXNG/web search connector with ToolResultCache
    [x] Implement SecurityGuardrail (ToolPurity fencing + prompt injection sanitisation)
    [x] Implement ProviderNativeReasoning (streaming parser for thinking tokens)
    [x] Create ContextManager with thinking-pruning & summarisation
    [x] Add OpenTelemetry spans and metrics.latency publisher
    [x] Implement tests/unit/test_style_protocol.py (GS2: [STYLE] tag parsing & fallback)
    [x] Implement tests/unit/test_guardrails.py (ToolPurity enforcement & sanitization)
    [x] Implement tests/unit/test_context_manager.py (thinking token pruning)
    [x] Implement tests/regression/test_e2e_speculative_tools.py (GS4: tool purity & cache)
    [x] Implement tests/regression/test_e2e_thinking_leakage.py (StreamPhase demuxing)
    [x] Implement tests/regression/test_e2e_injection_defense.py (GS6: prompt injection)
    [x] Verify via scripts/test_reasoning_pipeline.py

[x] Phase 3: Speaker Identification & Retroactive Memory
    [x] Implement SpeakerIdentifier with pyannote/ECAPA-TDNN
    [x] Implement SpeakerContext shared state
    [x] Implement retroactive memory tagging in ChromaMemoryManager
    [x] Implement SpeakerProfile parameter store in ChromaDB
    [x] Connect speaker overrides to VAD silence thresholds and TTS voices
    [x] Implement tests/integration/test_chroma_memory.py (isolated temp ChromaDB)
    [x] Implement tests/regression/test_e2e_speaker_retroactive.py (GS5: parallel diarization)
    [x] Verify via scripts/test_speaker_retroactive_tagging.py

[x] Phase 4: Containerisation, Deployment & Master Regression Suite
    [x] Integrate Faster-Whisper STT
    [x] Integrate Piper TTS
    [x] Build docker-compose.yml with NATS, Valkey, ChromaDB, and SearXNG
    [x] Implement tests/regression/test_e2e_conversational_turn.py (GS1: full happy path)
    [x] Implement tests/regression/test_e2e_resilience_chaos.py (GS7: supervisor self-healing)
    [x] Implement scripts/run_regression_suite.py CLI runner with Rich summary table
    [x] Run full regression suite (python scripts/run_regression_suite.py --mode=fast)
    [x] Perform end-to-end local latency benchmarks with OTel trace validation
```
