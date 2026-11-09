# AGENTS.md — MindMirror Project Guidelines & Memory

This file serves as operational memory and architectural rules for AI agents and developers working on **MindMirror**.

---

## 🏗️ 1. Architecture & Pipeline Contracts

- **Pipeline Flow**: `Audio Capture` ➡️ `VAD` ➡️ `STT (Speech-to-Text)` & `Parallel Speaker ID` ➡️ `TTT (Text-to-Thought / Reasoning)` ➡️ `TTS (Text-to-Speech)` ➡️ `Playback Sink`.
- **Dual Execution Modes**:
  - **Standalone Multiprocessing Mode ([src/mindmirror/main.py](src/mindmirror/main.py))**: Each module runs as an independent `multiprocessing.Process` communicating via `multiprocessing.Queue` instances (`stt_queue`, `ttt_queue`, `control_queue`, `log_queue`) and cross-process playback locks.
  - **Supervised MessageBus Mode (`src/mindmirror/*/service.py`)**: Components inherit from `PipelineComponent`, run under the Erlang-style `ComponentSupervisor` (supporting crash recovery and blue-green hot swapping), and communicate over pluggable transports (`LocalBus`, `ZmqBus`, `NatsBus`, `ValkeyBus`).
- **Style Tag Protocol**: 
  - All TTT (LLM) responses **must** start with an emotional style tag: `[NEUTRAL]`, `[EXCITED]`, `[SERIOUS]`, or `[LAZY]`.
  - Example: `[EXCITED] System checks passed! We are ready to roll.`
  - The TTS engine uses these tags to select appropriate emotion reference audio clips and pitch settings dynamically.

---

## ⚡ 2. Environment & Execution Guidelines

- **Environment**: All commands and tests must run in the `.venv` virtual environment (Python 3.12.10 via pyenv):
  ```bash
  /home/mario/1_study/repository/mindmirror/.venv/bin/python <script>
  ```
- **Dependencies**: Keep [requirements.txt](requirements.txt) clean and grouped by category (STT, TTT/LangChain, TTS, Audio DSP).
- **Credentials & Security**:
  - Never commit `.env` or Service Account JSON files (`GOOGLE_APPLICATION_CREDENTIALS`).
  - Access configuration values exclusively via `mindmirror.config`.

---

## 🧪 3. Testing & Verification Standards

- **Never declare success without verification**: Run relevant test scripts in `scripts/` before marking tasks complete:
  - `python scripts/run_regression_suite.py --mode=fast`: Run the 7 Golden Regression Scenarios (hermetic, zero-hardware).
  - `python scripts/run_regression_suite.py --mode=full-oss`: Run full regression, integration, and unit test suites.
  - `python scripts/deploy_component.py`: Verify zero-downtime blue-green component hot-swapping.
  - `python scripts/test_langchain_ttt.py`: Test LangChain TTT / DeepAgents generation and style parsing.
  - `python scripts/test_google_cloud_tts.py`: Test Google Cloud TTS synthesis.
  - `python scripts/test_inference.py`: Test F5-TTS audio inference styles.
- **Documentation Sync**: When updating core logic or configurations, update [README.md](README.md) and module-specific `README.md` files ([src/mindmirror/stt/README.md](src/mindmirror/stt/README.md), [src/mindmirror/llm/README.md](src/mindmirror/llm/README.md), [src/mindmirror/tts/README.md](src/mindmirror/tts/README.md)).
- **Relative Paths in Documentation**: Always use relative file paths (e.g. [data/AGENTS.md](data/AGENTS.md)) in repository Markdown files (`AGENTS.md`, `README.md`, module READMEs). Never hardcode absolute file system URLs (`file:///home/<username>/...`).

---

## 🛠️ 4. Code Principles & Latency Rules

- **Non-blocking Execution**: Ensure audio loops and queue processing remain asynchronous and low-latency to keep real-time voice conversations responsive.
- **Vendor-Agnostic Tools**: MCP tool integrations (`MCPClientManager`) must remain decoupled from specific LLM providers so we can swap backends seamlessly.

---

## 🧠 5. Persistent Vector Memory (ChromaDB) & Runtime Agent Files

- **Developer Memory vs. Runtime Agent Memory**:
  - Root `AGENTS.md` is exclusively for **developer guidelines and architectural rules** (for AI coding assistants).
  - The runtime agent's instructions, persistent skills, and ChromaDB vector store live in the **`data/` directory**:
    - Runtime System Instructions & Skills: [data/AGENTS.md](data/AGENTS.md)
    - Persistent Vector Database: `data/chroma_db/`
- **Memory Backend**: Managed via `ChromaMemoryManager` in `src/mindmirror/memory/chroma_store.py`.
- **Memory Tools**: `remember_fact` (store) and `recall_memories` (vector search).
