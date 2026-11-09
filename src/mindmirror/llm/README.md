# Text-to-Thought (TTT / LLM) Module

The Text-to-Thought module forms the conversational brain of the assistant. It accepts text transcripts from the Speech-to-Text module, processes them using a Large Language Model (LLM) with tool access via Model-Context Protocol (MCP) servers, and outputs response blocks that are synthesised to audio by the Text-to-Speech (TTS) module.

The default TTT interface is implemented using **LangChain & DeepAgents** (`LangChainLLMClient`) supporting Google Gemini backends (`vertexai` and `google`) as well as the direct **Google Gemini** SDK (`GeminiLLMClient`).

## Interface Definition

Any Text-to-Thought client implementation must adhere to the contract defined in [interface.py](interface.py):

*   **`async def init_chat(self) -> None`**: Sets up system instructions, maps MCP tool definitions to GenAI tool declarations, and establishes the async chat session.
*   **`async def send_message(self, text: str) -> str`**: Sends the text query to the LLM. It manages the function/tool execution loops asynchronously, returning the final text answer once all function calls have resolved.

---

## System Prompt and Style-Tag Protocol

The assistant uses a specialized system prompt configured in [main.py](../main.py). The LLM is instructed to prepend style tags to its responses:
*   `[NEUTRAL]`: General information, standard tone.
*   `[EXCITED]`: Success, good news, or high-energy output.
*   `[SERIOUS]`: Warnings, errors, or critical updates.
*   `[LAZY]`: Casual confirmations or when unsure.

The parser matches these tags and segments the output into tuples of `(style, text)`, allowing the TTS engine to adjust its speech pacing, pitch, and voice models to match the assistant's mood dynamically.

---

## Configuration and Parameterisation

### 1. Environment Variables (`.env`)
To use Google Gemini, you must configure Google Cloud authentication and project parameters:

```env
# Required: Google Application Credentials path and project parameters
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Optional: Hugging Face API key
HF_TOKEN=your_hugging_face_token_here
```

### 2. LLM Settings
You can customize the model used in [config.py](../config.py):
*   `LANGCHAIN_TTT_PROVIDER` (`vertexai` | `google`): Selects the backend provider wrapper (defaults to `vertexai`).
*   `LANGCHAIN_TTT_MODEL` (e.g. `gemini-2.5-flash-lite`): Model selection for text generation.
*   `LANGCHAIN_TEMPERATURE` (e.g. `0.7`): Model temperature setting.
*   `GOOGLE_TTT_MODEL` (e.g. `gemini-2.5-flash-lite`): Fallback model version used when running direct Gemini SDK client (`GeminiLLMClient`).

### 3. Model-Context Protocol (MCP) Integration
The assistant can be extended with tools provided by MCP servers. You can define servers in [config.py](../config.py):

```python
# MCP Server list configuration
MCP_SERVERS = [
    {
        "name": "fraud-detection",
        "type": "sse",
        "url": os.getenv("MCP_FRAUD_SSE_URL", "http://localhost:8088/sse")
    }
]
```

When initialized:
1.  The assistant connects to all configured MCP servers.
2.  It discovers available tools and converts their schemas to Google GenAI types.
3.  When the model issues a tool call request, `GeminiLLMClient` executes the callback, performs the operation, and feeds the output back into the model's active chat context.

---

### 4. Persistent Vector Memory (ChromaDB) & Memory Skill
The agent incorporates long-term speaker-partitioned vector memory using **ChromaDB** ([chroma_store.py](../memory/chroma_store.py) & [chroma_client.py](../memory/chroma_client.py)) stored locally in `data/chroma_db/` (or via a remote ChromaDB server when `CHROMA_HOST` is reachable, with automatic local fallback):
*   `remember_fact(fact, category, speaker_id)`: Stores user preferences, facts, and key details into vector memory. Unresolved speakers are buffered and retroactively tagged when `speaker.identified` fires on the message bus.
*   `recall_memories(query, limit, speaker_id)`: Performs similarity search over stored memories, optionally filtered by `speaker_id`.
*   **Persistent Memory Skill**: Defined in [data/skills/persistent-memory/SKILL.md](../../../data/skills/persistent-memory/SKILL.md) and loaded via DeepAgents' `SkillsMiddleware`.

---

## Supervised Reasoning Architecture & Pluggable Strategies

For distributed and message-bus deployments, the LLM subsystem provides a supervised `ReasoningComponent` ([component.py](component.py)) and microservice runner ([service.py](service.py)):

*   **`ReasoningComponent`** ([component.py](component.py)): Subscribes to `stt.transcript`, updates sliding-window conversation history via `ContextManager` ([context/manager.py](context/manager.py)), executes the active `ReasoningStrategy`, prunes internal deliberation traces after each turn, and publishes glass-to-glass latency telemetry via `TurnSpanTracker`.
*   **`ReasoningStrategyRegistry`** ([reasoning/registry.py](reasoning/registry.py)): Plug-and-play strategy registry supporting:
    *   `tree_of_thoughts` (`TreeOfThoughtsStrategy` in [reasoning/custom/tree_of_thoughts.py](reasoning/custom/tree_of_thoughts.py)): Multi-branch deliberation with speculative read-only tool execution (`SecurityGuardrail`) and two-tier result caching (`ToolResultCache`).
    *   `symbolic_poc` (`SymbolicHeuristicReasoningStrategy` in [reasoning/custom/poc_symbolic_reasoner.py](reasoning/custom/poc_symbolic_reasoner.py)): 3-stage symbolic intent decomposition and heuristic candidate selection.
    *   `native_thinking` (`ProviderNativeReasoningStrategy` in [reasoning/native/provider_native.py](reasoning/native/provider_native.py)): Demuxes `<think>...</think>` blocks onto `ttt.thinking` (`StreamPhase.THINKING`) while routing only style-tagged spoken text onto `ttt.response` (`StreamPhase.RESPONDING`).
    *   `local_llm` (`LocalLLMReasoningStrategy` in [reasoning/local_llm.py](reasoning/local_llm.py)): Connects to local Ollama / vLLM / `llama.cpp` endpoints.
*   **Microservice Entrypoint** ([service.py](service.py)):
    ```bash
    python -m mindmirror.llm.service
    ```
