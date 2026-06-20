# Text-to-Thought (TTT / LLM) Module

The Text-to-Thought module forms the conversational brain of the assistant. It accepts text transcripts from the Speech-to-Text module, processes them using a Large Language Model (LLM) with tool access via Model-Context Protocol (MCP) servers, and outputs response blocks that are synthesised to audio by the Text-to-Speech (TTS) module.

The default TTT interface is implemented using **Google Gemini** models.

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
*   `GOOGLE_TTT_MODEL` (e.g. `gemini-2.5-flash-lite`): The model version used for text generation. For latency-sensitive voice loops, lightweight models (like Flash-Lite) are recommended.

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
