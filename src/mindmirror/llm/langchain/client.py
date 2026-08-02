import os
import json
import asyncio
import warnings
import logging
from pathlib import Path
from typing import Any, List, Dict, Callable, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import StructuredTool
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from mindmirror.llm.interface import TTTInterface
from mindmirror import config
from mindmirror.memory import ChromaMemoryManager, get_memory_tools

# Suppress google SDK verbose schema notices
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.cloud").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

def clean_schema(schema: Any) -> Any:
    """Recursively remove keys unsupported by Vertex AI / Gemini API from schema dictionary."""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in ("additionalProperties", "additional_properties", "exclusiveMinimum", "exclusiveMaximum", "$schema"):
                continue
            cleaned[k] = clean_schema(v)
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    return schema

def sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Ensure no message in history has empty content/parts to avoid Vertex AI 400 errors."""
    sanitized = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            if (isinstance(msg.content, str) and not msg.content.strip()) and not getattr(msg, 'tool_calls', None):
                msg.content = "..."
        elif isinstance(msg, HumanMessage):
            if isinstance(msg.content, str) and not msg.content.strip():
                msg.content = "..."
        elif isinstance(msg, ToolMessage):
            if isinstance(msg.content, str) and not msg.content.strip():
                msg.content = "Tool execution completed."
        sanitized.append(msg)
    return sanitized

class LangChainLLMClient(TTTInterface):
    """
    LangChain & DeepAgents-backed TTT (Text-to-Thought) client.
    Uses deepagents.create_deep_agent with ChatVertexAI / ChatGoogleGenerativeAI
    while handling MCP tool executions and conversation state seamlessly.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        execute_tool_callback: Optional[Callable] = None,
        log_queue: Optional[Any] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        skills: Optional[List[str]] = None,
        memory_files: Optional[List[str]] = None
    ):
        self.model_name = model_name or getattr(config, 'LANGCHAIN_TTT_MODEL', 'gemini-2.5-flash-lite')
        self.provider = (provider or getattr(config, 'LANGCHAIN_TTT_PROVIDER', 'google')).lower()
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.execute_tool = execute_tool_callback
        self.log_queue = log_queue
        self.temperature = temperature
        
        if skills is not None:
            self.skills = skills
        else:
            skills_dir = getattr(config, 'SKILLS_DIR', config.PROJECT_ROOT / 'data' / 'skills')
            skills_path = Path(skills_dir)
            if skills_path.exists():
                try:
                    rel_path = str(skills_path.relative_to(config.PROJECT_ROOT)).rstrip('/') + '/'
                except ValueError:
                    rel_path = str(skills_path).rstrip('/') + '/'
                self.skills = [rel_path]
            else:
                self.skills = []

        if memory_files is not None:
            self.memory_files = memory_files
        else:
            agents_md = config.PROJECT_ROOT / 'data' / 'AGENTS.md'
            if agents_md.exists():
                self.memory_files = [str(agents_md.relative_to(config.PROJECT_ROOT))]
            else:
                self.memory_files = []

        if self.skills:
            print("\n🧠 [SKILLS CONFIG] Initializing DeepAgent with skills directories:", self.skills)
            for s_dir in self.skills:
                p = Path(s_dir)
                if p.exists():
                    skill_files = list(p.glob("**/*.md"))
                    print(f"   📂 Found {len(skill_files)} skill definition file(s) in '{s_dir}':")
                    for sf in skill_files:
                        try:
                            rel_path = sf.relative_to(config.PROJECT_ROOT)
                        except ValueError:
                            rel_path = sf
                        print(f"      📄 [SKILL] {rel_path}")
                else:
                    print(f"   ⚠️ Skills directory does not exist: '{s_dir}'")

        # Filter schema notices
        warnings.filterwarnings("ignore", message=".*is not supported in schema.*")

        # Ensure environment API key is set for ChatGoogleGenerativeAI
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
            os.environ["GEMINI_API_KEY"] = gemini_key

        key_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        if key_path and os.path.exists(key_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        project = getattr(config, 'GOOGLE_CLOUD_PROJECT', None)
        if project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project

        location = getattr(config, 'GOOGLE_CLOUD_LOCATION', None)
        if location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = location

        self.chat_model = None
        self.agent = None
        self.history: List[BaseMessage] = []

        self.memory_manager = None
        if getattr(config, 'ENABLE_PERSISTENT_MEMORY', True):
            try:
                self.memory_manager = ChromaMemoryManager()
            except Exception as e:
                if self.log_queue:
                    self.log_queue.put({"type": "error", "text": f"Failed to initialize ChromaMemoryManager: {e}"})

    def _build_model(self):
        """Factory method to instantiate Google Gemini chat models using langchain-google-genai."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        warnings.filterwarnings("ignore", message=".*is not supported in schema.*")

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature
            )

        key_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and os.path.exists(key_path):
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                credentials=credentials,
                temperature=self.temperature
            )

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature
        )

    def _convert_mcp_tools_to_langchain(self) -> List[StructuredTool]:
        """Converts MCP tool JSON schemas into LangChain StructuredTool objects with execution callbacks."""
        langchain_tools = []
        for t in self.tools:
            name = t["name"]
            description = t.get("description", "")

            # Closure wrapper for asynchronous MCP tool callback execution
            async def _async_tool_func(tool_name=name, **kwargs):
                if self.log_queue:
                    self.log_queue.put({
                        "type": "status",
                        "text": f"🤖 LLM requested tool '{tool_name}' via DeepAgents with args: {kwargs}"
                    })
                if self.execute_tool:
                    try:
                        return await self.execute_tool(tool_name, kwargs)
                    except Exception as e:
                        return f"Error executing tool '{tool_name}': {e}"
                return f"Tool '{tool_name}' execution callback not provided."

            tool_obj = StructuredTool.from_function(
                coroutine=_async_tool_func,
                name=name,
                description=description
            )
            langchain_tools.append(tool_obj)

        return langchain_tools

    def _create_read_file_tool(self) -> StructuredTool:
        """Helper tool allowing DeepAgents SkillsMiddleware to read skill files via read_file."""
        async def read_file(file_path: str, limit: int = 1000) -> str:
            """Reads text content from a file path. Used by SkillsMiddleware to read SKILL.md files."""
            if self.log_queue:
                self.log_queue.put({
                    "type": "status",
                    "text": f"📄 DeepAgents reading skill file: '{file_path}'"
                })
            try:
                p = Path(file_path)
                if not p.is_absolute():
                    p = config.PROJECT_ROOT / file_path
                if not p.exists():
                    return f"Error: File '{file_path}' does not exist."
                
                content = p.read_text(encoding="utf-8")
                lines = content.splitlines()
                if len(lines) > limit:
                    return "\n".join(lines[:limit]) + f"\n... (truncated at {limit} lines)"
                return content
            except Exception as e:
                return f"Error reading file '{file_path}': {e}"

        return StructuredTool.from_function(
            coroutine=read_file,
            name="read_file",
            description="Reads content from a file path. Required by DeepAgents SkillsMiddleware for reading SKILL.md files."
        )

    async def init_chat(self) -> None:
        """Initializes the chat model, builds the DeepAgent graph, and resets conversation state."""
        self.chat_model = self._build_model()
        
        langchain_tools = self._convert_mcp_tools_to_langchain()
        if self.memory_manager:
            memory_tools = get_memory_tools(self.memory_manager, log_queue=self.log_queue)
            langchain_tools.extend(memory_tools)

        # Ensure read_file tool exists for SkillsMiddleware progressive disclosure
        tool_names = [t.name for t in langchain_tools]
        if "read_file" not in tool_names:
            langchain_tools.append(self._create_read_file_tool())

        backend = FilesystemBackend(root_dir=str(config.PROJECT_ROOT))

        # Create the deep agent harness graph
        self.agent = create_deep_agent(
            model=self.chat_model,
            system_prompt=self.system_prompt,
            tools=langchain_tools,
            backend=backend,
            memory=self.memory_files if self.memory_files else None,
            skills=self.skills if self.skills else None
        )

        # Reset history
        self.history = []

        if self.log_queue:
            self.log_queue.put({
                "type": "info",
                "text": f"[bold green]✅ DeepAgents session initialized ({self.provider}:{self.model_name}) with {len(langchain_tools)} tools and skills: {self.skills}[/bold green]"
            })

    async def send_message(self, text: str) -> str:
        """
        Sends a user message to the DeepAgent graph and returns the final response string.
        """
        if not self.agent:
            await self.init_chat()

        self.history.append(HumanMessage(content=text))
        cleaned_history = sanitize_messages(self.history)

        # Invoke DeepAgent graph with current history
        result = await self.agent.ainvoke({"messages": cleaned_history})

        # Update persistent history from the graph output state
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            self.history = sanitize_messages(list(result["messages"]))
            last_msg = self.history[-1]
            
            content = last_msg.content
            if isinstance(content, list):
                text_blocks = []
                for block in content:
                    if isinstance(block, str):
                        text_blocks.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        text_blocks.append(block["text"])
                return "".join(text_blocks)
                
            return str(content or "")

        return ""
