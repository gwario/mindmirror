from typing import Any
from google import genai
from google.genai import types
from mindmirror.llm.interface import TTTInterface

def clean_schema(schema: Any) -> Any:
    """Recursively remove additionalProperties/additional_properties from schema dictionary."""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in ("additionalProperties", "additional_properties"):
                continue
            cleaned[k] = clean_schema(v)
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    return schema

class GeminiLLMClient(TTTInterface):
    """
    Decoupled Gemini LLM client.
    Converts generic dictionary tools into Google GenAI types
    and manages manual tool call execution loops.
    """
    def __init__(self, model_name: str, system_prompt: str, tools: list[dict], execute_tool_callback, log_queue=None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.execute_tool = execute_tool_callback
        self.log_queue = log_queue
        
        import os
        import json
        from mindmirror import config

        key_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        project_id = None
        if key_path and os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    key_data = json.load(f)
                    project_id = key_data.get("project_id")
            except Exception:
                pass

        project = project_id or getattr(config, 'GOOGLE_CLOUD_PROJECT', None)
        location = getattr(config, 'GOOGLE_CLOUD_LOCATION', None)

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location
        )
        self.chat = None

    async def init_chat(self):
        """Initializes the async chat session with system instruction and tool definitions."""
        function_declarations = []
        for t in self.tools:
            # Map and clean parameters schema
            params = clean_schema(t.get("inputSchema", {}))
            
            decl = types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=params
            )
            function_declarations.append(decl)
            
        tool_config = None
        # gemini-2.0-flash-lite-preview-02-05 / gemini-2.0-flash-lite / flash-lite models do not support function calling
        is_lite_model = "lite" in self.model_name.lower()
        if function_declarations and not is_lite_model:
            tool_config = [types.Tool(function_declarations=function_declarations)]
            
        self.chat = self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                tools=tool_config,
                system_instruction=self.system_prompt,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
            history=[]
        )
        
        if self.log_queue:
            self.log_queue.put({
                "type": "info", 
                "text": f"Gemini chat session initialized with model '{self.model_name}' and {len(function_declarations)} tools."
            })

    async def send_message(self, text: str) -> str:
        """
        Sends a message to the chat session. Handles the execution loop
        if the model requests any function calls.
        """
        if not self.chat:
            await self.init_chat()
            
        response = await self.chat.send_message(text)
        
        # Keep resolving function calls until the model returns a final text response
        while response.function_calls:
            parts = []
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = function_call.args
                
                if self.log_queue:
                    self.log_queue.put({
                        "type": "status", 
                        "text": f"🤖 LLM requested tool '{tool_name}' with arguments: {tool_args}"
                    })
                    
                # Call the decoupled execution callback
                try:
                    tool_result = await self.execute_tool(tool_name, tool_args)
                except Exception as e:
                    tool_result = f"Error: Tool execution failed: {e}"
                    
                parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result}
                    )
                )
                
            # Send all function responses together back to the model
            response = await self.chat.send_message(parts)
                
        return response.text or ""
