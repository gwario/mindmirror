import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

class MCPClientManager:
    """
    Manages connections to one or more MCP servers.
    Hides all MCP-specific connection details from the LLM client.
    """
    def __init__(self, servers_config, log_queue):
        self.servers_config = servers_config
        self.log_queue = log_queue
        self.exit_stack = AsyncExitStack()
        self.sessions = {}  # name -> ClientSession
        self.tool_to_server = {}  # tool_name -> server_name

    async def start(self):
        """Connects to all configured MCP servers."""
        self.log_queue.put({"type": "info", "text": "[blue]🔌 Initializing MCP client connections...[/blue]"})
        for cfg in self.servers_config:
            name = cfg["name"]
            transport_type = cfg["type"]
            self.log_queue.put({"type": "info", "text": f"Connecting to MCP server '{name}' via {transport_type}..."})
            try:
                if transport_type == "stdio":
                    server_params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env", None)
                    )
                    # Establish stdio connection
                    read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    self.sessions[name] = session
                    self.log_queue.put({"type": "info", "text": f"[green]✅ Connected to stdio MCP server '{name}'[/green]"})
                    
                elif transport_type == "sse":
                    url = cfg["url"]
                    # Establish SSE connection
                    read, write = await self.exit_stack.enter_async_context(sse_client(url))
                    session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    self.sessions[name] = session
                    self.log_queue.put({"type": "info", "text": f"[green]✅ Connected to SSE MCP server '{name}'[/green]"})
                    
            except Exception as e:
                self.log_queue.put({
                    "type": "status", 
                    "text": f"[red]❌ Failed to connect to MCP server '{name}': {e}[/red]"
                })

    async def get_all_tools(self) -> list[dict]:
        """
        Queries all active MCP servers for their tools and returns them
        as standard Python dictionaries representing JSON schemas.
        """
        all_tools = []
        self.tool_to_server.clear()
        
        for server_name, session in self.sessions.items():
            try:
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    # Map the tool name to its providing server
                    self.tool_to_server[tool.name] = server_name
                    
                    # Convert inputSchema to a plain dictionary if it is a Pydantic object
                    schema = tool.inputSchema
                    if not isinstance(schema, dict):
                        if hasattr(schema, "model_dump"):
                            schema = schema.model_dump()
                        elif hasattr(schema, "dict"):
                            schema = schema.dict()
                            
                    all_tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": schema
                    })
            except Exception as e:
                self.log_queue.put({
                    "type": "status", 
                    "text": f"[yellow]⚠️ Failed to list tools for '{server_name}': {e}[/yellow]"
                })
                
        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Executes a tool call on the appropriate MCP server and returns
        the result as a plain text string.
        """
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool '{tool_name}' is not registered with any active MCP server.")
            
        session = self.sessions[server_name]
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            
            # Aggregate text content blocks from tool response
            text_blocks = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_blocks.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    text_blocks.append(block["text"])
                else:
                    text_blocks.append(str(block))
                    
            return "\n".join(text_blocks)
        except Exception as e:
            self.log_queue.put({
                "type": "status",
                "text": f"[red]❌ Error calling tool '{tool_name}' on '{server_name}': {e}[/red]"
            })
            raise e

    async def close(self):
        """Closes all active MCP client sessions and connections."""
        await self.exit_stack.aclose()
        self.log_queue.put({"type": "info", "text": "All MCP client connections shut down cleanly."})
