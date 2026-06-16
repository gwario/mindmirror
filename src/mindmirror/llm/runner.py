import re
import time
import asyncio
from dotenv import load_dotenv
from google.api_core import exceptions

from mindmirror import config
from mindmirror.llm.gemini.mcp_client import MCPClientManager

def strip_markdown(text):
    """Remove markdown formatting but preserve all text content including code"""
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def parse_llm_response(response_text):
    """Parses LLM output to separate style tags from spoken text."""
    pattern = r'\[(NEUTRAL|EXCITED|SERIOUS|LAZY)\]\s*((?:(?!\[(?:NEUTRAL|EXCITED|SERIOUS|LAZY)\]).)+)'
    matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
    if matches:
        return [(style.lower(), text.strip()) for style, text in matches]
    return [("neutral", response_text.strip())]

def run_ttt_loop(ttt_class, ttt_kwargs, system_prompt, log_queue, text_queue, response_queue):
    """Process text from STT and send to TTT/LLM engine."""
    try:
        load_dotenv()
        asyncio.run(async_run_ttt_loop(ttt_class, ttt_kwargs, system_prompt, log_queue, text_queue, response_queue))
    except KeyboardInterrupt:
        print("TTT (AI) shutting down...")
    except Exception as e:
        log_queue.put({'type': 'status', 'text': f"❌ Critical error in TTT task: {e}"})

async def async_run_ttt_loop(ttt_class, ttt_kwargs, system_prompt, log_queue, text_queue, response_queue):
    """Async text-to-text loop task using MCP clients and custom TTT client."""
    mcp_servers_config = getattr(config, 'MCP_SERVERS', [])
    mcp_manager = MCPClientManager(mcp_servers_config, log_queue)
    await mcp_manager.start()

    try:
        # Discover tools from MCP servers
        tools = await mcp_manager.get_all_tools()

        # Decoupled callback for tool execution
        async def execute_tool_callback(name, args):
            return await mcp_manager.call_tool(name, args)

        # Merge composition configurations
        kwargs = dict(ttt_kwargs)
        kwargs.update({
            "system_prompt": system_prompt,
            "tools": tools,
            "execute_tool_callback": execute_tool_callback,
            "log_queue": log_queue
        })

        # Instantiate concrete TTT client inside child process boundary
        llm_client = ttt_class(**kwargs)
        await llm_client.init_chat()

        last_request_time = 0
        min_interval = 5.0  # 5 seconds between requests

        while True:
            text = await asyncio.to_thread(text_queue.get)

            if not text.strip():
                continue

            # Rate limiting
            elapsed = time.time() - last_request_time
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                log_queue.put({'type': 'status', 'text': f"⏱️  Rate limiting: waiting {wait_time:.1f}s..."})
                await asyncio.sleep(wait_time)

            # Retry logic
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    response_text = await llm_client.send_message(text)
                    log_queue.put({'type': 'ai', 'text': response_text})

                    # Parse response into segments based on style tags
                    segments = parse_llm_response(response_text)

                    # Send to response queue for TTS synthesis
                    for style, segment_text in segments:
                        clean_text = strip_markdown(segment_text)
                        response_queue.put((style, clean_text))

                    last_request_time = time.time()
                    break

                except exceptions.ResourceExhausted as e:
                    retry_count += 1
                    wait_time = 60 * retry_count  # Exponential backoff
                    log_queue.put({'type': 'status', 'text': f"❌ Rate limit hit! Retry {retry_count}/{max_retries}"})
                    log_queue.put({'type': 'status', 'text': f"⏱️  Waiting {wait_time}s before retrying..."})
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    log_queue.put({'type': 'status', 'text': f"❌ Error with TTT: {e}"})
                    log_queue.put({'type': 'status', 'text': "Skipping this message..."})
                    break

            if retry_count >= max_retries:
                log_queue.put({'type': 'status', 'text': f"❌ Failed after {max_retries} retries, skipping message"})
                
    finally:
        await mcp_manager.close()
