import os
import sys
import asyncio
from pathlib import Path

# Add src folder to sys.path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from mindmirror import config
from mindmirror.main import SYSTEM_PROMPT
from mindmirror.llm.langchain import LangChainLLMClient
from mindmirror.llm.runner import parse_llm_response, strip_markdown

async def test_langchain_client():
    print("=" * 60)
    print("🧪 Testing LangChain TTT Client Integration")
    print("=" * 60)
    
    provider = getattr(config, 'LANGCHAIN_TTT_PROVIDER', 'vertexai')
    model_name = getattr(config, 'LANGCHAIN_TTT_MODEL', 'gemini-2.0-flash-lite-preview-02-05')
    
    print(f"Provider   : {provider}")
    print(f"Model Name : {model_name}")
    print("-" * 60)
    
    # Mock tool callback
    async def mock_execute_tool(name, args):
        print(f"🔧 [Mock Tool Call] Executing '{name}' with args: {args}")
        return f"Mock result for tool {name}"

    mock_tools = [
        {
            "name": "check_fraud_score",
            "description": "Checks the fraud risk score for a user ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user ID to check"}
                },
                "required": ["user_id"]
            }
        }
    ]

    client = LangChainLLMClient(
        model_name=model_name,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        tools=mock_tools,
        execute_tool_callback=mock_execute_tool
    )

    print("⚡ Initializing chat session...")
    await client.init_chat()
    print("✅ Chat session initialized successfully!")
    print("-" * 60)

    prompts = [
        "Give a quick hello and confirm you are ready!",
        "Check fraud score for user_id 12345.",
        "Awesome, thanks!"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n💬 Prompt {i}: '{prompt}'")
        try:
            response_text = await client.send_message(prompt)
            print("🤖 Raw Model Response:")
            print(response_text)
            parsed = parse_llm_response(response_text)
            print("🎨 Parsed Style Segments:")
            for style, content in parsed:
                clean = strip_markdown(content)
                print(f"   - [{style.upper()}]: {clean}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Error during message processing: {e}")
            import traceback
            traceback.print_exc()
            break

    print("✅ LangChain TTT multi-turn test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_langchain_client())
