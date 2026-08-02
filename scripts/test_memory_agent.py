import sys
import asyncio
from pathlib import Path

# Add src to sys.path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mindmirror.llm.langchain.client import LangChainLLMClient
from mindmirror.llm.runner import parse_llm_response, strip_markdown
from mindmirror import config

SYSTEM_PROMPT = """
You are a smart, helpful assistant with a chill and upbeat vibe. Keep
responses short and snappy — no waffle, no fluff. Match the user's energy.

When you reply, strictly prepend a style tag to your message: [NEUTRAL], [EXCITED], [SERIOUS], [LAZY].
""".strip()

async def test_memory_agent():
    print("=" * 60)
    print("🧪 Testing ChromaDB Persistent Memory Agent Integration")
    print("=" * 60)

    client = LangChainLLMClient(
        system_prompt=SYSTEM_PROMPT
    )

    print("⚡ Initializing chat session & persistent ChromaDB memory...")
    await client.init_chat()
    print("✅ Chat session & memory initialized successfully!")
    print("-" * 60)

    prompts = [
        "Please remember that my favorite programming language is Python and I live in Munich, Germany.",
        "What is my favorite programming language and where do I live?",
        "Can you look up the memories stored in your Chroma DB?"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n💬 Turn {i}: '{prompt}'")
        try:
            response_text = await client.send_message(prompt)
            print("🤖 Raw Model Response:")
            print(response_text)
            parsed = parse_llm_response(response_text)
            print("🎨 Parsed Style Segments:")
            for style, content in parsed:
                clean = strip_markdown(content)
                print(f"   - [{style.upper()}]: {clean}")
            if client.memory_manager:
                print(f"📊 [VECTOR DB COUNT] Current Total Stored Memories: {client.memory_manager.collection.count()}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Error during turn {i}: {e}")
            import traceback
            traceback.print_exc()
            break

    print("✅ Persistent Memory Agent test complete!")

if __name__ == "__main__":
    asyncio.run(test_memory_agent())
