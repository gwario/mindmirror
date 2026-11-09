from typing import List, Optional, Any
from langchain_core.tools import StructuredTool

from mindmirror.memory.chroma_store import ChromaMemoryManager

def get_memory_tools(memory_manager: ChromaMemoryManager, log_queue: Optional[Any] = None) -> List[StructuredTool]:
    """
    Constructs LangChain StructuredTools bound to the given ChromaMemoryManager instance.
    """
    
    async def remember_fact(fact: str, category: str = "general", speaker_id: Optional[str] = None) -> str:
        """Stores a fact, user preference, or key piece of information into persistent memory.
        Use this tool whenever the user asks you to remember something, or when they share important personal details.

        Args:
            fact: The factual statement or preference to store (e.g., 'User loves coffee', 'User lives in London').
            category: Optional category classification (e.g. 'preference', 'personal', 'work', 'general').
            speaker_id: Optional speaker ID (if known). If omitted, memory is saved as unresolved and tagged retroactively.
        """
        print(f"\n💾 [MEMORY STORE] Storing fact: '{fact}' | Category: '{category}' | Speaker: '{speaker_id or 'unresolved'}'")
        mem_id = memory_manager.add_memory(fact=fact, category=category, speaker_id=speaker_id)
        total_docs = memory_manager.collection.count()
        print(f"✅ [MEMORY STORED] ID: '{mem_id}' | Total Vector Docs in DB: {total_docs}")

        if log_queue:
            log_queue.put({
                "type": "info",
                "text": f"💾 [cyan][MEMORY STORED][/cyan] Saved memory [bold]'{fact}'[/bold] (ID: {mem_id}, Speaker: {speaker_id or 'unresolved'})"
            })
            
        return f"Successfully saved memory (ID: {mem_id}): '{fact}'"

    async def recall_memories(query: str, limit: int = 5, speaker_id: Optional[str] = None) -> str:
        """Searches persistent memory for stored facts, preferences, or details matching the query.
        Use this tool when answering questions about past context, user preferences, or saved facts.

        Args:
            query: The search context or query to look up in persistent memory.
            limit: Maximum number of memories to return (default: 5).
            speaker_id: Optional speaker ID to filter memories by.
        """
        print(f"\n🔍 [MEMORY RECALL] Querying vector store for: '{query}' (limit={limit}, speaker={speaker_id})")
        results = memory_manager.recall_memories(query=query, limit=limit, speaker_id=speaker_id)

        if not results:
            print("⚠️ [MEMORY RECALL] No matching memories found in vector store.")
            if log_queue:
                log_queue.put({
                    "type": "status",
                    "text": f"🔍 [MEMORY RECALL] Query '{query}' ➡️ 0 matching memories in ChromaDB"
                })
            return "No relevant memories found in persistent memory."

        print(f"✅ [MEMORY RECALLED] Found {len(results)} matching memory item(s):")
        formatted = []
        for i, item in enumerate(results, 1):
            print(f"   {i}. [{item['category']}] '{item['fact']}' (Distance: {item.get('distance', 0.0):.4f}, Saved: {item['timestamp']})")
            formatted.append(f"{i}. [{item['category']}] {item['fact']} (Saved: {item['timestamp']})")

        if log_queue:
            log_queue.put({
                "type": "info",
                "text": f"🔍 [cyan][MEMORY RECALLED][/cyan] Query '{query}' ➡️ Found {len(results)} item(s): {', '.join(f'[{x['fact']}]' for x in results[:3])}"
            })

        return "\n".join(formatted)

    tool_remember = StructuredTool.from_function(
        coroutine=remember_fact,
        name="remember_fact",
        description="Stores a fact, user preference, or key info into persistent vector memory (ChromaDB). MANDATORY: Call this tool whenever the user says 'remember this', 'keep in mind...', or shares personal preferences/facts."
    )

    tool_recall = StructuredTool.from_function(
        coroutine=recall_memories,
        name="recall_memories",
        description="Searches persistent vector memory (ChromaDB / memory store) for past facts, memories, ChromaDB entries, and saved details. MANDATORY: Call this tool whenever the user asks about stored memories, ChromaDB, past context, or saved facts."
    )

    return [tool_remember, tool_recall]
