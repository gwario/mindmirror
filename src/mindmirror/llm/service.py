"""MindMirror Autonomous Reasoning Microservice Runner."""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

from mindmirror.llm.component import ReasoningComponent
from mindmirror.llm.context.manager import ContextManager
from mindmirror.llm.reasoning.custom.tree_of_thoughts import TreeOfThoughtsStrategy
from mindmirror.memory.chroma_store import ChromaMemoryManager
from mindmirror.pipeline.supervisor import ComponentSupervisor
from mindmirror.pipeline.transport import create_message_bus
from mindmirror.speaker.context import SpeakerContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mindmirror.llm.service")


def get_reasoning_strategy():
    """Builds configured reasoning strategy from environment using the strategy registry."""
    strategy_name = os.getenv("REASONING_STRATEGY", "tree_of_thoughts")
    from mindmirror.llm.reasoning.registry import ReasoningStrategyRegistry
    import mindmirror.llm.reasoning.custom.poc_symbolic_reasoner  # noqa: F401
    try:
        return ReasoningStrategyRegistry.get(strategy_name)
    except Exception as e:
        logger.warning("Could not load strategy '%s' (%s); falling back to TreeOfThoughtsStrategy.", strategy_name, e)
        from mindmirror.llm.reasoning.custom.tree_of_thoughts import TreeOfThoughtsStrategy
        return TreeOfThoughtsStrategy()


async def main():
    logger.info("Initializing MindMirror Autonomous Reasoning Service...")
    bus = create_message_bus()
    supervisor = ComponentSupervisor(check_interval=0.5)

    # Initialize ChromaDB Memory Manager
    chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")) if not os.getenv("CHROMA_HOST") else None
    memory_manager = ChromaMemoryManager(db_dir=chroma_dir)
    await memory_manager.attach_bus(bus)

    # Initialize Speaker Context
    speaker_context = SpeakerContext(bus=bus)

    # Initialize Context Manager
    context_manager = ContextManager()

    # Register Reasoning Component
    strategy = get_reasoning_strategy()
    supervisor.register(
        "reasoning",
        lambda: ReasoningComponent(
            bus=bus,
            strategy=strategy,
            context_manager=context_manager,
            speaker_context=speaker_context,
            memory_manager=memory_manager,
        ),
    )

    await supervisor.start()
    logger.info("Reasoning Service running and supervised under ComponentSupervisor.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down Reasoning Service...")
        await supervisor.stop()
        await speaker_context.close()
        await memory_manager.close()
        await bus.close()
        logger.info("Reasoning Service stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
