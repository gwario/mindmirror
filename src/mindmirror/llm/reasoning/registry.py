"""Dynamic Registry and Pluggable Factory for Reasoning Strategies."""

import importlib
import logging
from typing import Any, Callable, Type
from mindmirror.llm.reasoning.interface import ReasoningStrategy

logger = logging.getLogger(__name__)


class ReasoningStrategyRegistry:
    """Registry facilitating plug-and-play attachment of custom reasoning algorithms.

    Allows developers to attach arbitrary reasoning logic (MCTS, Tree of Thoughts,
    Symbolic Solvers, Reflexion, or Local LLMs) without modifying core pipeline code.
    """

    _registry: dict[str, Type[ReasoningStrategy] | Callable[..., ReasoningStrategy]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_cls_or_factory: Type[ReasoningStrategy] | Callable[..., ReasoningStrategy],
    ) -> None:
        """Registers a custom reasoning strategy under a unique identifier."""
        cls._registry[name.lower()] = strategy_cls_or_factory
        logger.info("Registered custom reasoning strategy '%s'.", name)

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> ReasoningStrategy:
        """Instantiates a registered strategy by name, or dynamically loads from module path.

        Examples:
            strategy = ReasoningStrategyRegistry.get("tree_of_thoughts")
            strategy = ReasoningStrategyRegistry.get("local_llm", model="qwen2.5:7b")
            strategy = ReasoningStrategyRegistry.get("my_package.custom_solver.CustomStrategy")
        """
        key = name.lower()
        if key in cls._registry:
            factory = cls._registry[key]
            return factory(**kwargs)

        # Attempt dynamic module import if name contains a dot (e.g. 'my_pkg.module.MyClass')
        if "." in name:
            module_name, class_name = name.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_name)
                cls_obj = getattr(mod, class_name)
                if not issubclass(cls_obj, ReasoningStrategy):
                    raise TypeError(f"Class '{class_name}' must inherit from ReasoningStrategy.")
                logger.info("Dynamically loaded custom reasoning strategy '%s'.", name)
                return cls_obj(**kwargs)
            except Exception as e:
                raise ImportError(f"Failed to dynamically import reasoning strategy '{name}': {e}") from e

        available = list(cls._registry.keys())
        raise KeyError(f"Reasoning strategy '{name}' not found. Available strategies: {available}")

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Returns the list of registered strategy identifiers."""
        return list(cls._registry.keys())


# Pre-register standard built-in strategies
def _register_builtins():
    try:
        from mindmirror.llm.reasoning.custom.tree_of_thoughts import TreeOfThoughtsStrategy
        ReasoningStrategyRegistry.register("tree_of_thoughts", TreeOfThoughtsStrategy)
    except ImportError:
        pass

    try:
        from mindmirror.llm.reasoning.native.provider_native import ProviderNativeReasoningStrategy
        ReasoningStrategyRegistry.register("native_thinking", ProviderNativeReasoningStrategy)
    except ImportError:
        pass

    try:
        from mindmirror.llm.reasoning.local_llm import LocalLLMReasoningStrategy
        ReasoningStrategyRegistry.register("local_llm", LocalLLMReasoningStrategy)
    except ImportError:
        pass


_register_builtins()
