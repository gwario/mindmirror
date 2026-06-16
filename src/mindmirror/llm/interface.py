from abc import ABC, abstractmethod

class TTTInterface(ABC):
    """
    Abstract Base Class defining the contract for all Text-to-Thought (TTT) / Large Language Model (LLM) engines.
    """

    @abstractmethod
    async def init_chat(self) -> None:
        """
        Initializes the conversational session, setting up system prompts, tool schemas, and chat history.
        
        Raises:
            Exception: If initialization fails.
        """
        pass

    @abstractmethod
    async def send_message(self, text: str) -> str:
        """
        Sends a text input to the conversational engine and returns the text response.
        Should handle multi-turn function/tool call execution loops automatically if requested by the engine.
        
        Args:
            text (str): The input user text transcript.
            
        Returns:
            str: The raw model response text (including style tags or formatting).
            
        Raises:
            Exception: If inference fails or rate limits are hit.
        """
        pass
