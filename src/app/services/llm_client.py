import os
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Optional
import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- LLM Interface Definition (Dependency Inversion) ---

@runtime_checkable
class LLMInterface(Protocol):
    """Abstract interface for Language Model Client interaction."""
    
    @property
    @abstractmethod
    def api_key(self) -> Optional[str]:
        """Returns the API key used by the client, if configured."""
        ...

    @abstractmethod
    async def get_completion(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> str:
        """Generates a text completion using the language model.

        Args:
            prompt: The prompt to send to the model.
            model: The specific model to use (defaulting to a common choice).
            **kwargs: Additional arguments for the OpenAI client.

        Returns:
            The generated text completion.
            
        Raises:
             ValueError: If the API key is missing.
             Exception: Propagates exceptions from the underlying API call.
        """
        ...

# --- Concrete Implementation ---

class LLMClient(LLMInterface): # Implement the interface
    """Client for interacting with the OpenAI API, implementing LLMInterface."""

    def __init__(self):
        """Initializes the asynchronous OpenAI client."""
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            # Use logger.warning instead of print
            logger.warning("OPENAI_API_KEY environment variable not set.") 
            self.client = None # Explicitly set client to None
        else:
            self.client = AsyncOpenAI(api_key=self._api_key)

    @property
    def api_key(self) -> Optional[str]:
        """Returns the configured OpenAI API key."""
        return self._api_key

    async def get_completion(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> str:
        """Generates a text completion using the configured OpenAI model."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. API key may be missing.")

        try:
            response = await self.client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                 **kwargs # Pass through any extra arguments
            )
            # Basic error handling for response structure
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                return content.strip() if content else "" 
            else:
                 # Log unexpected response structure
                 print(f"Warning: Unexpected OpenAI response structure: {response}")
                 return "" # Return empty string or raise error?
        except Exception as e:
            # Log or handle specific OpenAI errors if needed
            print(f"Error during OpenAI API call: {e}")
            raise # Re-raise the exception to be handled by the caller

    # Deprecated generate method - replace usages with get_completion
    async def generate(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Deprecated: Use get_completion instead."""
        print("Warning: LLMClient.generate is deprecated. Use get_completion.")
        return await self.get_completion(prompt, model=model)

# Example usage (optional, for testing purposes)
# async def main():
#     client = LLMClient()
#     if client.api_key: # Only run if key is present
#         try:
#             response = await client.get_completion("Explain the concept of reflection in learning in one sentence.")
#             print("LLM Response:", response)
#         except Exception as e:
#             print(f"Error getting completion: {e}")
#     else:
#         print("Skipping example usage as API key is not configured.")

# if __name__ == "__main__":
#     import asyncio
#     # To run this: python -m app.services.llm_client
#     # Ensure OPENAI_API_KEY is set in your environment or a .env file
#     # asyncio.run(main()) # Commented out to avoid running by default
#     pass 