import os
import openai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    A wrapper class for interacting with the OpenAI API.
    Handles API key loading, prompt execution, and basic error handling.
    """
    def __init__(self):
        """
        Initializes the LLMClient, loads the API key from environment variables.
        """
        load_dotenv()  # Load environment variables from .env file if present
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
            # Consider raising an error or handling this case more robustly
            # For now, we allow initialization but API calls will fail.
        else:
            # Setting the API key globally for the openai library instance
            # Note: If managing multiple keys or instances, a different approach might be needed.
            openai.api_key = self.api_key
            logger.info("LLMClient initialized with API key.")

    async def get_completion(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Gets a completion from the OpenAI API based on the provided prompt.

        Args:
            prompt: The input prompt for the LLM.
            model: The model to use for completion (defaults to gpt-4o-mini).

        Returns:
            The text response from the LLM.

        Raises:
            openai.error.OpenAIError: If there's an issue with the API call.
            ValueError: If the API key is missing.
        """
        if not self.api_key:
             logger.error("OpenAI API key is missing. Cannot make API calls.")
             raise ValueError("OpenAI API key is missing. Cannot make API calls.")

        try:
            logger.info(f"Requesting completion from model '{model}'...")
            # Note: Using ChatCompletion for newer models like gpt-4o-mini/gpt-4
            # Using the async client is recommended for FastAPI/async applications
            client = openai.AsyncOpenAI() # Initialize async client here
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to facilitate reflective learning conversations."}, # Added more specific system prompt
                    {"role": "user", "content": prompt}
                ]
                # Add other parameters like temperature, max_tokens etc. if needed
            )
            completion = response.choices[0].message.content
            if completion is None:
                 logger.warning("Received None as completion content from OpenAI.")
                 completion = "" # Return empty string if content is None
            else:
                 completion = completion.strip()

            logger.info("Completion received successfully.")
            return completion
        except openai.AuthenticationError as e:
             logger.error(f"OpenAI Authentication Error: Check your API key. Details: {e}")
             raise
        except openai.APIError as e: # Catching more specific API errors
            logger.error(f"OpenAI API Error: {e}")
            # Implement basic retry logic here if desired (e.g., for 5xx errors)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM interaction: {e}")
            raise

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