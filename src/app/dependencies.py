# src/app/dependencies.py

import logging
from functools import lru_cache
from .services.llm_client import LLMClient

logger = logging.getLogger(__name__)

@lru_cache()
def get_llm_client() -> LLMClient:
    """Dependency function to get a cached instance of LLMClient."""
    logger.info("Initializing LLMClient instance.")
    # LLMClient internally handles dotenv loading for the API key
    client = LLMClient()
    if not client.api_key:
        # Log a warning if the key isn't found, but allow initialization
        # The graph nodes will handle the error if the key is missing during calls
        logger.warning("LLMClient initialized, but OpenAI API key was not found.")
    return client 