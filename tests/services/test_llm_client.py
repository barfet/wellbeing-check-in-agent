import pytest
import os
from unittest.mock import patch, AsyncMock

# Ensure the src directory is in the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.services.llm_client import LLMClient
import openai

@pytest.fixture
def mock_openai_client():
    """Fixture to mock the openai.AsyncOpenAI client and its methods."""
    # Mock the response structure expected from chat.completions.create
    mock_completion = AsyncMock()
    mock_completion.choices = [AsyncMock()]
    mock_completion.choices[0].message = AsyncMock()
    mock_completion.choices[0].message.content = "Mocked LLM Response"

    mock_async_client = AsyncMock()
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_async_client

# Test successful initialization and completion
@pytest.mark.asyncio
@patch('openai.AsyncOpenAI') # Patch the class where it's looked up
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
async def test_llm_client_get_completion_success(mock_async_openai_class, mock_openai_client):
    """Test successful completion retrieval when API key is set."""
    mock_async_openai_class.return_value = mock_openai_client # Configure the class mock

    client = LLMClient()
    assert client.api_key == "test_key"

    prompt = "Test prompt"
    response = await client.get_completion(prompt)

    assert response == "Mocked LLM Response"
    mock_async_openai_class.assert_called_once()
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to facilitate reflective learning conversations."}, # Added more specific system prompt
            {"role": "user", "content": prompt}
        ]
    )

# Test initialization warning when API key is missing
# Patch os.getenv specifically where it's used in the llm_client module
@patch('app.services.llm_client.os.getenv', return_value=None)
@patch('app.services.llm_client.logger.warning') # Mock logger
# @patch('app.services.llm_client.load_dotenv') # No longer mocking load_dotenv here
def test_llm_client_init_no_api_key(mock_logger_warning, mock_getenv):
    """Test that a warning is logged if OPENAI_API_KEY is missing."""
    # Ensure getenv returns None for the specific key check
    def getenv_side_effect(key, default=None):
        if key == "OPENAI_API_KEY":
            return None
        return os.environ.get(key, default) # Allow other env vars potentially
    mock_getenv.side_effect = getenv_side_effect
    
    LLMClient()
    # Assert warning was called
    mock_logger_warning.assert_called_once_with(
        "OPENAI_API_KEY not found in environment variables."
    )
    # Verify os.getenv was called for the API key
    mock_getenv.assert_any_call("OPENAI_API_KEY")

# Test get_completion raises ValueError when API key is missing
@pytest.mark.asyncio
@patch.dict(os.environ, {}, clear=True)
async def test_llm_client_get_completion_no_api_key():
    """Test get_completion raises ValueError if API key was not set during init."""
    client = LLMClient()
    with pytest.raises(ValueError, match="OpenAI API key is missing"): # Verify exception is raised
        await client.get_completion("Test prompt")

# Test handling of OpenAI API errors
@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
async def test_llm_client_get_completion_api_error(mock_async_openai_class, mock_openai_client):
    """Test that OpenAI API errors are caught and re-raised."""
    # Configure mock to raise an API error
    mock_openai_client.chat.completions.create.side_effect = openai.APIError(
        "Test API Error", request=None, body=None
    )
    mock_async_openai_class.return_value = mock_openai_client

    client = LLMClient()
    with pytest.raises(openai.APIError, match="Test API Error"): # Verify the specific error is raised
        await client.get_completion("Test prompt")

# Test handling of Authentication Errors
@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
async def test_llm_client_get_completion_auth_error(mock_async_openai_class, mock_openai_client):
    """Test that OpenAI Authentication errors are caught and re-raised."""
    mock_openai_client.chat.completions.create.side_effect = openai.AuthenticationError(
        "Invalid API Key", request=None, body=None
    )
    mock_async_openai_class.return_value = mock_openai_client

    client = LLMClient()
    with pytest.raises(openai.AuthenticationError, match="Invalid API Key"):
        await client.get_completion("Test prompt")

# Test handling when LLM returns None content
@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
async def test_llm_client_get_completion_none_content(mock_async_openai_class, mock_openai_client):
    """Test handling when the LLM response content is None."""
    # Mock the response structure with None content
    mock_completion = AsyncMock()
    mock_completion.choices = [AsyncMock()]
    mock_completion.choices[0].message = AsyncMock()
    mock_completion.choices[0].message.content = None # Simulate None content
    mock_openai_client.chat.completions.create.return_value = mock_completion
    mock_async_openai_class.return_value = mock_openai_client

    client = LLMClient()
    response = await client.get_completion("Test prompt")

    assert response == "" # Expect empty string as per implementation
