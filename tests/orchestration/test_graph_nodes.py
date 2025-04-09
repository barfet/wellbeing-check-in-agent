import pytest
import os
from unittest.mock import patch, AsyncMock

# Ensure the src directory is in the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.orchestration.state import AgentState
# Import the node functions directly for isolated testing
from app.orchestration.graph import initiate, probe, summarize, check_summary, route_after_summary_check
from langgraph.graph import END # Import END for routing check

# --- Tests for initiate node --- 

@pytest.mark.asyncio
async def test_initiate_node_with_topic():
    """Test the initiate node when a topic is provided."""
    initial_state = AgentState(topic="My Test Topic")
    result_state = await initiate(initial_state)

    expected_question = "Okay, let's reflect on 'My Test Topic'. To start, could you tell me briefly what happened regarding this?"
    assert result_state.current_question == expected_question
    assert result_state.history == [("agent", expected_question)]
    assert result_state.topic == "My Test Topic"
    assert result_state.error_message is None

@pytest.mark.asyncio
async def test_initiate_node_without_topic():
    """Test the initiate node when no topic is provided."""
    initial_state = AgentState() # No topic
    result_state = await initiate(initial_state)

    expected_question = "Hello! What topic or experience would you like to reflect on today?"
    assert result_state.current_question == expected_question
    assert result_state.history == [("agent", expected_question)]
    assert result_state.topic is None
    assert result_state.error_message is None

# --- Tests for probe node --- 

@pytest.fixture
def mock_llm_client():
    """Fixture to create a mock LLMClient instance."""
    mock_client = AsyncMock()
    mock_client.get_completion = AsyncMock(return_value="Mocked follow-up question?")
    # Set api_key attribute to simulate configured client
    mock_client.api_key = "fake_key"
    return mock_client

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client') # Patch the instance used in graph.py
async def test_probe_node_success(mock_llm_client_instance): # Use the mock instance passed by patch
    """Test the probe node successful execution with user input."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="Mocked follow-up question?")
    mock_llm_client_instance.api_key = "fake_key" # Ensure mock instance has api_key

    initial_state = AgentState(
        history=[
            ("agent", "Initial question?"),
            ("user", "This is my user response.")
        ]
    )
    result_state = await probe(initial_state)

    expected_question = "Mocked follow-up question?"
    assert result_state.current_question == expected_question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", expected_question)
    assert result_state.error_message is None

    # Verify LLMClient was called with a prompt containing history
    mock_llm_client_instance.get_completion.assert_called_once()
    call_args, _ = mock_llm_client_instance.get_completion.call_args
    prompt_arg = call_args[0]
    assert "This is my user response." in prompt_arg
    assert "Initial question?" in prompt_arg

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_probe_node_llm_error(mock_llm_client_instance):
    """Test the probe node when the LLM call raises an exception."""
    test_exception = Exception("LLM Unavailable")
    mock_llm_client_instance.get_completion = AsyncMock(side_effect=test_exception)
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    result_state = await probe(initial_state)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_probe_node_llm_empty_response(mock_llm_client_instance):
    """Test the probe node when the LLM returns an empty string."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="") # Simulate empty response
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    result_state = await probe(initial_state)

    assert result_state.current_question == "Could you please elaborate on that?" # Check specific fallback
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "LLM failed to generate a specific question" in result_state.error_message

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_probe_node_no_user_input_last(mock_llm_client_instance):
    """Test probe node behavior when the last turn wasn't the user."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="Generated Question")
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("agent", "Q1"), ("agent", "Q2")]) # No user turn last
    result_state = await probe(initial_state)

    assert result_state.current_question == "Generated Question"
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert result_state.error_message is None # Should still proceed

    # Verify prompt was generic
    mock_llm_client_instance.get_completion.assert_called_once_with(
        "Ask a generic open-ended question to encourage further reflection."
    )

@pytest.mark.asyncio
async def test_probe_node_empty_history():
    """Test probe node behavior with empty history."""
    initial_state = AgentState(history=[])
    result_state = await probe(initial_state)

    assert result_state.current_question is None
    assert result_state.history == [] # History remains empty
    assert result_state.error_message == "Internal Error: Prober requires history."

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_probe_node_llm_api_key_missing(mock_llm_client_instance):
    """Test probe node when LLMClient API key is missing at call time."""
    mock_llm_client_instance.api_key = None # Simulate missing API key
    # Mock get_completion shouldn't be called, but mock it just in case
    mock_llm_client_instance.get_completion = AsyncMock()

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    result_state = await probe(initial_state)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    assert "OpenAI API key is not configured" in result_state.error_message

# --- Tests for summarize node --- 

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_summarize_node_success(mock_llm_client_instance):
    """Test the summarize node successfully generates a summary."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="This is the mocked summary.")
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(
        history=[
            ("agent", "Q1"),
            ("user", "A1"),
            ("agent", "Q2"),
            ("user", "A2")
        ]
    )
    result_state = await summarize(initial_state)

    assert result_state.summary == "This is the mocked summary."
    assert result_state.error_message is None
    mock_llm_client_instance.get_completion.assert_called_once()
    call_args, _ = mock_llm_client_instance.get_completion.call_args
    prompt_arg = call_args[0]
    assert "A1" in prompt_arg
    assert "A2" in prompt_arg

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_summarize_node_llm_error(mock_llm_client_instance):
    """Test summarize node handling of LLM exceptions."""
    test_exception = Exception("Summarization Failed")
    mock_llm_client_instance.get_completion = AsyncMock(side_effect=test_exception)
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    result_state = await summarize(initial_state)

    assert "(Summary generation encountered an error.)" in result_state.summary
    assert "Error generating summary" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_summarize_node_llm_empty_response(mock_llm_client_instance):
    """Test summarize node handling of empty LLM response."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="") # Empty response
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    result_state = await summarize(initial_state)

    assert "(Summary generation failed - empty response)" in result_state.summary
    assert "LLM failed to generate a summary" in result_state.error_message

@pytest.mark.asyncio
async def test_summarize_node_empty_history():
    """Test summarize node handling of empty history."""
    initial_state = AgentState(history=[])
    result_state = await summarize(initial_state)

    assert result_state.summary is None
    assert "Internal Error: Summarizer requires history." in result_state.error_message

# --- Tests for check_summary node --- 

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_check_summary_node_approves(mock_llm_client_instance):
    """Test check_summary node when LLM approves (responds YES)."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value=" YES ")
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(
        history=[("user", "input")],
        summary="A valid summary."
    )
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is False
    assert result_state.error_message is None
    mock_llm_client_instance.get_completion.assert_called_once()
    call_args, _ = mock_llm_client_instance.get_completion.call_args
    prompt_arg = call_args[0]
    assert "A valid summary." in prompt_arg
    assert "input" in prompt_arg

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_check_summary_node_rejects(mock_llm_client_instance):
    """Test check_summary node when LLM rejects (responds NO or other)."""
    mock_llm_client_instance.get_completion = AsyncMock(return_value="NO, it is not relevant.")
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(
        history=[("user", "input")],
        summary="A poor summary."
    )
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is True
    assert "Summary quality check indicated potential issues" in result_state.error_message
    assert "NO, IT IS NOT RELEVANT." in result_state.error_message # Check response in message

@pytest.mark.asyncio
@patch('app.orchestration.graph.llm_client')
async def test_check_summary_node_llm_error(mock_llm_client_instance):
    """Test check_summary node handling of LLM exceptions."""
    test_exception = Exception("Check Failed")
    mock_llm_client_instance.get_completion = AsyncMock(side_effect=test_exception)
    mock_llm_client_instance.api_key = "fake_key"

    initial_state = AgentState(history=[("user", "input")], summary="Valid summary.")
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is True # Default to True on error
    assert "Error checking summary quality" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
async def test_check_summary_node_no_summary():
    """Test check_summary node when summary is missing."""
    initial_state = AgentState(history=[("user", "input")], summary=None)
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is True
    assert "Summary check skipped due to missing or failed summary" in result_state.error_message

@pytest.mark.asyncio
async def test_check_summary_node_failed_summary():
    """Test check_summary node when summary indicates failure."""
    initial_state = AgentState(history=[("user", "input")], summary="(Summary generation failed)")
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is True
    assert "Summary check skipped due to missing or failed summary" in result_state.error_message

@pytest.mark.asyncio
async def test_check_summary_node_no_history():
    """Test check_summary node when history is missing."""
    initial_state = AgentState(history=[], summary="Valid summary.")
    result_state = await check_summary(initial_state)

    assert result_state.needs_correction is True
    assert "Summary check skipped due to missing history." in result_state.error_message

# --- Tests for route_after_summary_check --- 

def test_route_after_summary_check_correction_needed():
    """Test routing function when needs_correction is True."""
    state = AgentState(needs_correction=True)
    next_node = route_after_summary_check(state)
    assert next_node == "summarize"

def test_route_after_summary_check_correction_not_needed():
    """Test routing function when needs_correction is False."""
    state = AgentState(needs_correction=False)
    next_node = route_after_summary_check(state)
    assert next_node == END

def test_route_after_summary_check_skipped_check():
    """Test routing function when summary check was skipped."""
    state = AgentState(needs_correction=True, error_message="Summary check skipped due to...")
    next_node = route_after_summary_check(state)
    assert next_node == END
