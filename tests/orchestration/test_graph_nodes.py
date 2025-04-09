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
from app.services.llm_client import LLMClient # Import the actual client for type hinting if needed

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
def mock_llm_client() -> AsyncMock: # Add type hint for clarity
    """Fixture to create a mock LLMClient instance."""
    mock_client = AsyncMock(spec=LLMClient) # Use spec for better mocking
    mock_client.get_completion = AsyncMock(return_value="Mocked follow-up question?")
    # Set api_key attribute to simulate configured client
    mock_client.api_key = "fake_key"
    return mock_client

@pytest.mark.asyncio
# Remove @patch decorator
async def test_probe_node_success(mock_llm_client: AsyncMock): # Inject the fixture
    """Test the probe node successful execution with user input."""
    # Configure the mock directly if needed (though fixture sets defaults)
    mock_llm_client.get_completion.return_value = "Mocked follow-up question?"

    initial_state = AgentState(
        history=[
            ("agent", "Initial question?"),
            ("user", "This is my user response.")
        ]
    )
    # Pass the mock client to the node function
    result_state = await probe(initial_state, llm_client=mock_llm_client)

    expected_question = "Mocked follow-up question?"
    assert result_state.current_question == expected_question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", expected_question)
    assert result_state.error_message is None

    # Verify the mock was called correctly
    mock_llm_client.get_completion.assert_called_once()
    call_args, _ = mock_llm_client.get_completion.call_args
    prompt_arg = call_args[0]
    assert "This is my user response." in prompt_arg
    assert "Initial question?" in prompt_arg

@pytest.mark.asyncio
# Remove @patch decorator
async def test_probe_node_llm_error(mock_llm_client: AsyncMock): # Inject the fixture
    """Test the probe node when the LLM call raises an exception."""
    test_exception = Exception("LLM Unavailable")
    # Configure the mock's side effect
    mock_llm_client.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock client
    result_state = await probe(initial_state, llm_client=mock_llm_client)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
# Remove @patch decorator
async def test_probe_node_llm_empty_response(mock_llm_client: AsyncMock): # Inject the fixture
    """Test the probe node when the LLM returns an empty string."""
    # Configure the mock's return value
    mock_llm_client.get_completion.return_value = ""

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock client
    result_state = await probe(initial_state, llm_client=mock_llm_client)

    assert result_state.current_question == "Could you please elaborate on that?" # Check specific fallback
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "LLM failed to generate a specific question" in result_state.error_message

@pytest.mark.asyncio
# Remove @patch decorator
async def test_probe_node_no_user_input_last(mock_llm_client: AsyncMock): # Inject the fixture
    """Test probe node behavior when the last turn wasn't the user."""
    mock_llm_client.get_completion.return_value = "Generated Question"

    initial_state = AgentState(history=[("agent", "Q1"), ("agent", "Q2")]) # No user turn last
    # Pass the mock client
    result_state = await probe(initial_state, llm_client=mock_llm_client)

    assert result_state.current_question == "Generated Question"
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert result_state.error_message is None # Should still proceed

    # Verify prompt was generic
    mock_llm_client.get_completion.assert_called_once_with(
        "Ask a generic open-ended question to encourage further reflection."
    )

@pytest.mark.asyncio
# probe doesn't use the client if history is empty, no mock needed
async def test_probe_node_empty_history(): 
    """Test probe node behavior with empty history."""
    initial_state = AgentState(history=[])
    # Pass None or a dummy object if required, but probe should handle empty history first
    # Creating a dummy mock to satisfy the signature if strictly needed, 
    # but ideally the function handles the case before using the client.
    # Let's assume probe checks history before using llm_client.
    # If it failed, we would need to pass a mock here.
    result_state = await probe(initial_state, llm_client=AsyncMock()) # Pass a dummy mock

    assert result_state.current_question is None
    assert result_state.history == [] # History remains empty
    assert result_state.error_message == "Internal Error: Prober requires history."

@pytest.mark.asyncio
# Remove @patch decorator
async def test_probe_node_llm_api_key_missing(mock_llm_client: AsyncMock): # Inject the fixture
    """Test probe node when LLMClient API key is missing at call time."""
    mock_llm_client.api_key = None # Simulate missing API key
    # Mock get_completion shouldn't be called, but configure it just in case
    mock_llm_client.get_completion.return_value = "Should not be called"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the modified mock client
    result_state = await probe(initial_state, llm_client=mock_llm_client)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    assert "OpenAI API key is not configured" in result_state.error_message
    mock_llm_client.get_completion.assert_not_called() # Verify it wasn't called

# --- Tests for summarize node --- 

@pytest.mark.asyncio
# Remove @patch decorator
async def test_summarize_node_success(mock_llm_client: AsyncMock): # Inject the fixture
    """Test the summarize node successfully generates a summary."""
    mock_llm_client.get_completion.return_value = "This is the mocked summary."

    initial_state = AgentState(
        history=[
            ("agent", "Q1"),
            ("user", "A1"),
            ("agent", "Q2"),
            ("user", "A2")
        ]
    )
    # Pass the mock client
    result_state = await summarize(initial_state, llm_client=mock_llm_client)

    assert result_state.summary == "This is the mocked summary."
    assert result_state.error_message is None
    assert result_state.correction_attempts == 1 # First attempt
    mock_llm_client.get_completion.assert_called_once()
    call_args, _ = mock_llm_client.get_completion.call_args
    prompt_arg = call_args[0]
    assert "A1" in prompt_arg
    assert "A2" in prompt_arg
    assert "PREVIOUS ATTEMPT FEEDBACK" not in prompt_arg # Ensure no feedback on first try

@pytest.mark.asyncio
async def test_summarize_node_with_feedback(mock_llm_client: AsyncMock):
    """Test summarize node incorporates feedback into the prompt."""
    feedback = "Missing the part about the deadline."
    mock_llm_client.get_completion.return_value = "Revised summary including deadline."

    initial_state = AgentState(
        history=[("agent", "Q"), ("user", "A")],
        correction_feedback=feedback,
        correction_attempts=1 # Simulating entry for the 2nd attempt
    )
    result_state = await summarize(initial_state, llm_client=mock_llm_client)

    assert result_state.summary == "Revised summary including deadline."
    assert result_state.error_message is None
    assert result_state.correction_attempts == 2 # Incremented
    mock_llm_client.get_completion.assert_called_once()
    call_args, _ = mock_llm_client.get_completion.call_args
    prompt_arg = call_args[0]
    assert feedback in prompt_arg
    assert "PREVIOUS ATTEMPT FEEDBACK" in prompt_arg

@pytest.mark.asyncio
# Remove @patch decorator
async def test_summarize_node_llm_error(mock_llm_client: AsyncMock): # Inject the fixture
    """Test summarize node handling of LLM exceptions."""
    test_exception = Exception("Summarization Failed")
    mock_llm_client.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock client
    result_state = await summarize(initial_state, llm_client=mock_llm_client)

    # Check for attempt number in the summary string
    assert f"(Summary generation attempt {result_state.correction_attempts} encountered an error.)" in result_state.summary
    assert result_state.correction_attempts == 1
    assert "Error generating summary" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
# Remove @patch decorator
async def test_summarize_node_llm_empty_response(mock_llm_client: AsyncMock): # Inject the fixture
    """Test summarize node handling of empty LLM response."""
    mock_llm_client.get_completion.return_value = "" # Empty response

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock client
    result_state = await summarize(initial_state, llm_client=mock_llm_client)

    # Check for attempt number in the summary string
    assert f"(Summary generation attempt {result_state.correction_attempts} failed - empty response)" in result_state.summary
    assert result_state.correction_attempts == 1
    assert "LLM failed to generate a summary" in result_state.error_message

@pytest.mark.asyncio
# summarize doesn't use the client if history is empty
async def test_summarize_node_empty_history():
    """Test summarize node handling of empty history."""
    initial_state = AgentState(history=[])
    # Pass dummy mock
    result_state = await summarize(initial_state, llm_client=AsyncMock())

    # Check for the specific skipped summary string
    assert result_state.summary == "(Summary generation skipped: No history)"
    assert result_state.correction_attempts == 1 # Still increments attempt
    assert "Internal Error: Summarizer requires history." in result_state.error_message

# --- Tests for check_summary node --- 

@pytest.mark.asyncio
# Remove @patch decorator
async def test_check_summary_node_approves(mock_llm_client: AsyncMock): # Inject the fixture
    """Test check_summary node when LLM approves (responds YES)."""
    mock_llm_client.get_completion.return_value = " YES "

    initial_state = AgentState(
        history=[("user", "input")],
        summary="A valid summary."
    )
    # Pass the mock client
    result_state = await check_summary(initial_state, llm_client=mock_llm_client)

    assert result_state.needs_correction is False
    assert result_state.correction_feedback is None
    assert result_state.error_message is None # Should remain None on success
    mock_llm_client.get_completion.assert_called_once()
    call_args, kwargs = mock_llm_client.get_completion.call_args
    prompt_arg = call_args[0]
    assert "A valid summary." in prompt_arg
    assert "input" in prompt_arg
    assert kwargs.get("model") == "gpt-4o-mini" # Verify model used for check

@pytest.mark.asyncio
# Remove @patch decorator
async def test_check_summary_node_rejects(mock_llm_client: AsyncMock): # Inject the fixture
    """Test check_summary node when LLM rejects (responds NO or other)."""
    response_text = "NO, it is not relevant."
    expected_feedback = "it is not relevant"
    mock_llm_client.get_completion.return_value = response_text

    initial_state = AgentState(
        history=[("user", "input")],
        summary="Summary to reject."
    )
    # Pass the mock client
    result_state = await check_summary(initial_state, llm_client=mock_llm_client)

    assert result_state.needs_correction is True
    assert result_state.correction_feedback == expected_feedback # Check feedback is extracted
    assert result_state.error_message is None # Error message not set, only feedback
    mock_llm_client.get_completion.assert_called_once()

@pytest.mark.asyncio
# Remove @patch decorator
async def test_check_summary_node_llm_error(mock_llm_client: AsyncMock): # Inject the fixture
    """Test check_summary node handling of LLM exceptions."""
    test_exception = Exception("Check Failed")
    mock_llm_client.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("user", "input")], summary="A summary.")
    # Pass the mock client
    result_state = await check_summary(initial_state, llm_client=mock_llm_client)

    assert result_state.needs_correction is True # Default to correction needed on error
    assert "Error checking summary quality" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
# check_summary doesn't use client if summary invalid
async def test_check_summary_node_no_summary():
    """Test check_summary skips check if no summary exists."""
    initial_state = AgentState(history=[("user", "input")], summary=None)
    # Pass dummy mock
    result_state = await check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True # Default
    assert "Summary check skipped" in result_state.error_message
    # Assert LLMClient was not called if possible (difficult without patching)

@pytest.mark.asyncio
# check_summary doesn't use client if summary invalid
async def test_check_summary_node_failed_summary():
    """Test check_summary skips check if summary indicates failure."""
    initial_state = AgentState(
        history=[("user", "input")], 
        summary="(Summary generation failed)"
    )
    # Pass dummy mock
    result_state = await check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True # Default
    assert "Summary check skipped" in result_state.error_message

@pytest.mark.asyncio
# check_summary doesn't use client if history missing
async def test_check_summary_node_no_history():
    """Test check_summary skips check if history is missing."""
    initial_state = AgentState(history=[], summary="A summary.")
    # Pass dummy mock
    result_state = await check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True # Default
    assert "Summary check skipped due to missing history." in result_state.error_message

# --- Tests for routing logic --- 

def test_route_after_summary_check_correction_needed():
    """Test routing when correction is needed."""
    state = AgentState(needs_correction=True)
    assert route_after_summary_check(state) == "summarize"

def test_route_after_summary_check_correction_not_needed():
    """Test routing when correction is not needed."""
    state = AgentState(needs_correction=False)
    assert route_after_summary_check(state) == END

def test_route_after_summary_check_skipped_check():
    """Test routing when the check was skipped (e.g., no summary)."""
    state = AgentState(needs_correction=True, error_message="Summary check skipped due to missing summary.")
    assert route_after_summary_check(state) == END
