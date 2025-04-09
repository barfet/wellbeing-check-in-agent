import pytest
import os
from unittest.mock import patch, AsyncMock

# Ensure the src directory is in the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.orchestration.state import AgentState
# Import the logic functions and interface directly for isolated testing
from app.orchestration.graph_logic import (
    run_initiate,
    run_probe,
    run_summarize,
    run_check_summary,
    run_classify_sentiment, # Added
    # Routing functions need separate tests
    route_after_summary_check, 
    should_continue_probing_route,
    handle_summary_feedback_route
)
from langgraph.graph import END # Import END for routing check
from app.services.llm_client import LLMInterface # Import the interface for mocking

# --- Tests for run_initiate node --- 

@pytest.mark.asyncio
async def test_run_initiate_node_with_topic(): # Renamed test
    """Test the initiate node when a topic is provided."""
    initial_state = AgentState(topic="My Test Topic")
    result_state = await run_initiate(initial_state)

    expected_question = "Okay, let's reflect on 'My Test Topic'. To start, could you tell me briefly what happened regarding this?"
    assert result_state.current_question == expected_question
    assert result_state.history == [("agent", expected_question)]
    assert result_state.topic == "My Test Topic"
    assert result_state.error_message is None

@pytest.mark.asyncio
async def test_run_initiate_node_without_topic(): # Renamed test
    """Test the initiate node when no topic is provided."""
    initial_state = AgentState() # No topic
    result_state = await run_initiate(initial_state)

    expected_question = "Hello! What topic or experience would you like to reflect on today?"
    assert result_state.current_question == expected_question
    assert result_state.history == [("agent", expected_question)]
    assert result_state.topic is None
    assert result_state.error_message is None

# --- Tests for run_probe node --- 

@pytest.fixture
def mock_llm_interface() -> AsyncMock: # Renamed fixture, mock interface
    """Fixture to create a mock LLMInterface instance."""
    mock_client = AsyncMock(spec=LLMInterface) # Use spec for the interface
    mock_client.get_completion = AsyncMock(return_value="Mocked follow-up question?")
    # Set api_key attribute to simulate configured client
    mock_client.api_key = "fake_key"
    return mock_client

@pytest.mark.asyncio
async def test_run_probe_node_success(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test the probe node successful execution with user input."""
    # Configure the mock directly if needed (though fixture sets defaults)
    mock_llm_interface.get_completion.return_value = "Mocked follow-up question?"

    initial_state = AgentState(
        history=[
            ("agent", "Initial question?"),
            ("user", "This is my user response.")
        ]
    )
    # Pass the mock interface to the node function
    result_state = await run_probe(initial_state, llm_client=mock_llm_interface)

    expected_question = "Mocked follow-up question?"
    assert result_state.current_question == expected_question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", expected_question)
    assert result_state.error_message is None

    # Verify the mock was called correctly
    mock_llm_interface.get_completion.assert_called_once()
    call_args, _ = mock_llm_interface.get_completion.call_args
    prompt_arg = call_args[0]
    # Check prompt content (using the actual prompt function logic)
    assert "Based on the following conversation history:" in prompt_arg
    assert "agent: Initial question?" in prompt_arg
    assert "user: This is my user response." in prompt_arg
    assert "Ask the user *one* relevant, open-ended follow-up question" in prompt_arg

@pytest.mark.asyncio
async def test_run_probe_node_llm_error(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test the probe node when the LLM call raises an exception."""
    test_exception = Exception("LLM Unavailable")
    # Configure the mock's side effect
    mock_llm_interface.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock interface
    result_state = await run_probe(initial_state, llm_client=mock_llm_interface)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
async def test_run_probe_node_llm_empty_response(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test the probe node when the LLM returns an empty string."""
    # Configure the mock's return value
    mock_llm_interface.get_completion.return_value = ""

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock interface
    result_state = await run_probe(initial_state, llm_client=mock_llm_interface)

    assert result_state.current_question == "Could you please elaborate on that?" # Check specific fallback
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "LLM failed to generate a specific question" in result_state.error_message

@pytest.mark.asyncio
async def test_run_probe_node_no_user_input_last(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test probe node behavior when the last turn wasn't the user."""
    mock_llm_interface.get_completion.return_value = "Generated Question"

    initial_state = AgentState(history=[("agent", "Q1"), ("agent", "Q2")]) # No user turn last
    # Pass the mock interface
    result_state = await run_probe(initial_state, llm_client=mock_llm_interface)

    assert result_state.current_question == "Generated Question"
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert result_state.error_message is None # Should still proceed

    # Verify prompt was generic (using the actual prompt function logic)
    mock_llm_interface.get_completion.assert_called_once()
    call_args, _ = mock_llm_interface.get_completion.call_args
    prompt_arg = call_args[0]
    assert "Ask a generic open-ended question to encourage further reflection." in prompt_arg

@pytest.mark.asyncio
async def test_run_probe_node_empty_history(): # Renamed test
    """Test probe node behavior with empty history."""
    initial_state = AgentState(history=[])
    # run_probe checks history before using llm_client
    result_state = await run_probe(initial_state, llm_client=AsyncMock()) # Pass a dummy mock

    assert result_state.current_question is None
    assert result_state.history == [] # History remains empty
    assert result_state.error_message == "Internal Error: Prober requires history."

@pytest.mark.asyncio
async def test_run_probe_node_llm_api_key_missing(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test probe node when LLMInterface API key is missing at call time."""
    mock_llm_interface.api_key = None # Simulate missing API key
    mock_llm_interface.get_completion.return_value = "Should not be called"

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the modified mock interface
    result_state = await run_probe(initial_state, llm_client=mock_llm_interface)

    assert "I encountered an issue." in result_state.current_question # Check fallback question
    assert len(result_state.history) == 3
    assert result_state.history[-1] == ("agent", result_state.current_question)
    assert "Error generating follow-up" in result_state.error_message
    # Check the specific ValueError raised in the logic function
    assert "LLM Interface API key is not configured." in result_state.error_message
    mock_llm_interface.get_completion.assert_not_called() # Verify it wasn't called

# --- Tests for run_summarize node --- 

@pytest.mark.asyncio
async def test_run_summarize_node_success(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test the summarize node successfully generates a summary."""
    mock_llm_interface.get_completion.return_value = "This is the mocked summary."

    initial_state = AgentState(
        history=[
            ("agent", "Q1"),
            ("user", "A1"),
            ("agent", "Q2"),
            ("user", "A2")
        ]
    )
    # Pass the mock interface
    result_state = await run_summarize(initial_state, llm_client=mock_llm_interface)

    assert result_state.summary == "This is the mocked summary."
    assert result_state.error_message is None
    assert result_state.correction_attempts == 1 # First attempt
    mock_llm_interface.get_completion.assert_called_once()
    call_args, _ = mock_llm_interface.get_completion.call_args
    prompt_arg = call_args[0]
    assert "A1" in prompt_arg
    assert "A2" in prompt_arg
    assert "PREVIOUS ATTEMPT FEEDBACK" not in prompt_arg # Ensure no feedback on first try

@pytest.mark.asyncio
async def test_run_summarize_node_with_feedback(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test summarize node incorporates feedback into the prompt."""
    feedback = "Missing the part about the deadline."
    mock_llm_interface.get_completion.return_value = "Revised summary including deadline."

    initial_state = AgentState(
        history=[("agent", "Q"), ("user", "A")],
        correction_feedback=feedback,
        correction_attempts=1 # Simulating entry for the 2nd attempt
    )
    result_state = await run_summarize(initial_state, llm_client=mock_llm_interface)

    assert result_state.summary == "Revised summary including deadline."
    assert result_state.error_message is None
    assert result_state.correction_attempts == 2 # Incremented
    mock_llm_interface.get_completion.assert_called_once()
    call_args, _ = mock_llm_interface.get_completion.call_args
    prompt_arg = call_args[0]
    assert feedback in prompt_arg
    assert "PREVIOUS ATTEMPT FEEDBACK" in prompt_arg

@pytest.mark.asyncio
async def test_run_summarize_node_llm_error(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test summarize node handling of LLM exceptions."""
    test_exception = Exception("Summarization Failed")
    mock_llm_interface.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock interface
    result_state = await run_summarize(initial_state, llm_client=mock_llm_interface)

    # Check for attempt number in the summary string
    assert f"(Summary generation attempt {result_state.correction_attempts} encountered an error.)" in result_state.summary
    assert result_state.correction_attempts == 1
    assert "Error generating summary" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
async def test_run_summarize_node_llm_empty_response(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test summarize node handling of empty LLM response."""
    mock_llm_interface.get_completion.return_value = "" # Empty response

    initial_state = AgentState(history=[("agent", "Q1"), ("user", "A1")])
    # Pass the mock interface
    result_state = await run_summarize(initial_state, llm_client=mock_llm_interface)

    # Check for attempt number in the summary string
    assert f"(Summary generation attempt {result_state.correction_attempts} failed - empty response)" in result_state.summary
    assert result_state.correction_attempts == 1
    assert "LLM failed to generate a summary." in result_state.error_message

@pytest.mark.asyncio
async def test_run_summarize_node_empty_history(): # Renamed test
    """Test summarize node behavior with empty history."""
    initial_state = AgentState(history=[])
    # run_summarize checks history before using llm_client
    result_state = await run_summarize(initial_state, llm_client=AsyncMock()) # Pass dummy mock

    assert result_state.summary == "(Summary generation skipped: No history)"
    assert result_state.error_message == "Internal Error: Summarizer requires history."

# --- Tests for run_check_summary node --- 

@pytest.mark.asyncio
async def test_run_check_summary_node_approves(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test check_summary node when LLM approves the summary."""
    mock_llm_interface.get_completion.return_value = "YES"

    initial_state = AgentState(
        history=[("agent", "Q"), ("user", "A")],
        summary="Good summary."
    )
    # Pass the mock interface
    result_state = await run_check_summary(initial_state, llm_client=mock_llm_interface)

    assert result_state.needs_correction is False
    assert result_state.correction_feedback is None
    assert result_state.error_message is None
    # Verify LLM was called with the correct prompt and model
    mock_llm_interface.get_completion.assert_called_once()
    call_args, call_kwargs = mock_llm_interface.get_completion.call_args
    prompt_arg = call_args[0]
    assert "SUMMARY:\nGood summary." in prompt_arg
    assert call_kwargs.get('model') == 'gpt-4o-mini'

@pytest.mark.asyncio
async def test_run_check_summary_node_rejects(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test check_summary node when LLM rejects and provides feedback."""
    feedback = "It missed the key point about X."
    mock_llm_interface.get_completion.return_value = f"NO, {feedback}"

    initial_state = AgentState(history=[("agent", "Q"), ("user", "A")], summary="Flawed summary.")
    # Pass the mock interface
    result_state = await run_check_summary(initial_state, llm_client=mock_llm_interface)

    assert result_state.needs_correction is True
    assert result_state.correction_feedback == feedback
    assert result_state.error_message is None

@pytest.mark.asyncio
async def test_run_check_summary_node_llm_error(mock_llm_interface: AsyncMock): # Renamed test, use new fixture
    """Test check_summary node handling of LLM exceptions."""
    test_exception = Exception("Check Failed")
    mock_llm_interface.get_completion.side_effect = test_exception

    initial_state = AgentState(history=[("agent", "Q"), ("user", "A")], summary="Summary to check.")
    # Pass the mock interface
    result_state = await run_check_summary(initial_state, llm_client=mock_llm_interface)

    assert result_state.needs_correction is True # Default on error
    assert "Failed to perform summary check" in result_state.correction_feedback
    assert "Error checking summary quality" in result_state.error_message
    assert str(test_exception) in result_state.error_message

@pytest.mark.asyncio
async def test_run_check_summary_node_no_summary(): # Renamed test
    """Test check_summary behavior when summary is missing."""
    initial_state = AgentState(history=[("agent", "Q"), ("user", "A")], summary=None)
    # Doesn't need LLM client
    result_state = await run_check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True # Default state
    assert result_state.correction_feedback is None
    assert "Summary check skipped due to missing/failed summary." in result_state.error_message

@pytest.mark.asyncio
async def test_run_check_summary_node_failed_summary(): # Renamed test
    """Test check_summary behavior with a previously failed summary placeholder."""
    initial_state = AgentState(history=[("agent", "Q"), ("user", "A")], summary="(Summary generation failed)")
    # Doesn't need LLM client
    result_state = await run_check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True
    assert result_state.correction_feedback is None
    assert "Summary check skipped due to missing/failed summary." in result_state.error_message

@pytest.mark.asyncio
async def test_run_check_summary_node_no_history(): # Renamed test
    """Test check_summary behavior when history is missing."""
    initial_state = AgentState(history=[], summary="A summary")
    # Doesn't need LLM client
    result_state = await run_check_summary(initial_state, llm_client=AsyncMock())

    assert result_state.needs_correction is True
    assert result_state.correction_feedback is None
    assert "Summary check skipped due to missing history." in result_state.error_message

# --- Tests for route_after_summary_check --- 
# This is synchronous and doesn't need LLM mock

def test_route_after_summary_check_correction_needed():
    """Test routing when correction is needed and attempts remain."""
    state = AgentState(needs_correction=True, correction_attempts=1) # 1 attempt made, 2 allowed
    next_node = route_after_summary_check(state)
    assert next_node == "SUMMARIZE" # Should route back to summarize

def test_route_after_summary_check_correction_not_needed():
    """Test routing when correction is not needed."""
    state = AgentState(needs_correction=False, correction_attempts=1)
    next_node = route_after_summary_check(state)
    assert next_node == END

def test_route_after_summary_check_max_attempts_reached():
    """Test routing when correction is needed but max attempts are reached."""
    # Assuming MAX_CORRECTION_ATTEMPTS = 2 (so 3 total attempts: 0, 1, 2)
    # This state means 3 attempts (0, 1, 2) have already been made
    state = AgentState(needs_correction=True, correction_attempts=3)
    next_node = route_after_summary_check(state)
    assert next_node == END
    assert "Summary failed validation" in state.error_message

def test_route_after_summary_check_skipped_check():
    """Test routing when the summary check was skipped."""
    state = AgentState(error_message="Summary check skipped due to missing history.")
    next_node = route_after_summary_check(state)
    assert next_node == END

# --- Tests for run_classify_sentiment --- 

@pytest.mark.asyncio
async def test_run_classify_sentiment_success(mock_llm_interface: AsyncMock):
    """Test successful sentiment classification."""
    mock_llm_interface.get_completion.return_value = " positive " # Test stripping/lowering
    initial_state = AgentState(history=[("agent", "Q"), ("user", "This is great!")])
    result_state = await run_classify_sentiment(initial_state, llm_client=mock_llm_interface)

    assert result_state.last_sentiment == "positive"
    assert result_state.error_message is None
    mock_llm_interface.get_completion.assert_called_once()
    prompt_arg = mock_llm_interface.get_completion.call_args[0][0]
    assert 'User Message: "This is great!"' in prompt_arg

@pytest.mark.asyncio
async def test_run_classify_sentiment_invalid_response(mock_llm_interface: AsyncMock):
    """Test sentiment classification with unexpected LLM response."""
    mock_llm_interface.get_completion.return_value = "happy"
    initial_state = AgentState(history=[("agent", "Q"), ("user", "Feeling okay.")])
    result_state = await run_classify_sentiment(initial_state, llm_client=mock_llm_interface)

    assert result_state.last_sentiment == "neutral" # Should default
    assert result_state.error_message is None # Logged as warning

@pytest.mark.asyncio
async def test_run_classify_sentiment_llm_error(mock_llm_interface: AsyncMock):
    """Test sentiment classification handling LLM errors."""
    mock_llm_interface.get_completion.side_effect = Exception("Sentiment API failed")
    initial_state = AgentState(history=[("agent", "Q"), ("user", "Not good.")])
    result_state = await run_classify_sentiment(initial_state, llm_client=mock_llm_interface)

    assert result_state.last_sentiment == "neutral" # Should default on error
    assert result_state.error_message is None # Error logged, not set in state typically

@pytest.mark.asyncio
async def test_run_classify_sentiment_no_user_message(mock_llm_interface: AsyncMock):
    """Test sentiment classification when last message is not from user."""
    initial_state = AgentState(history=[("agent", "Q1"), ("agent", "Q2")])
    result_state = await run_classify_sentiment(initial_state, llm_client=mock_llm_interface)

    assert result_state.last_sentiment == "neutral"
    mock_llm_interface.get_completion.assert_not_called()

@pytest.mark.asyncio
async def test_run_classify_sentiment_empty_history(mock_llm_interface: AsyncMock):
    """Test sentiment classification with empty history."""
    initial_state = AgentState(history=[])
    result_state = await run_classify_sentiment(initial_state, llm_client=mock_llm_interface)

    assert result_state.last_sentiment == "neutral"
    mock_llm_interface.get_completion.assert_not_called()

# --- Tests for should_continue_probing_route --- 

# Define constants used in the route logic for testing
MAX_PROBE_ATTEMPTS_TEST = 5 # Match the constant used in graph_logic

@pytest.mark.asyncio
async def test_should_continue_probing_route_max_probes(mock_llm_interface: AsyncMock):
    """Test routing to SUMMARIZE when max probes are reached."""
    state = AgentState(probe_count=MAX_PROBE_ATTEMPTS_TEST, history=[("a","b"),("u","c"),("a","d")])
    # Should not call LLM
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "SUMMARIZE"
    mock_llm_interface.get_completion.assert_not_called()

@pytest.mark.asyncio
async def test_should_continue_probing_route_short_history(mock_llm_interface: AsyncMock):
    """Test routing to PROBE when history is too short."""
    state = AgentState(probe_count=1, history=[("a","b")]) # Only 1 turn
    # Should not call LLM
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "PROBE"
    mock_llm_interface.get_completion.assert_not_called()

@pytest.mark.asyncio
async def test_should_continue_probing_route_llm_yes(mock_llm_interface: AsyncMock):
    """Test routing to SUMMARIZE when LLM assesses depth as sufficient (YES)."""
    mock_llm_interface.get_completion.return_value = " YES, it looks deep enough. "
    state = AgentState(probe_count=1, history=[("a","b"),("u","c"),("a","d")])
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "SUMMARIZE"
    mock_llm_interface.get_completion.assert_called_once()
    # Check correct model was used (optional, depends on logic detail)
    call_args, call_kwargs = mock_llm_interface.get_completion.call_args
    assert call_kwargs.get('model') == 'gpt-3.5-turbo'

@pytest.mark.asyncio
async def test_should_continue_probing_route_llm_no(mock_llm_interface: AsyncMock):
    """Test routing to PROBE when LLM assesses depth as insufficient (NO)."""
    mock_llm_interface.get_completion.return_value = "no"
    state = AgentState(probe_count=1, history=[("a","b"),("u","c"),("a","d")])
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "PROBE"
    mock_llm_interface.get_completion.assert_called_once()

@pytest.mark.asyncio
async def test_should_continue_probing_route_llm_error(mock_llm_interface: AsyncMock):
    """Test routing to PROBE when the LLM call fails."""
    mock_llm_interface.get_completion.side_effect = Exception("Depth check failed")
    state = AgentState(probe_count=1, history=[("a","b"),("u","c"),("a","d")])
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "PROBE" # Default to probe on error
    mock_llm_interface.get_completion.assert_called_once()

@pytest.mark.asyncio
async def test_should_continue_probing_route_api_key_missing(mock_llm_interface: AsyncMock):
    """Test routing to PROBE when the API key is missing."""
    mock_llm_interface.api_key = None
    state = AgentState(probe_count=1, history=[("a","b"),("u","c"),("a","d")])
    result = await should_continue_probing_route(state, mock_llm_interface)
    assert result == "PROBE" # Default to probe on error
    mock_llm_interface.get_completion.assert_not_called()

# --- Tests for handle_summary_feedback_route --- 
# This is synchronous and does not require LLM mock

def test_handle_summary_feedback_route_accept():
    """Test routing to 'end' when user accepts summary."""
    state = AgentState(history=[("agent","Summary..."), ("user", "Yes, that looks good thanks")])
    result = handle_summary_feedback_route(state)
    assert result == "end"

def test_handle_summary_feedback_route_continue():
    """Test routing to 'continue' when user wants to add more."""
    state = AgentState(history=[("agent","Summary..."), ("user", "actually, can we add something?")])
    result = handle_summary_feedback_route(state)
    assert result == "continue"

def test_handle_summary_feedback_route_ambiguous():
    """Test routing to 'end' when user feedback is ambiguous."""
    state = AgentState(history=[("agent","Summary..."), ("user", "Okay then")])
    result = handle_summary_feedback_route(state)
    assert result == "end"

def test_handle_summary_feedback_route_not_user_last():
    """Test routing to 'end' when last turn is not user."""
    state = AgentState(history=[("agent","Summary..."), ("agent", "Anything else?")])
    result = handle_summary_feedback_route(state)
    assert result == "end"

def test_handle_summary_feedback_route_empty_history():
    """Test routing to 'end' with empty history."""
    state = AgentState(history=[])
    result = handle_summary_feedback_route(state)
    assert result == "end" 