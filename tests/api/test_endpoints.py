import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
import os
import sys
from typing import AsyncIterator, Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.main import app
from app.orchestration.state import AgentState

BASE_URL = "http://test"
API_ENDPOINT = "/api/v1/reflections/turns"

# --- Helper for Mocking astream_events ---

async def mock_astream_events_generator(final_state_dict: Optional[Dict[str, Any]], 
                                      final_node_name: Optional[str] = None, 
                                      reached_end: bool = False) -> AsyncIterator[Dict[str, Any]]:
    """Simulates the event stream, yielding a final state snapshot associated with a node name."""
    # Simulate some intermediate events if needed (optional)
    # yield {"event": "on_node_start", "name": "some_node", ...}
    
    if final_state_dict and final_node_name:
        # Yield the crucial final state snapshot event, associated with the correct node
        yield {
            "event": "on_node_end", 
            "name": final_node_name, # Associate state with this node
            "data": {"chunk": final_state_dict}
        }
    elif final_state_dict: # If name not provided, use generic
         yield {
            "event": "on_node_end", 
            "name": "last_node_before_interrupt_or_end", 
            "data": {"chunk": final_state_dict}
         }

    if reached_end:
         yield {"event": "on_node_end", "tags": ["__end__"], "name": "__end__", "data": {}}
    # Stream ends naturally

# --- Test Fixtures --- 

@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
        yield client

@pytest.fixture
def initial_state_dict() -> dict:
    # State dict representing a point after initial question is asked
    return AgentState(
        topic="Test Topic",
        history=[("agent", "Initial question?")],
        current_question="Initial question?",
    ).model_dump()

# --- Test Cases --- 

@pytest.mark.asyncio
async def test_initiation_turn(async_client: AsyncClient):
    """Test the first turn (initiation) using streaming."""
    payload = {"topic": "Starting a reflection"}

    # Expected state after INITIATE -> wait_for_input (interrupt)
    expected_agent_question = "Okay, let's reflect on 'Starting a reflection'. To start, could you tell me briefly what happened regarding this?"
    expected_state_dict_at_interrupt = AgentState(
        topic="Starting a reflection",
        history=[("agent", expected_agent_question)],
        current_question=expected_agent_question,
        probe_count=0 # Initial value
    ).model_dump()

    # Patch astream_events to yield the state associated with INITIATE node end
    with patch("app.api.endpoints.app_graph.astream_events", 
               return_value=mock_astream_events_generator(expected_state_dict_at_interrupt, final_node_name="INITIATE")) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    
    # Assert API response reflects the expected state after interrupt
    assert data["agent_response"] == expected_agent_question
    assert not data["is_final_turn"]
    # Compare state fields individually, converting history for comparison
    returned_state = data["next_state"]
    expected_history_list = [list(item) for item in expected_state_dict_at_interrupt["history"]]
    assert returned_state["history"] == expected_history_list
    assert returned_state["topic"] == expected_state_dict_at_interrupt["topic"]
    assert returned_state["current_question"] == expected_state_dict_at_interrupt["current_question"]
    assert returned_state["probe_count"] == expected_state_dict_at_interrupt["probe_count"]
    
    # Assert astream_events was called correctly
    mock_astream.assert_called_once()
    call_args, call_kwargs = mock_astream.call_args
    # Check input state passed was the initial empty state with topic
    assert isinstance(call_args[0], dict)
    invoked_state_dict = call_args[0]
    assert invoked_state_dict["topic"] == "Starting a reflection"
    assert invoked_state_dict["history"] == []
    assert invoked_state_dict["probe_count"] == 0

@pytest.mark.asyncio
async def test_subsequent_turn_probe(async_client: AsyncClient, initial_state_dict: dict):
    """Test a subsequent turn resulting in another probe question (interrupt)."""
    user_input = "I had a difficult conversation."
    current_state_before_turn = initial_state_dict.copy()
    
    payload = {"user_input": user_input, "current_state": current_state_before_turn}

    # Expected state dict when graph interrupts after PROBE generates question
    expected_state_dict_at_interrupt = AgentState(**current_state_before_turn)
    expected_state_dict_at_interrupt.history.append(("user", user_input))
    expected_probe_question = "Mocked probe question 2"
    expected_state_dict_at_interrupt.history.append(("agent", expected_probe_question))
    expected_state_dict_at_interrupt.current_question = expected_probe_question
    expected_state_dict_at_interrupt.probe_count += 1 
    expected_state_dict_at_interrupt = expected_state_dict_at_interrupt.model_dump()
    
    # Patch astream_events to yield state associated with PROBE node end
    with patch("app.api.endpoints.app_graph.astream_events", 
               return_value=mock_astream_events_generator(expected_state_dict_at_interrupt, final_node_name="PROBE")) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == expected_probe_question
    assert not data["is_final_turn"]
    # Compare relevant state fields individually
    returned_state = data["next_state"]
    expected_history_list = [list(item) for item in expected_state_dict_at_interrupt["history"]]
    assert returned_state["history"] == expected_history_list
    assert returned_state["current_question"] == expected_state_dict_at_interrupt["current_question"]
    assert returned_state["probe_count"] == expected_state_dict_at_interrupt["probe_count"]

    mock_astream.assert_called_once()
    call_args, _ = mock_astream.call_args
    # Check input state passed to stream includes the user input
    assert isinstance(call_args[0], dict)
    invoked_state_dict = call_args[0]
    assert invoked_state_dict["history"][-1] == ("user", user_input)
    assert invoked_state_dict["probe_count"] == initial_state_dict["probe_count"]

@pytest.mark.asyncio
async def test_subsequent_turn_triggers_summary(async_client: AsyncClient, initial_state_dict: dict):
    """Test turn leading to summary (graph reaches END)."""
    user_input = "Final user input before summary."
    current_state_before_turn = initial_state_dict.copy()
    current_state_before_turn["history"].append(("user", "Intermediate response"))
    current_state_before_turn["history"].append(("agent", "Intermediate question"))
    current_state_before_turn["probe_count"] = 2 # Set count

    payload = {"user_input": user_input, "current_state": current_state_before_turn}

    # Expected final state dict when graph reaches END after successful summary
    expected_state_dict_at_end = AgentState(**current_state_before_turn)
    expected_state_dict_at_end.history.append(("user", user_input))
    expected_state_dict_at_end.probe_count += 1 # Probe runs before deciding to summarize
    expected_state_dict_at_end.current_question = None # Cleared by summarize path
    expected_state_dict_at_end.summary = "Generated Mock Summary"
    expected_state_dict_at_end.needs_correction = False # Assume check passes
    expected_state_dict_at_end.correction_attempts = 0 # Assume reset on success
    expected_state_dict_at_end = expected_state_dict_at_end.model_dump()

    with patch("app.api.endpoints.app_graph.astream_events", return_value=mock_astream_events_generator(expected_state_dict_at_end, reached_end=True)) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == "Generated Mock Summary" 
    assert data["is_final_turn"] # Should be final as END was reached
    # assert data["next_state"] == expected_state_dict_at_end
    returned_state = data["next_state"]
    expected_history_list = [list(item) for item in expected_state_dict_at_end["history"]]
    assert returned_state["history"] == expected_history_list
    assert returned_state["summary"] == expected_state_dict_at_end["summary"]
    assert returned_state["needs_correction"] == expected_state_dict_at_end["needs_correction"]
    assert returned_state["correction_attempts"] == expected_state_dict_at_end["correction_attempts"]
    mock_astream.assert_called_once()

@pytest.mark.asyncio
async def test_invalid_current_state(async_client: AsyncClient):
    """Test sending an invalid structure for current_state."""
    # Test case 1: history is None
    payload1 = {"user_input": "Some input", "current_state": {"topic": "test", "history": None}}
    response1 = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload1)
    assert response1.status_code == 400
    data1 = response1.json()
    assert "Invalid current_state provided" in data1["detail"]

    # Test case 2: history is not a list
    payload2 = {"user_input": "Some input", "current_state": {"topic": "test", "history": "not a list"}}
    response2 = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload2)
    assert response2.status_code == 400
    data2 = response2.json()
    assert "Invalid current_state provided" in data2["detail"]

@pytest.mark.asyncio
async def test_graph_invocation_error(async_client: AsyncClient, initial_state_dict: dict):
    """Test handling when the graph streaming itself raises an error."""
    payload = {"user_input": "Trigger error", "current_state": initial_state_dict}
    error_message = "Graph simulation stream error"

    # Patch astream_events to raise an error
    with patch("app.api.endpoints.app_graph.astream_events", side_effect=Exception(error_message)) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 500
    data = response.json()
    assert "Internal server error processing reflection" in data["detail"]
    assert error_message in data["detail"]
    mock_astream.assert_called_once()

@pytest.mark.asyncio
async def test_summary_correction_loop_success(async_client: AsyncClient, initial_state_dict: dict):
    """Test the summary correction loop success path (reaches END)."""
    user_input = "Input triggering second summary attempt."
    # Simulate state after first check failed
    current_state_before_turn = initial_state_dict.copy()
    current_state_before_turn["history"].append(("user", "Input before first summary attempt"))
    current_state_before_turn["history"].append(("agent", "Probe Q leading to first summary"))
    current_state_before_turn["summary"] = "Initial flawed summary"
    current_state_before_turn["needs_correction"] = True
    current_state_before_turn["correction_feedback"] = "Missing key challenge."
    current_state_before_turn["correction_attempts"] = 1
    current_state_before_turn["probe_count"] = 3 # Assume irrelevant now
    
    payload = {"user_input": user_input, "current_state": current_state_before_turn}

    # Expected final state dict after graph reaches END on successful correction
    expected_state_dict_at_end = AgentState(**current_state_before_turn)
    expected_state_dict_at_end.history.append(("user", user_input))
    # Correction attempts gets reset by routing logic BEFORE end state is finalized
    expected_state_dict_at_end.correction_attempts = 0 
    expected_state_dict_at_end.correction_feedback = None # Cleared by check_summary
    expected_state_dict_at_end.current_question = None # Cleared by summarize path
    expected_state_dict_at_end.summary = "Improved Mock Summary"
    expected_state_dict_at_end.needs_correction = False # Check passes now
    expected_state_dict_at_end.error_message = None
    expected_state_dict_at_end = expected_state_dict_at_end.model_dump()

    # Mock stream to return the successful end state
    with patch("app.api.endpoints.app_graph.astream_events", return_value=mock_astream_events_generator(expected_state_dict_at_end, reached_end=True)) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == "Improved Mock Summary" 
    assert data["is_final_turn"]
    # assert data["next_state"] == expected_state_dict_at_end # Compare full state
    returned_state = data["next_state"]
    expected_history_list = [list(item) for item in expected_state_dict_at_end["history"]]
    assert returned_state["history"] == expected_history_list
    assert returned_state["summary"] == expected_state_dict_at_end["summary"]
    assert returned_state["needs_correction"] == expected_state_dict_at_end["needs_correction"]
    assert returned_state["correction_attempts"] == expected_state_dict_at_end["correction_attempts"]
    mock_astream.assert_called_once()

@pytest.mark.asyncio
async def test_summary_correction_loop_max_attempts(async_client: AsyncClient, initial_state_dict: dict):
    """Test the summary correction loop failing after max attempts (reaches END)."""
    user_input = "User input triggering final summary attempt."
    # Simulate state where check failed and max attempts are reached
    current_state_before_turn = initial_state_dict.copy()
    current_state_before_turn["history"].append(("user", "Input before last summary attempt"))
    current_state_before_turn["summary"] = "Still flawed summary"
    current_state_before_turn["needs_correction"] = True
    current_state_before_turn["correction_feedback"] = "Still inaccurate."
    # Entering the turn where the 3rd attempt will be made (attempts are 0-indexed in state logic)
    current_state_before_turn["correction_attempts"] = 2 # Max retries = 2
    current_state_before_turn["probe_count"] = 3
    
    payload = {"user_input": user_input, "current_state": current_state_before_turn}

    # Expected final state dict when graph reaches END after max attempts
    expected_state_dict_at_end = AgentState(**current_state_before_turn)
    expected_state_dict_at_end.history.append(("user", user_input))
    expected_state_dict_at_end.correction_attempts = 3 # Summarize increments one last time
    expected_state_dict_at_end.correction_feedback = "Final check failed feedback." # Mock feedback from final check
    expected_state_dict_at_end.current_question = None 
    expected_state_dict_at_end.summary = "Final flawed summary attempt 3" # Mock summary from final attempt
    expected_state_dict_at_end.needs_correction = True # Check fails again
    # Routing logic adds error message
    expected_state_dict_at_end.error_message = f"Summary failed validation after {2+1} attempts."
    expected_state_dict_at_end = expected_state_dict_at_end.model_dump()

    # Mock stream to return the error end state
    with patch("app.api.endpoints.app_graph.astream_events", return_value=mock_astream_events_generator(expected_state_dict_at_end, reached_end=True)) as mock_astream:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200 # API call succeeds
    data = response.json()
    # Response should contain the error message
    assert "Summary failed validation" in data["agent_response"]
    assert data["is_final_turn"]
    # assert data["next_state"] == expected_state_dict_at_end # Compare full state
    returned_state = data["next_state"]
    expected_history_list = [list(item) for item in expected_state_dict_at_end["history"]]
    assert returned_state["history"] == expected_history_list
    assert returned_state["summary"] == expected_state_dict_at_end["summary"]
    assert returned_state["needs_correction"] == expected_state_dict_at_end["needs_correction"]
    assert returned_state["correction_attempts"] == expected_state_dict_at_end["correction_attempts"]
    assert returned_state["error_message"] == expected_state_dict_at_end["error_message"]
    mock_astream.assert_called_once()
