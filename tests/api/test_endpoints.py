import pytest
import pytest_asyncio # Import the asyncio fixture decorator
from httpx import AsyncClient, ASGITransport # Import ASGITransport
from unittest.mock import patch, AsyncMock
import os

# Ensure the src directory is in the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import the FastAPI app instance
# Assumes pytest can find and use this for the test client
from app.main import app
from app.orchestration.state import AgentState

# Base URL for the test client
BASE_URL = "http://test"
API_ENDPOINT = "/api/v1/reflections/turns"

# --- Test Fixtures --- 

@pytest_asyncio.fixture(scope="function") # Changed decorator
async def async_client() -> AsyncClient:
    """Provides an asynchronous test client for the FastAPI app."""
    # Use ASGITransport to route requests directly to the app
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
        yield client

@pytest.fixture
def initial_state_dict() -> dict:
    # A plausible state dictionary after the initiate node has run
    return {
        "topic": "Test Topic",
        "history": [("agent", "Okay, let's reflect on 'Test Topic'. To start...")],
        "current_question": "Okay, let's reflect on 'Test Topic'. To start...",
        "summary": None,
        "needs_correction": False,
        "error_message": None
    }

# --- Test Cases --- 

@pytest.mark.asyncio
async def test_initiation_turn(async_client: AsyncClient):
    """Test the first turn (initiation) of the conversation."""
    payload = {"topic": "Starting a reflection"}

    # Mock the response from the graph orchestrator
    mock_response_state = AgentState(
        topic="Starting a reflection",
        history=[("agent", "Okay, let's reflect on 'Starting a reflection'. What happened?")],
        current_question="Okay, let's reflect on 'Starting a reflection'. What happened?",
    ).model_dump() # Use model_dump for pydantic v2

    with patch("app.api.endpoints.app_graph.ainvoke", new_callable=AsyncMock, return_value=mock_response_state) as mock_ainvoke:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == "Okay, let's reflect on 'Starting a reflection'. What happened?"
    assert not data["is_final_turn"]
    assert data["next_state"]["topic"] == "Starting a reflection"
    assert len(data["next_state"]["history"]) == 1
    mock_ainvoke.assert_called_once()
    call_args, call_kwargs = mock_ainvoke.call_args
    assert call_args[0] == {"topic": "Starting a reflection"} # Check input to ainvoke

@pytest.mark.asyncio
async def test_subsequent_turn(async_client: AsyncClient, initial_state_dict: dict):
    """Test a subsequent turn providing user input and current state."""
    user_input = "I had a difficult conversation with a colleague."
    payload = {
        "user_input": user_input,
        "current_state": initial_state_dict
    }

    # Mock the response from the graph orchestrator after processing user input
    mock_response_state = AgentState(
        topic="Test Topic",
        history=initial_state_dict['history'] + [("user", user_input), ("agent", "Tell me more about that conversation.")],
        current_question="Tell me more about that conversation.",
    ).model_dump()

    with patch("app.api.endpoints.app_graph.ainvoke", new_callable=AsyncMock, return_value=mock_response_state) as mock_ainvoke:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == "Tell me more about that conversation."
    assert not data["is_final_turn"]
    assert data["next_state"]["topic"] == "Test Topic"
    assert len(data["next_state"]["history"]) == 3
    assert data["next_state"]["history"][1] == ["user", user_input]
    mock_ainvoke.assert_called_once()
    call_args, call_kwargs = mock_ainvoke.call_args
    # Check that the input to ainvoke was an AgentState object with updated history
    assert isinstance(call_args[0], AgentState)
    assert call_args[0].history[-1] == ("user", user_input)

@pytest.mark.asyncio
async def test_final_turn_summary(async_client: AsyncClient, initial_state_dict: dict):
    """Test a turn that results in a final summary."""
    user_input = "That's all for now."
    # Simulate state just before the summary node would run
    state_before_summary = initial_state_dict.copy()
    state_before_summary['history'].append(("user", user_input))
    payload = {
        "user_input": user_input,
        "current_state": state_before_summary
    }

    # Mock the final state returned by the graph (summary generated, validated)
    final_summary = "Summary of the reflection on Test Topic."
    mock_final_state = AgentState(
        topic="Test Topic",
        history=state_before_summary['history'] + [("agent", "Generating summary...")], # Simulate agent turn before summary
        summary=final_summary,
        needs_correction=False,
        current_question=None # No question in final state
    ).model_dump()

    with patch("app.api.endpoints.app_graph.ainvoke", new_callable=AsyncMock, return_value=mock_final_state) as mock_ainvoke:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_response"] == final_summary
    assert data["is_final_turn"]
    assert data["next_state"]["summary"] == final_summary
    assert data["next_state"]["needs_correction"] is False
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_invalid_current_state(async_client: AsyncClient):
    """Test sending an invalid structure for current_state."""
    payload = {
        "user_input": "Some input",
        # Send None for history, which is invalid for AgentState
        "current_state": {"topic": "test", "history": None} 
    }

    # No need to mock ainvoke as it shouldn't be reached
    response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 400 # Bad Request
    data = response.json()
    assert "Invalid current_state provided" in data["detail"]
    # Check for the specific validation error message related to the invalid type
    assert "Input should be a valid list" in data["detail"]

@pytest.mark.asyncio
async def test_graph_invocation_error(async_client: AsyncClient):
    """Test handling when the graph invocation itself raises an error."""
    payload = {"topic": "Error test"}
    error_message = "Graph simulation error"

    with patch("app.api.endpoints.app_graph.ainvoke", new_callable=AsyncMock, side_effect=Exception(error_message)) as mock_ainvoke:
        response = await async_client.post(f"{BASE_URL}{API_ENDPOINT}", json=payload)

    assert response.status_code == 500 # Internal Server Error
    data = response.json()
    assert "Internal server error processing reflection" in data["detail"]
    assert error_message in data["detail"]
    mock_ainvoke.assert_called_once()
