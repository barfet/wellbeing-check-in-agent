import pytest
import os
from dotenv import load_dotenv

# Ensure the src directory is in the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.orchestration.state import AgentState
# Import the node functions directly, not the compiled graph
from app.orchestration.graph import initiate, probe
from app.services.llm_client import LLMClient

# Load environment variables (especially OPENAI_API_KEY for the test)
load_dotenv()

# Check if OPENAI_API_KEY is available
openai_api_key = os.getenv("OPENAI_API_KEY")
requires_openai_key = pytest.mark.skipif(
    not openai_api_key,
    reason="Requires OPENAI_API_KEY environment variable to be set"
)

@requires_openai_key
@pytest.mark.asyncio
async def test_initiate_and_probe_integration():
    """Test the initiate and probe node functions directly, including a real LLM call in probe."""
    initial_topic = "My recent project presentation"
    initial_state = AgentState(topic=initial_topic)
    
    # Instantiate a real LLM client for the integration test
    llm_client = LLMClient()
    assert llm_client.api_key, "Test requires OPENAI_API_KEY to be set"

    # --- Initiate Step (direct call) ---
    print(f"\n[Integration Test] Running initiate for topic: {initial_topic}")
    state_after_initiate = await initiate(initial_state)
    print(f"[Integration Test] State after initiate: {state_after_initiate}")

    # Basic assertions for initiate output
    assert isinstance(state_after_initiate, AgentState)
    assert state_after_initiate.current_question is not None
    assert initial_topic in state_after_initiate.current_question
    assert state_after_initiate.error_message is None
    assert isinstance(state_after_initiate.history, list)
    assert len(state_after_initiate.history) == 1
    assert state_after_initiate.history[0][0] == "agent"
    initiate_question = state_after_initiate.history[0][1]
    assert initiate_question == state_after_initiate.current_question

    # --- Simulate User Input ---
    user_response = "It went reasonably well, but I felt quite nervous beforehand and during the Q&A."
    # Modify the state returned by initiate
    state_before_probe = state_after_initiate
    state_before_probe.history.append(("user", user_response))

    print(f"\n[Integration Test] Running probe after user input: '{user_response}'")

    # --- Probe Step (direct call) ---
    # Pass the instantiated LLM client to the probe function
    state_after_probe = await probe(state_before_probe, llm_client=llm_client)
    print(f"[Integration Test] State after probe: {state_after_probe}")

    # --- Assertions for Probe Output ---
    assert isinstance(state_after_probe, AgentState)
    # LLM response varies, so check for presence and type
    assert state_after_probe.current_question is not None
    assert isinstance(state_after_probe.current_question, str)
    # Ensure the question is different from the initiate question
    assert state_after_probe.current_question != initiate_question 
    # Check history has been updated with user input and new agent question
    assert len(state_after_probe.history) == 3
    assert state_after_probe.history[1] == ("user", user_response)
    assert state_after_probe.history[2][0] == "agent"
    assert state_after_probe.history[2][1] == state_after_probe.current_question
    # Check error message is None (assuming successful LLM call)
    assert state_after_probe.error_message is None

# Optional: Add more tests for different scenarios, error handling through the graph, etc.
