import pytest
from pydantic import ValidationError

from app.orchestration.state import AgentState


def test_agent_state_defaults():
    """Test that AgentState initializes with correct default values."""
    state = AgentState()
    assert state.topic is None
    assert state.history == []
    assert state.current_question is None
    assert state.summary is None
    assert state.needs_correction is False
    assert state.error_message is None


def test_agent_state_initialization():
    """Test initializing AgentState with specific values."""
    topic = "Test Reflection"
    history = [("user", "Said something"), ("agent", "Asked something")]
    question = "What next?"
    summary = "A summary."
    needs_correction = True
    error = "An error occurred"

    state = AgentState(
        topic=topic,
        history=history,
        current_question=question,
        summary=summary,
        needs_correction=needs_correction,
        error_message=error,
    )

    assert state.topic == topic
    assert state.history == history
    assert state.current_question == question
    assert state.summary == summary
    assert state.needs_correction == needs_correction
    assert state.error_message == error


def test_agent_state_type_validation():
    """Test Pydantic validation for field types."""
    # Test invalid history type
    with pytest.raises(ValidationError):
        AgentState(history=[("user", 123)])  # Utterance should be str

    # Test invalid needs_correction type
    with pytest.raises(ValidationError):
        AgentState(needs_correction=123)  # Should be bool, not int

    # Test correct types work
    state = AgentState(
        topic="Valid Topic",
        history=[("user", "Valid input"), ("agent", "Valid question")],
        needs_correction=False,
    )
    assert state.topic == "Valid Topic"
    assert len(state.history) == 2
    assert state.needs_correction is False 