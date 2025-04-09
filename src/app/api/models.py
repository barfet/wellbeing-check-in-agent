from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Note: We might eventually want a more specific type for the state dictionary,
# potentially mirroring AgentState fields, but using Dict[str, Any] for flexibility initially.
# Alternatively, import AgentState and use it directly if the API layer should
# strictly enforce the internal state structure on the wire.
# Let's use Dict for now as per the implementation doc guidance suggestion.

class ReflectionTurnRequest(BaseModel):
    """Request model for processing a turn in the reflection conversation."""
    topic: Optional[str] = None # Used only for the initiation turn
    user_input: Optional[str] = None # User's response in subsequent turns
    current_state: Optional[Dict[str, Any]] = None # State from the previous turn

class ReflectionTurnResponse(BaseModel):
    """Response model after processing a reflection turn."""
    agent_response: str # The agent's question or final summary
    next_state: Dict[str, Any] # The updated state to be passed in the next request
    is_final_turn: bool # Flag indicating if the conversation has concluded 