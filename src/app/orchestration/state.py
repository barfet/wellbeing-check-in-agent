from typing import List, Tuple, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Represents the state of the reflection agent's conversation.

    Attributes:
        topic: The optional topic provided by the user for the reflection.
        history: A list of tuples representing the conversation history.
                 Each tuple contains (speaker, utterance), e.g., [("user", "..."), ("agent", "...")].
        current_question: The most recent question asked by the agent.
        summary: The generated summary of the reflection.
        needs_correction: Flag indicating if the summary needs correction.
        error_message: Any error message encountered during the process.
        probe_count: The count of probes made to the agent.
        # Fields for advanced summary correction
        correction_attempts: int = 0
        correction_feedback: Optional[str] = None
        # Field for sentiment analysis
        last_sentiment: Optional[str] = None
    """

    topic: Optional[str] = None
    history: List[Tuple[str, str]] = Field(default_factory=list)
    current_question: Optional[str] = None
    summary: Optional[str] = None
    needs_correction: bool = False
    error_message: Optional[str] = None
    probe_count: int = 0
    # Fields for advanced summary correction
    correction_attempts: int = 0
    correction_feedback: Optional[str] = None
    # Field for sentiment analysis
    last_sentiment: Optional[str] = None 