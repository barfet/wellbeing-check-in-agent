from typing import Optional, List, Tuple

# --- Sentiment Classification ---

def get_sentiment_prompt(user_message: str) -> str:
    """Creates a prompt to classify the sentiment of a user message.

    Args:
        user_message: The user message content.

    Returns:
        A prompt string for sentiment classification.
    """
    # Ensure user_message is escaped properly if needed, though f-string usually handles it.
    return f"""
Classify the sentiment of the following user message. Respond with only one word: 'positive', 'negative', or 'neutral'.

User Message: "{user_message}"

Sentiment:"""


# --- Reflection Initiation ---

def get_initiation_prompt(topic: Optional[str]) -> str:
    """Creates the initial prompt to start the reflection.

    Args:
        topic: The optional topic provided by the user.

    Returns:
        The initial agent question.
    """
    if topic:
        # Use f-string for safe embedding of the topic
        return f"Okay, let's reflect on '{topic}'. To start, could you tell me briefly what happened regarding this?"
    else:
        return "Hello! What topic or experience would you like to reflect on today?"


# --- Reflection Probing ---

def get_probe_prompt(history: List[Tuple[str, str]]) -> str:
    """Creates a prompt to generate a follow-up question based on history.

    Args:
        history: The conversation history list of (speaker, utterance) tuples.

    Returns:
        A prompt string for generating a probe question.
    """
    if not history:
        # Should ideally not happen if called correctly, but handle defensively.
        return "Ask a generic open-ended question to start the reflection."

    last_speaker, last_utterance = history[-1]

    if last_speaker != "user":
        # If the last speaker wasn't the user, ask a generic question.
        # This might indicate a flow issue, but we need a prompt regardless.
        return "Ask a generic open-ended question to encourage further reflection."
    else:
        # Format history for the prompt
        formatted_history = "\\n".join([f"{spk}: {utt}" for spk, utt in history])
        # Ensure the last utterance is included clearly for the LLM focus
        return f"""Based on the following conversation history:
{formatted_history}

Ask the user *one* relevant, open-ended follow-up question to encourage deeper reflection on their last statement ('{last_utterance}'). Focus on exploring feelings, challenges, learnings, or specifics. Avoid simple yes/no questions."""


# --- Reflection Summarization ---

def get_summarize_prompt(history: List[Tuple[str, str]], correction_feedback: Optional[str] = None) -> str:
    """Creates a prompt to summarize the conversation history.

    Args:
        history: The conversation history list of (speaker, utterance) tuples.
        correction_feedback: Optional feedback from a previous failed summary check.

    Returns:
        A prompt string for generating a summary.
    """
    if not history:
        return "Cannot generate summary: No conversation history provided." # Or handle differently

    formatted_history = "\\n".join([f"{spk}: {utt}" for spk, utt in history])
    base_prompt = f"""Based on the following conversation history between an agent and a user:

{formatted_history}

Please provide a concise summary of the key points discussed, focusing on the user's reflections, challenges mentioned, and any potential learnings or insights revealed. Structure it as a short paragraph or a few bullet points."""

    # Add feedback if this is a correction attempt
    if correction_feedback:
        # Ensure feedback is clearly delineated
        return f"{base_prompt}\\n\\nPREVIOUS ATTEMPT FEEDBACK: {correction_feedback}\\nPlease generate a *revised* summary addressing this feedback."
    else:
        return base_prompt


# --- Summary Checking ---

def get_check_summary_prompt(history: List[Tuple[str, str]], summary: str) -> str:
    """Creates a prompt to check the quality and accuracy of a generated summary.

    Args:
        history: The conversation history list of (speaker, utterance) tuples.
        summary: The generated summary to be checked.

    Returns:
        A prompt string for checking the summary.
    """
    if not history or not summary:
        return "Cannot check summary: Missing history or summary." # Or handle differently

    formatted_history = "\\n".join([f"{spk}: {utt}" for spk, utt in history])
    # Ensure summary is clearly marked
    return f"""Review the following conversation history:

{formatted_history}

Now review this generated summary:

SUMMARY:
{summary}

Critique this summary based on the history. Is it accurate, relevant, and does it capture the key points, feelings, and challenges discussed?
If the summary is good and requires no changes, respond with only YES.
If the summary is lacking or inaccurate, respond with NO, followed by a brief explanation of what specific information is missing or needs correction based *only* on the conversation history."""


# --- Reflection Depth Check (for Probing vs Summarizing) ---

def get_reflection_depth_prompt(history: List[Tuple[str, str]]) -> str:
    """Creates a prompt to assess if the conversation depth warrants summarization.

    Args:
        history: The conversation history list of (speaker, utterance) tuples.

    Returns:
        A prompt string for assessing reflection depth.
    """
    if not history or len(history) < 3: # Check if history is sufficient for context
         # This case should be handled by the caller, but return a non-committal prompt if forced.
        return "Cannot assess depth: Insufficient history provided. Respond with NO."

    formatted_history = "\\n".join([f"{spk}: {utt}" for spk, utt in history])
    return f"""Review the following conversation history between an agent and a user reflecting on a topic:

{formatted_history}

Based *only* on this history, has the user explored their experience, challenges, feelings, or learnings in sufficient detail to allow for a meaningful summary? Consider if the core aspects seem covered. Answer only with YES or NO.""" 