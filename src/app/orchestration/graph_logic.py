# src/app/orchestration/graph_logic.py

import logging
from typing import Literal

from langgraph.graph import END

from ..llm.prompts import (
    get_initiation_prompt,
    get_probe_prompt,
    get_summarize_prompt,
    get_check_summary_prompt,
    get_reflection_depth_prompt,
    get_sentiment_prompt
)
from ..services.llm_client import LLMInterface # Depend on the interface
from .state import AgentState
from .constants import MAX_PROBE_ATTEMPTS, MAX_CORRECTION_ATTEMPTS # Import constants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Node Logic Functions ---

async def run_initiate(state: AgentState) -> AgentState:
    """Generates the initial question based on the topic."""
    if not state.history:
        logger.info("--- Running Initiator Node (First Turn) ---")
        question = get_initiation_prompt(state.topic)
        state.current_question = question
        state.history = [("agent", question)]
        state.error_message = None
        logger.info(f"Initiator generated question: {question}")
    else:
        logger.debug("--- Initiator Node (Skipping - History already exists) ---")
    logger.debug(f"Updated State after initiate: {state}")
    return state

async def run_probe(state: AgentState, llm_client: LLMInterface) -> AgentState:
    """Generates a follow-up question using the LLM."""
    logger.info("--- Running Prober Node ---")
    state.error_message = None
    state.probe_count += 1
    logger.info(f"Probe attempt number: {state.probe_count}")

    if not state.history:
        logger.warning("Probe node called with empty history.")
        state.error_message = "Internal Error: Prober requires history."
        return state

    prompt = get_probe_prompt(state.history)
    logger.info(f"Generating probe question with prompt: {prompt[:100]}...")

    try:
        if not llm_client.api_key:
             raise ValueError("LLM Interface API key is not configured.")
        question = await llm_client.get_completion(prompt)
        if not question:
             logger.warning("LLM returned an empty question.")
             question = "Could you please elaborate on that?"
             state.error_message = "LLM failed to generate a specific question, using fallback."
        state.current_question = question
        state.history.append(("agent", question))
        logger.info(f"Prober generated question: {question}")
    except Exception as e:
        logger.error(f"Error during LLM call in Prober node: {e}", exc_info=True)
        state.error_message = f"Error generating follow-up: {e}"
        state.current_question = "I encountered an issue. Could you perhaps rephrase or tell me more generally?"
        state.history.append(("agent", state.current_question))

    logger.debug(f"Updated State after probe: {state}")
    return state

async def run_summarize(state: AgentState, llm_client: LLMInterface) -> AgentState:
    """Generates a summary of the conversation using the LLM."""
    logger.info("--- Running Summarizer Node ---")
    state.error_message = None
    state.summary = None
    state.current_question = None
    state.correction_attempts += 1
    logger.info(f"Summary generation attempt: {state.correction_attempts}")

    if not state.history:
        logger.warning("Summarizer node called with empty history.")
        state.error_message = "Internal Error: Summarizer requires history."
        state.summary = "(Summary generation skipped: No history)"
        return state

    prompt = get_summarize_prompt(state.history, state.correction_feedback)
    logger.info("Generating summary with prompt...")

    try:
        if not llm_client.api_key:
            raise ValueError("LLM Interface API key is not configured.")
        summary_text = await llm_client.get_completion(prompt)
        if not summary_text:
            logger.warning("LLM returned an empty summary.")
            state.summary = f"(Summary generation attempt {state.correction_attempts} failed - empty response)"
            state.error_message = "LLM failed to generate a summary."
        else:
            state.summary = summary_text
            logger.info(f"Generated summary (Attempt {state.correction_attempts}): {summary_text[:100]}...")
    except Exception as e:
        logger.error(f"Error during LLM call in Summarizer node: {e}", exc_info=True)
        state.error_message = f"Error generating summary: {e}"
        state.summary = f"(Summary generation attempt {state.correction_attempts} encountered an error.)"

    logger.debug(f"Updated State after summarize: {state}")
    return state

async def run_check_summary(state: AgentState, llm_client: LLMInterface) -> AgentState:
    """Checks the generated summary for accuracy using the LLM."""
    logger.info("--- Running Summary Checker Node ---")
    state.error_message = None
    state.needs_correction = True
    state.correction_feedback = None

    if not state.summary or "(Summary generation" in state.summary:
        logger.warning(f"Skipping summary check: Missing or failed summary ('{state.summary}')")
        state.error_message = "Summary check skipped due to missing/failed summary."
        return state
    if not state.history:
        logger.warning("Skipping summary check: Missing history.")
        state.error_message = "Summary check skipped due to missing history."
        return state

    prompt = get_check_summary_prompt(state.history, state.summary)
    logger.info("Checking summary coherence and requesting feedback...")

    try:
        if not llm_client.api_key:
            raise ValueError("LLM Interface API key is not configured.")
        response = await llm_client.get_completion(prompt, model="gpt-4o-mini")
        response_text = response.strip()
        logger.info(f"LLM response for summary check: '{response_text}'")

        if response_text.upper().startswith("YES"):
            state.needs_correction = False
            logger.info("Summary deemed sufficient.")
        else:
            state.needs_correction = True
            feedback = response_text
            if response_text.upper().startswith("NO"):
                parts = response_text.split(maxsplit=1)
                if len(parts) > 1:
                    feedback = parts[1].strip()
            if not feedback or feedback.upper() == "NO":
                feedback = "Summary deemed insufficient, but no specific feedback provided."
            state.correction_feedback = feedback
            logger.warning(f"Summary needs correction. Feedback: '{feedback}'")
    except Exception as e:
        logger.error(f"Error during LLM call in Summary Checker node: {e}", exc_info=True)
        state.error_message = f"Error checking summary quality: {e}"
        state.needs_correction = True
        state.correction_feedback = "Failed to perform summary check due to an error."

    logger.debug(f"Updated State after check_summary: {state}")
    return state

async def run_classify_sentiment(state: AgentState, llm_client: LLMInterface) -> AgentState:
    """Classifies the sentiment of the last user message."""
    logger.info("--- Running Sentiment Classifier Node ---")
    if not state.history or state.history[-1][0] != "user":
        logger.warning("No user message found or last message not from user. Skipping sentiment classification.")
        state.last_sentiment = "neutral"
        return state

    last_user_message = state.history[-1][1]
    prompt = get_sentiment_prompt(last_user_message)

    try:
        if not llm_client.api_key:
            raise ValueError("LLM Interface API key is not configured.")
        response = await llm_client.get_completion(prompt) # Use get_completion
        sentiment = response.strip().lower()
        logger.info(f"Sentiment classified as: {sentiment}")
        if sentiment in ["positive", "negative", "neutral"]:
            state.last_sentiment = sentiment
        else:
            logger.warning(f"Unexpected sentiment format received: {sentiment}. Defaulting to neutral.")
            state.last_sentiment = "neutral"
    except Exception as e:
        logger.error(f"LLM call failed during sentiment classification: {e}", exc_info=True)
        state.last_sentiment = "neutral"

    logger.debug(f"Updated State after classify_sentiment: {state}")
    return state

async def run_present_summary(state: AgentState) -> AgentState:
    """Presents the final summary to the user."""
    logger.info("--- Running Present Summary Node ---")
    if state.summary and not state.error_message:
        state.current_question = None
        logger.info("Summary prepared for presentation.")
    elif state.error_message:
        logger.error(f"Present Summary node encountered error state: {state.error_message}")
    else:
        logger.warning("Present Summary node called without a valid summary.")
        state.error_message = state.error_message or "Internal error: Reached summary presentation without a summary."
    logger.debug(f"Updated State after present_summary: {state}")
    return state

async def run_end_conversation(state: AgentState) -> AgentState:
    """Marks the conversation as ended."""
    logger.info("--- Running End Conversation Node --- ")
    state.current_question = None
    logger.info("Conversation marked as ended.")
    logger.debug(f"Updated State after end_conversation: {state}")
    return state

async def run_wait_for_input(state: AgentState) -> AgentState:
    """Represents a suspension point to wait for user input."""
    logger.info("--- Reached Wait State (Suspending for user input) ---")
    # No state changes needed
    return state


# --- Conditional Routing Logic ---

async def should_continue_probing_route(state: AgentState, llm_client: LLMInterface) -> str:
    """Determines whether to continue probing ('PROBE') or summarize ('SUMMARIZE')."""
    logger.info(f"--- Routing: Checking reflection depth (Probe count: {state.probe_count}) ---")

    if state.probe_count >= MAX_PROBE_ATTEMPTS:
        logger.warning(f"Max probe attempts ({MAX_PROBE_ATTEMPTS}) reached. Forcing summarization.")
        return "SUMMARIZE"

    if not state.history or len(state.history) < 3:
        logger.info("History too short for depth check, continuing probing.")
        return "PROBE"

    prompt = get_reflection_depth_prompt(state.history)

    try:
        logger.info("Asking LLM to assess reflection depth...")
        if not llm_client.api_key:
            raise ValueError("LLM Interface API key missing for depth check.")
        response = await llm_client.get_completion(prompt, model="gpt-3.5-turbo")
        response_text = response.strip().upper()
        logger.info(f"LLM depth assessment response: '{response_text}'")
        if response_text.startswith("YES"):
            logger.info("LLM assessment: Reflection depth sufficient. Routing to SUMMARIZE.")
            return "SUMMARIZE"
        else:
            logger.info("LLM assessment: Reflection depth insufficient. Routing to PROBE.")
            return "PROBE"
    except Exception as e:
        logger.error(f"Error during LLM depth check: {e}. Defaulting to continue probing (PROBE).")
        return "PROBE"


def route_after_summary_check(state: AgentState) -> str:
    """Routes to 'SUMMARIZE' for correction or END after summary check."""
    logger.info(
        f"--- Routing after summary check. Needs Correction: {state.needs_correction}, "
        f"Attempts: {state.correction_attempts}/{MAX_CORRECTION_ATTEMPTS+1} ---"
    )

    if state.error_message and "Summary check skipped" in state.error_message:
        logger.warning("Routing to END due to skipped summary check.")
        return END

    if state.needs_correction and state.correction_attempts <= MAX_CORRECTION_ATTEMPTS:
        logger.info(f"Routing back to Summarizer for correction attempt {state.correction_attempts + 1}.")
        # Note: The graph definition maps this string to the 'SUMMARIZE' node
        return "SUMMARIZE" 
    else:
        if state.needs_correction and state.correction_attempts > MAX_CORRECTION_ATTEMPTS:
            logger.warning(f"Max correction attempts ({MAX_CORRECTION_ATTEMPTS+1}) reached. Proceeding to END with potentially flawed summary.")
            state.error_message = state.error_message or f"Summary failed validation after {MAX_CORRECTION_ATTEMPTS+1} attempts."
        else: # needs_correction is False
            logger.info("Summary approved. Routing to END.")
            state.correction_attempts = 0 # Reset attempts counter on success
        return END


def handle_summary_feedback_route(state: AgentState) -> Literal["continue", "end"]:
    """Routes to 'classify_sentiment' (continue) or 'END_CONVERSATION' (end) based on user feedback."""
    logger.info("--- Routing: Handling Summary Feedback ---")
    if not state.history or state.history[-1][0] != "user":
        logger.warning("Expected user feedback after summary, but last turn was not user. Defaulting to end.")
        return "end"

    last_utterance = state.history[-1][1]
    feedback_lower = last_utterance.lower()

    # Simple keyword matching
    accept_keywords = ["looks good", "that's correct", "yes", "perfect", "sounds right", "agree"]
    continue_keywords = ["more", "actually", "wait", "add", "change", "forgot"]

    if any(phrase in feedback_lower for phrase in accept_keywords):
        logger.info("User accepted the summary. Routing to END_CONVERSATION.")
        return "end"
    elif any(phrase in feedback_lower for phrase in continue_keywords):
        logger.info("User wants to continue/modify after summary. Routing back to classify_sentiment.")
        # The graph definition will map 'continue' to 'classify_sentiment'
        return "continue"
    else:
        logger.info("User feedback unclear after summary. Assuming end.")
        return "end" 