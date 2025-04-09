from langgraph.graph import StateGraph, END
from functools import partial # Import partial

from .state import AgentState
from ..services.llm_client import LLMClient
from ..dependencies import get_llm_client # Import the dependency function
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remove the global LLM Client instantiation
# llm_client = LLMClient()

# Define node functions - add llm_client parameter where needed
async def initiate(state: AgentState) -> AgentState:
    # Only generate initial question if history is empty
    if not state.history:
        logger.info("--- Running Initiator Node (First Turn) ---")
        topic = state.topic
        if topic:
            question = f"Okay, let's reflect on '{topic}'. To start, could you tell me briefly what happened regarding this?"
        else:
            question = "Hello! What topic or experience would you like to reflect on today?"

        state.current_question = question
        state.history = [("agent", question)]
        state.error_message = None # Clear previous errors
        logger.info(f"Initiator generated question: {question}")
        logger.debug(f"Updated State after initiate: {state}")
    else:
        # If history exists, this node is being re-entered after the actual start,
        # likely due to how astream_events restarts from the entry point.
        # Pass the state through unchanged.
        logger.debug("--- Initiator Node (Skipping - History already exists) ---")
        pass # State remains unchanged

    return state


async def probe(state: AgentState, llm_client: LLMClient) -> AgentState:
    logger.info("--- Running Prober Node ---")
    state.error_message = None # Clear previous errors
    state.probe_count += 1 # Increment probe count
    logger.info(f"Probe attempt number: {state.probe_count}")

    if not state.history:
        logger.warning("Probe node called with empty history.")
        state.error_message = "Internal Error: Prober requires history."
        # Decide how to handle this - end the graph? Ask initial question again?
        # For now, returning state with error.
        return state # Or raise an exception?

    # Construct prompt using history
    # Simple approach: Use the last user utterance
    last_speaker, last_utterance = state.history[-1]

    if last_speaker != "user":
        logger.warning(f"Probe node expected last speaker to be 'user', but got '{last_speaker}'. History: {state.history}")
        # This might happen if the flow is misconfigured or called incorrectly.
        # Handle gracefully: maybe use the last agent question? Or default prompt?
        # For now, let's try a generic probe if history looks odd.
        prompt = "Ask a generic open-ended question to encourage further reflection."
    else:
        # Format history for the prompt (optional, could just send last utterance)
        formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
        prompt = f"Based on the following conversation history:\n{formatted_history}\n\nAsk the user *one* relevant, open-ended follow-up question to encourage deeper reflection on their last statement ('{last_utterance}'). Focus on exploring feelings, challenges, learnings, or specifics. Avoid simple yes/no questions."

    logger.info(f"Generating probe question with prompt: {prompt[:100]}..." ) # Log snippet

    try:
        # Ensure LLM Client is available (using the injected client)
        if not llm_client.api_key:
             raise ValueError("OpenAI API key is not configured for LLMClient.")

        question = await llm_client.get_completion(prompt)

        if not question:
             logger.warning("LLM returned an empty question.")
             # Fallback question
             question = "Could you please elaborate on that?"
             state.error_message = "LLM failed to generate a specific question, using fallback."

        state.current_question = question
        state.history.append(("agent", question))
        logger.info(f"Prober generated question: {question}")

    except Exception as e:
        logger.error(f"Error during LLM call in Prober node: {e}", exc_info=True)
        state.error_message = f"Error generating follow-up: {e}"
        # Fallback question in case of error
        state.current_question = "I encountered an issue. Could you perhaps rephrase or tell me more generally?"
        state.history.append(("agent", state.current_question))

    logger.debug(f"Updated State after probe: {state}")
    return state


async def summarize(state: AgentState, llm_client: LLMClient) -> AgentState:
    logger.info("--- Running Summarizer Node ---")
    state.error_message = None # Clear previous errors
    state.summary = None # Clear previous summary
    state.current_question = None # Clear question when starting summary path
    
    # Increment attempt count *before* generating summary
    # Note: This means the first attempt is attempt 1
    state.correction_attempts += 1
    logger.info(f"Summary generation attempt: {state.correction_attempts}")

    if not state.history:
        logger.warning("Summarizer node called with empty history.")
        state.error_message = "Internal Error: Summarizer requires history."
        state.summary = "(Summary generation skipped: No history)"
        return state

    # Base prompt
    formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
    base_prompt = (
        f"Based on the following conversation history between an agent and a user:\n\n{formatted_history}\n\n" 
        f"Please provide a concise summary of the key points discussed, focusing on the user's reflections, challenges mentioned, and any potential learnings or insights revealed. "
        f"Structure it as a short paragraph or a few bullet points."
    )

    # Add feedback if this is a correction attempt
    prompt = base_prompt
    if state.correction_feedback:
        logger.info(f"Incorporating previous correction feedback: {state.correction_feedback}")
        prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK: {state.correction_feedback}\nPlease generate a *revised* summary addressing this feedback."
        # Clear feedback after using it
        # state.correction_feedback = None # Let check_summary clear it on next run

    logger.info("Generating summary with prompt...")

    try:
        if not llm_client.api_key:
            raise ValueError("OpenAI API key is not configured for LLMClient.")

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


async def check_summary(state: AgentState, llm_client: LLMClient) -> AgentState:
    logger.info("--- Running Summary Checker Node ---")
    # Reset fields for this check
    state.error_message = None 
    state.needs_correction = True # Default to needing correction
    state.correction_feedback = None # Clear previous feedback

    if not state.summary or "(Summary generation" in state.summary:
        logger.warning(f"Skipping summary check: Missing or failed summary ('{state.summary}')")
        state.error_message = "Summary check skipped due to missing/failed summary."
        return state

    if not state.history:
        logger.warning("Skipping summary check: Missing history.")
        state.error_message = "Summary check skipped due to missing history."
        return state

    # Provide context and ask for specific feedback
    formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
    prompt = (
        f"Review the following conversation history:\n\n{formatted_history}\n\n" 
        f"Now review this generated summary:\n\nSUMMARY:\n{state.summary}\n\n" 
        f"Critique this summary based on the history. Is it accurate, relevant, and does it capture the key points, feelings, and challenges discussed? "
        f"If the summary is good and requires no changes, respond with only YES. "
        f"If the summary is lacking or inaccurate, respond with NO, followed by a brief explanation of what specific information is missing or needs correction based *only* on the conversation history."
    )

    logger.info("Checking summary coherence and requesting feedback...")

    try:
        if not llm_client.api_key:
            raise ValueError("OpenAI API key is not configured for LLMClient.")

        response = await llm_client.get_completion(prompt, model="gpt-4o-mini") # Use a capable model for critique
        response_text = response.strip()

        logger.info(f"LLM response for summary check: '{response_text}'")

        if response_text.upper().startswith("YES"):
            state.needs_correction = False
            logger.info("Summary deemed sufficient.")
        else:
            state.needs_correction = True 
            # Attempt to extract feedback after potential "NO"
            feedback = response_text
            if response_text.upper().startswith("NO"):
                # Try to remove the leading "NO" and surrounding punctuation/whitespace
                parts = response_text.split(maxsplit=1)
                if len(parts) > 1:
                    feedback = parts[1].strip(".,:;\n ")
            
            if not feedback or feedback.upper() == "NO":
                feedback = "Summary deemed insufficient, but no specific feedback provided." 
            
            state.correction_feedback = feedback
            logger.warning(f"Summary needs correction. Feedback: '{feedback}'")
            # Optionally set error message, but feedback field is primary
            # state.error_message = f"Summary needs correction. Feedback: {feedback}"

    except Exception as e:
        logger.error(f"Error during LLM call in Summary Checker node: {e}", exc_info=True)
        state.error_message = f"Error checking summary quality: {e}"
        state.needs_correction = True # Assume correction needed if check fails
        state.correction_feedback = "Failed to perform summary check due to an error."

    logger.debug(f"Updated State after check_summary: {state}")
    return state


async def wait_for_input(state: AgentState) -> AgentState:
    """Dummy node representing a suspension point to wait for user input."""
    logger.info("--- Reached Wait State (Suspending for user input) ---")
    # No state changes, graph execution stops here for this invocation.
    return state


# --- Define Conditional Logic --- 

# Define max probes constant (fallback mechanism)
MAX_PROBE_ATTEMPTS = 5 # Increase fallback slightly

async def should_continue_probing(state: AgentState, llm_client: LLMClient) -> str:
    """Determines whether to continue probing (wait) or proceed to summarization, using LLM assessment."""
    logger.info(f"--- Checking reflection depth (Probe count: {state.probe_count}) ---")

    # Safety check: Max probes override
    if state.probe_count >= MAX_PROBE_ATTEMPTS:
        logger.warning(f"Max probe attempts ({MAX_PROBE_ATTEMPTS}) reached. Forcing summarization.")
        return "summarize"

    # LLM-based check for reflection depth
    if not state.history or len(state.history) < 3: # Need at least init Q + user A + probe Q
        logger.info("History too short for depth check, continuing probing.")
        return "wait_for_input"
    
    formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
    prompt = (
        f"Review the following conversation history between an agent and a user reflecting on a topic:\n\n{formatted_history}\n\n" 
        f"Based *only* on this history, has the user explored their experience, challenges, feelings, or learnings "
        f"in sufficient detail to allow for a meaningful summary? Consider if the core aspects seem covered." 
        f" Answer only with YES or NO."
    )

    try:
        logger.info("Asking LLM to assess reflection depth...")
        if not llm_client.api_key:
            raise ValueError("LLMClient API key missing for depth check.")
        
        # Use a relatively fast/cheap model for this check
        response = await llm_client.get_completion(prompt, model="gpt-3.5-turbo") 
        response_text = response.strip().upper()
        logger.info(f"LLM depth assessment response: '{response_text}'")

        if response_text.startswith("YES"):
            logger.info("LLM assessment: Reflection depth sufficient. Routing to Summarize.")
            return "summarize"
        else:
            # Assume NO or unclear response means more probing needed
            logger.info("LLM assessment: Reflection depth insufficient. Routing to wait_for_input.")
            return "wait_for_input"
            
    except Exception as e:
        logger.error(f"Error during LLM depth check: {e}. Defaulting to continue probing.")
        return "wait_for_input" # Default to continue probing if check fails

# Define max correction attempts constant
MAX_CORRECTION_ATTEMPTS = 2 # Allows initial attempt + 2 retries = 3 total

def route_after_summary_check(state: AgentState) -> str:
    """Determines the next step after the summary check.

    Routes back to summarize if correction is needed and attempts are not exhausted,
    otherwise proceeds to END.
    """
    logger.info(
        f"--- Routing after summary check. Needs Correction: {state.needs_correction}, "
        f"Attempts: {state.correction_attempts}/{MAX_CORRECTION_ATTEMPTS+1} ---"
    )
    
    # End immediately if check was skipped
    if state.error_message and "Summary check skipped" in state.error_message:
        logger.warning("Routing to END due to skipped summary check.")
        return END 
        
    # Check if correction is needed and if we have attempts left
    if state.needs_correction and state.correction_attempts <= MAX_CORRECTION_ATTEMPTS:
        logger.info(f"Routing back to Summarizer for correction attempt {state.correction_attempts + 1}.")
        return "summarize"
    else:
        if state.needs_correction and state.correction_attempts > MAX_CORRECTION_ATTEMPTS:
            logger.warning(f"Max correction attempts ({MAX_CORRECTION_ATTEMPTS+1}) reached. Proceeding to END with potentially flawed summary.")
            # Optionally add a persistent error/warning to the state
            state.error_message = state.error_message or f"Summary failed validation after {MAX_CORRECTION_ATTEMPTS+1} attempts."
        else: # needs_correction is False
            logger.info("Summary approved. Routing to END.")
            # Reset attempts counter on success (optional, good practice)
            state.correction_attempts = 0
        return END


# --- Define the Graph --- 

# Get the LLM client instance using the dependency function
llm_dependency = get_llm_client()

# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes, using partial to inject the llm_client dependency
workflow.add_node("initiate", initiate) 
workflow.add_node("probe", partial(probe, llm_client=llm_dependency))
workflow.add_node("summarize", partial(summarize, llm_client=llm_dependency))
workflow.add_node("check_summary", partial(check_summary, llm_client=llm_dependency))
workflow.add_node("wait_for_input", wait_for_input) # Add the new wait node

# Set the entry point
workflow.set_entry_point("initiate")

# Define edges
workflow.add_edge("initiate", "probe") 

# Add conditional edge after probe
workflow.add_conditional_edges(
    "probe",
    # Inject llm_client dependency into the condition function
    partial(should_continue_probing, llm_client=llm_dependency), 
    {
        # If probing should continue, go to the wait state
        "wait_for_input": "wait_for_input", 
        # If max probes reached, proceed to summarize node.
        "summarize": "summarize" 
    }
)
# Note: No edge out of wait_for_input - it's a terminal node for the invocation

workflow.add_edge("summarize", "check_summary")

# Add the conditional edge based on the summary check (existing)
workflow.add_conditional_edges(
    "check_summary",
    route_after_summary_check,
    {
        "summarize": "summarize", 
        END: END
    }
)

# Compile the graph
app_graph = workflow.compile()

# Example invocation (for testing/debugging locally)
# Needs to be async now
async def run_graph_example():
    initial_state = AgentState(topic="Team Presentation")
    print(f"Initial State: {initial_state}")

    # The API layer would handle getting user input and invoking the graph step-by-step.
    # For a full invoke test, we need to manually insert simulated user turns
    # between the steps where the graph expects them (e.g., before 'probe').

    # Run initiate
    state_after_initiate = await app_graph.ainvoke(initial_state, config={"run_name": "initiate_run"})
    print("\n--- State after Initiate ---")
    print(state_after_initiate)

    if state_after_initiate and not state_after_initiate.get('error_message'):
        # Simulate user response before probe
        user_input_1 = "The presentation went okay, but I felt nervous."
        state_before_probe = AgentState(**state_after_initiate)
        state_before_probe.history.append(("user", user_input_1))

        # Run probe (and subsequent steps based on edges)
        # Note: .ainvoke runs from the *start* unless specifying checkpoints.
        # To run just the next step requires managing checkpoints or using stream/update.
        # For a simple test, let's invoke fully again, but the state includes the user input now.
        # This isn't ideal graph usage but shows the node logic.
        # A better test would use `astream` or manage state externally.
        print(f"\n--- Invoking graph again with user input for Probe: {user_input_1} ---")
        final_state = await app_graph.ainvoke(state_before_probe, config={"run_name": "full_run_after_input"})
        print("\n--- Final State ---")
        print(final_state)
    else:
         print("\n--- Skipping further execution due to error in initiate ---")

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set in environment or .env file
    # Check key via the dependency getter
    if get_llm_client().api_key:
        print("Running graph example...")
        asyncio.run(run_graph_example())
    else:
        print("Skipping graph example: OPENAI_API_KEY not found.") 