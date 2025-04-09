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

    if not state.history:
        logger.warning("Summarizer node called with empty history.")
        state.error_message = "Internal Error: Summarizer requires history."
        return state

    # Format history for the prompt
    formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
    prompt = f"Based on the following conversation history between an agent and a user:\n\n{formatted_history}\n\nPlease provide a concise summary of the key points discussed, focusing on the user's reflections, challenges mentioned, and any potential learnings or insights revealed. Structure it as a short paragraph or a few bullet points."

    logger.info("Generating summary with prompt...")

    try:
        # Ensure LLM Client is available (using the injected client)
        if not llm_client.api_key:
            raise ValueError("OpenAI API key is not configured for LLMClient.")

        summary_text = await llm_client.get_completion(prompt)

        if not summary_text:
            logger.warning("LLM returned an empty summary.")
            state.summary = "(Summary generation failed - empty response)"
            state.error_message = "LLM failed to generate a summary."
        else:
            state.summary = summary_text
            logger.info(f"Generated summary: {summary_text[:100]}...") # Log snippet

    except Exception as e:
        logger.error(f"Error during LLM call in Summarizer node: {e}", exc_info=True)
        state.error_message = f"Error generating summary: {e}"
        state.summary = "(Summary generation encountered an error.)"

    logger.debug(f"Updated State after summarize: {state}")
    return state


async def check_summary(state: AgentState, llm_client: LLMClient) -> AgentState:
    logger.info("--- Running Summary Checker Node ---")
    state.error_message = None # Clear previous errors
    state.needs_correction = True # Default to needing correction unless validation passes

    if not state.summary or "(Summary generation" in state.summary:
        logger.warning(f"Skipping summary check because summary is missing or indicates generation failure: '{state.summary}'")
        state.error_message = "Summary check skipped due to missing or failed summary."
        # Keep needs_correction=True to potentially loop back or signal issue
        return state

    if not state.history:
        logger.warning("Skipping summary check because history is missing.")
        state.error_message = "Summary check skipped due to missing history."
        return state

    # Provide context for the check
    formatted_history = "\n".join([f"{spk}: {utt}" for spk, utt in state.history])
    prompt = f"Consider the following conversation history:\n\n{formatted_history}\n\nNow consider this summary generated from the conversation:\n\nSUMMARY:\n{state.summary}\n\nIs this summary relevant and coherent based *only* on the provided conversation history? Does it accurately reflect the main points discussed? Answer with only YES or NO."

    logger.info("Checking summary coherence with LLM...")

    try:
        # Ensure LLM Client is available (using the injected client)
        if not llm_client.api_key:
            raise ValueError("OpenAI API key is not configured for LLMClient.")

        response = await llm_client.get_completion(prompt, model="gpt-3.5-turbo") # Use a reliable model
        response_text = response.strip().upper()

        logger.info(f"LLM response for summary check: '{response_text}'")

        # Use stricter check
        if response_text.startswith("YES"):
            state.needs_correction = False
            logger.info("Summary deemed coherent.")
        else:
            state.needs_correction = True # Keep as True if not explicitly YES
            logger.warning(f"Summary deemed potentially incoherent or LLM response unclear ('{response_text}'). Setting needs_correction=True.")
            state.error_message = f"Summary quality check indicated potential issues (LLM response: '{response_text}')."

    except Exception as e:
        logger.error(f"Error during LLM call in Summary Checker node: {e}", exc_info=True)
        state.error_message = f"Error checking summary quality: {e}"
        state.needs_correction = True # Assume correction needed if check fails

    logger.debug(f"Updated State after check_summary: {state}")
    return state


# --- Define Conditional Logic --- 
def route_after_summary_check(state: AgentState) -> str:
    """Determines the next step after the summary check.

    Args:
        state: The current agent state.

    Returns:
        The name of the next node ('summarize' or '__end__').
    """
    logger.info(f"--- Routing based on summary check (needs_correction={state.needs_correction}) ---")
    if state.error_message and "Summary check skipped" in state.error_message:
        logger.warning("Routing to END due to skipped summary check.")
        return END # End if check was skipped due to missing data
        
    if state.needs_correction:
        logger.info("Routing back to Summarizer.")
        # Optionally clear the bad summary before retrying, although summarize node already does this
        # state.summary = None 
        return "summarize"
    else:
        logger.info("Routing to END.")
        return END


# --- Define the Graph --- 

# Get the LLM client instance using the dependency function
llm_dependency = get_llm_client()

# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes, using partial to inject the llm_client dependency
workflow.add_node("initiate", initiate) # Initiate doesn't need the client
workflow.add_node("probe", partial(probe, llm_client=llm_dependency))
workflow.add_node("summarize", partial(summarize, llm_client=llm_dependency))
workflow.add_node("check_summary", partial(check_summary, llm_client=llm_dependency))

# Set the entry point
workflow.set_entry_point("initiate")

# Define edges for the basic flow (conditional logic later)
workflow.add_edge("initiate", "probe")
workflow.add_edge("probe", "summarize")
workflow.add_edge("summarize", "check_summary")

# Remove the direct edge to END from check_summary
# workflow.add_edge("check_summary", END)

# Add the conditional edge based on the summary check
workflow.add_conditional_edges(
    "check_summary",
    route_after_summary_check,
    {
        "summarize": "summarize", # Map return value to node name
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