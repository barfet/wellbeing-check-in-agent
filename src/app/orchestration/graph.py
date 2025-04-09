from langgraph.graph import StateGraph, END

from .state import AgentState
from ..services.llm_client import LLMClient
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate LLM Client (consider dependency injection for more complex scenarios)
# For simplicity in MVP, we can instantiate it here or within the node.
# Instantiating outside might be slightly more efficient if reused across nodes.
llm_client = LLMClient()

# Define node functions
async def initiate(state: AgentState) -> AgentState:
    logger.info("--- Running Initiator Node ---")
    topic = state.topic
    if topic:
        question = f"Okay, let's reflect on '{topic}'. To start, could you tell me briefly what happened regarding this?"
    else:
        question = "Hello! What topic or experience would you like to reflect on today?"

    state.current_question = question
    # Clear history in case this is a re-invocation or loop
    state.history = [("agent", question)]
    state.error_message = None # Clear previous errors
    logger.info(f"Initiator generated question: {question}")
    logger.debug(f"Updated State after initiate: {state}")
    return state


async def probe(state: AgentState) -> AgentState:
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
        # Ensure LLM Client is available
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


async def summarize(state: AgentState) -> AgentState:
    logger.info("--- Running Summarizer Node ---")
    state.error_message = None # Clear previous errors
    # Placeholder logic - Actual LLM call in Task 4.1
    summary = f"Placeholder summary based on history: {len(state.history)} turns."
    state.summary = summary
    logger.info(f"Generated placeholder summary: {summary}")
    logger.debug(f"Updated State after summarize: {state}")
    return state


async def check_summary(state: AgentState) -> AgentState:
    logger.info("--- Running Summary Checker Node ---")
    state.error_message = None # Clear previous errors
    # Placeholder logic - Actual LLM call in Task 4.2
    state.needs_correction = False # Assume good for now
    logger.info(f"Summary check result: needs_correction={state.needs_correction}")
    logger.debug(f"Updated State after check_summary: {state}")
    return state


# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("initiate", initiate)
workflow.add_node("probe", probe)
workflow.add_node("summarize", summarize)
workflow.add_node("check_summary", check_summary)

# Set the entry point
workflow.set_entry_point("initiate")

# Define edges for the basic flow (conditional logic later)
workflow.add_edge("initiate", "probe")
workflow.add_edge("probe", "summarize")
workflow.add_edge("summarize", "check_summary")

# For now, always end after checking the summary
# Conditional logic will be added in Epic 4
workflow.add_edge("check_summary", END)

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
    if llm_client.api_key:
        print("Running graph example...")
        asyncio.run(run_graph_example())
    else:
        print("Skipping graph example: OPENAI_API_KEY not found.") 