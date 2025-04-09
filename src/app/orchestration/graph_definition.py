# src/app/orchestration/graph_definition.py

import logging
from functools import partial

from langgraph.graph import StateGraph, END

from ..dependencies import get_llm_client # To get the LLMInterface instance
from .state import AgentState
from .constants import MAX_PROBE_ATTEMPTS, MAX_CORRECTION_ATTEMPTS # Import constants

# Import the node logic and conditional routing functions
from .graph_logic import (
    run_initiate,
    run_probe,
    run_summarize,
    run_check_summary,
    run_present_summary,
    run_end_conversation,
    run_classify_sentiment,
    run_wait_for_input,
    # Routing functions
    should_continue_probing_route,
    route_after_summary_check,
    handle_summary_feedback_route
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define the Graph --- 

# Get the LLM client instance (conforming to LLMInterface)
llm_dependency = get_llm_client()

# Define the graph structure using AgentState
workflow = StateGraph(AgentState)

# --- Add Nodes --- 
# Nodes that don't need the LLM client directly
workflow.add_node("INITIATE", run_initiate)
workflow.add_node("PRESENT_SUMMARY", run_present_summary) 
workflow.add_node("END_CONVERSATION", run_end_conversation)
workflow.add_node("wait_for_input", run_wait_for_input)

# Nodes that require the LLM client dependency
workflow.add_node("PROBE", partial(run_probe, llm_client=llm_dependency))
workflow.add_node("SUMMARIZE", partial(run_summarize, llm_client=llm_dependency))
workflow.add_node("CHECK_SUMMARY", partial(run_check_summary, llm_client=llm_dependency))
workflow.add_node("classify_sentiment", partial(run_classify_sentiment, llm_client=llm_dependency))

# --- Define Edges --- 

# Set the entry point
workflow.set_entry_point("INITIATE")

# Define standard transitions
workflow.add_edge("INITIATE", "wait_for_input")          # -> Wait for first user input
workflow.add_edge("wait_for_input", "classify_sentiment") # -> Classify response 
workflow.add_edge("PROBE", "wait_for_input")              # -> Wait after probe question
workflow.add_edge("SUMMARIZE", "CHECK_SUMMARY")         # -> Check summary after generation

# Define conditional transitions

# After classifying sentiment, decide whether to probe again or summarize
workflow.add_conditional_edges(
    "classify_sentiment",
    # Inject llm_client dependency into the routing function
    partial(should_continue_probing_route, llm_client=llm_dependency),
    {
        "PROBE": "PROBE",         # Continue probing
        "SUMMARIZE": "SUMMARIZE",   # Proceed to summarization
    },
)

# After checking the summary, decide whether to re-summarize or end
workflow.add_conditional_edges(
    "CHECK_SUMMARY",
    route_after_summary_check, # This routing logic doesn't need LLM client directly
    {
        "SUMMARIZE": "SUMMARIZE", # Route back to SUMMARIZE for correction
        END: END                # Route to END if summary is ok or max attempts reached
    }
)

# After presenting the summary, decide whether to continue or end based on feedback
workflow.add_conditional_edges(
    "PRESENT_SUMMARY",
    handle_summary_feedback_route, # This routing logic doesn't need LLM client directly
    {
        "continue": "classify_sentiment", # Loop back to classify new input
        "end": "END_CONVERSATION"       # Proceed to final end node
    },
)

# --- Compile the Graph --- 
# Add interrupt_before to pause execution before the wait node runs
app_graph = workflow.compile(interrupt_before=["wait_for_input"])

logger.info("Reflection agent graph compiled successfully with interrupt.")

# Remove the example invocation block - keep this file focused on definition 