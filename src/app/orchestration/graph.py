from langgraph.graph import StateGraph, END

from .state import AgentState


# Define dummy node functions
def initiate(state: AgentState):
    print("--- Running Initiator Node ---")
    # Placeholder logic: Will be replaced with actual implementation
    state.current_question = "Placeholder: Initial question?"
    state.history.append(("agent", state.current_question))
    print(f"Updated State: {state}")
    return state


def probe(state: AgentState):
    print("--- Running Prober Node ---")
    # Placeholder logic: Needs user input handling (from API layer)
    # Assuming user input was added to history before calling this node
    state.current_question = "Placeholder: Follow-up question?"
    state.history.append(("agent", state.current_question))
    print(f"Updated State: {state}")
    return state


def summarize(state: AgentState):
    print("--- Running Summarizer Node ---")
    # Placeholder logic
    state.summary = "Placeholder: This is a summary."
    print(f"Updated State: {state}")
    return state


def check_summary(state: AgentState):
    print("--- Running Summary Checker Node ---")
    # Placeholder logic: Always assumes summary is good for now
    state.needs_correction = False
    print(f"Updated State: {state}")
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
if __name__ == "__main__":
    initial_state = AgentState(topic="Test Topic")
    print(f"Initial State: {initial_state}")

    # Simulate a user response before the probe node
    # This would typically happen in the API layer based on user input
    # initial_state.history.append(("user", "My initial thoughts are..."))

    final_state = app_graph.invoke(initial_state)
    print("\n--- Final State ---")
    print(final_state) 