from fastapi import APIRouter, HTTPException, Body, status
import logging
from typing import Dict, Any, Union, Optional, AsyncIterator, Tuple

# Ensure relative imports work correctly
from .models import ReflectionTurnRequest, ReflectionTurnResponse
from ..orchestration.state import AgentState
# Import from the new definition file
from ..orchestration.graph_definition import app_graph 
# Import logic function for manual initiation - NO LONGER NEEDED
# from ..orchestration.graph_logic import run_initiate as initiate_node 
from pydantic import ValidationError
from langgraph.errors import GraphRecursionError

router = APIRouter()
logger = logging.getLogger(__name__)

# Remove OUTPUT_NODES if no longer used by stream logic
# OUTPUT_NODES = {"initiate", "probe", "summarize", "check_summary"} 

# Remove the entire get_final_state_from_stream function
# async def get_final_state_from_stream(...) -> Dict[str, Any]: ...

async def _get_final_state_from_stream(stream: AsyncIterator[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Consumes the LangGraph event stream and extracts the final state dictionary
    and the name of the node associated with that state.

    Handles cases where the stream ends normally (at __end__), is interrupted 
    (at wait_for_input), or terminates unexpectedly.
    """
    node_end_states: Dict[str, Dict[str, Any]] = {}
    last_node_name_processed: Optional[str] = None
    final_node_name: Optional[str] = None
    last_received_state: Optional[Dict[str, Any]] = None # Track the last state seen

    async for event in stream:
        kind = event["event"]
        tags = event.get("tags", [])
        name = event.get("name")
        data = event.get("data", {})

        # Capture the latest state snapshot from any event containing output data
        if isinstance(data.get("output"), dict):
            last_received_state = data["output"]
            logger.trace(f"Captured state snapshot from event: {kind} / {name}")
        
        if kind == "on_node_start":
            last_node_name_processed = name
            logger.debug(f"Node started: {name}")
        elif kind == "on_node_end":
            output_data = data.get("output")
            if isinstance(output_data, dict):
                node_end_states[name] = output_data
                logger.debug(f"Node ended: {name}. Captured state keys: {output_data.keys()}")
            else:
                 logger.warning(f"Node ended: {name}, but output was not a dict: {type(output_data)}")
        elif kind == "on_chain_end": # Might be relevant for final state in some graphs
             output_data = data.get("output")
             if isinstance(output_data, dict):
                last_received_state = output_data # Update with chain end state if available
                logger.debug(f"Chain ended. Final output keys: {output_data.keys()}")

    logger.debug(f"Stream finished. Last node processed: {last_node_name_processed}. Captured node end states: {list(node_end_states.keys())}")
    
    # Determine final state based on how the stream ended
    if last_node_name_processed == "__end__":
        # Find state from the node that led to END
        # Note: LangGraph's final __end__ event might not contain the state itself.
        # We rely on the state captured from the *preceding* node's end event.
        if "CHECK_SUMMARY" in node_end_states: 
            final_state_snapshot = node_end_states["CHECK_SUMMARY"]
            final_node_name = "CHECK_SUMMARY"
        # Add other nodes leading to END if necessary
        elif node_end_states: # Fallback: take the latest captured state before end
            last_captured_node = list(node_end_states.keys())[-1]
            final_state_snapshot = node_end_states[last_captured_node]
            final_node_name = last_captured_node
            logger.warning("Reached END but couldn't definitively determine preceding node state, using last captured before END.")
        else: 
             final_node_name = "__end__"
             final_state_snapshot = last_received_state # Use last known state if no node state captured before END
             logger.warning("Reached END but no prior node end state was captured. Using last overall state received.")


    elif last_node_name_processed == "wait_for_input":
        # Stream likely interrupted by wait_for_input config. Find state from the node *before* wait_for_input
        if "INITIATE" in node_end_states: # After first turn
            final_state_snapshot = node_end_states["INITIATE"]
            final_node_name = "INITIATE"
        elif "PROBE" in node_end_states: # After a probe
            final_state_snapshot = node_end_states["PROBE"]
            final_node_name = "PROBE"
        # Add other nodes leading to wait_for_input if necessary
        else: # Fallback if the preceding node wasn't captured correctly
            logger.warning(f"Interrupted at wait_for_input, but couldn't find state from preceding node (INITIATE/PROBE). Using last overall snapshot.")
            if node_end_states:
                 last_captured_node = list(node_end_states.keys())[-1]
                 final_state_snapshot = node_end_states[last_captured_node]
                 final_node_name = last_captured_node # May not be correct node for interrupt state
            else:
                 final_node_name = "wait_for_input" # Indicate interrupt happened
                 final_state_snapshot = last_received_state # Use last known state
                 logger.warning("Interrupted at wait_for_input, but no node end state was captured. Using last overall state received.")
                 
    else:
         # Stream ended unexpectedly or after a node not explicitly handled above
         logger.warning(f"Stream ended after unexpected node: {last_node_name_processed}. Using last captured state if available.")
         if node_end_states: 
             # Prioritize state from the last node that ended, if any
             last_captured_node = list(node_end_states.keys())[-1]
             final_state_snapshot = node_end_states[last_captured_node]
             final_node_name = last_captured_node
         else:
             # Fallback: Use the very last state received from any event
             final_node_name = last_node_name_processed or "Unknown" # Provide a name if possible
             final_state_snapshot = last_received_state 
             logger.warning(f"No node end states captured for unexpected end ({final_node_name}). Using last overall state received.")

    if not final_state_snapshot and last_received_state:
         # Final fallback if main logic failed but we saw *some* state during the stream
         logger.warning(f"Main logic failed to find final state after node {final_node_name}. Using last overall state received as fallback.")
         final_state_snapshot = last_received_state
         # Keep final_node_name as determined by the logic above

    logger.debug(f"_get_final_state_from_stream returning state from node '{final_node_name}', keys: {final_state_snapshot.keys() if final_state_snapshot else 'None'}")
    return final_state_snapshot, final_node_name

@router.post(
    "/turns",
    response_model=ReflectionTurnResponse,
    summary="Process a single turn in a reflection conversation",
    description="Handles reflection turns using graph streaming."
)
async def process_turn(payload: ReflectionTurnRequest = Body(...)):
    """Handles reflection turns by streaming the compiled LangGraph app."""
    logger.debug(f"Received turn request payload: {payload}")
    try:
        input_state_dict: dict
        # --- Prepare Input State ---
        if payload.current_state:
            logger.info(f"Processing subsequent turn. User input: '{payload.user_input[:50]}...'")
            try:
                # 1. Validate incoming state first
                state_dict = payload.current_state
                # Basic validation, AgentState init happens implicitly in graph
                if not isinstance(state_dict.get("history"), list):
                    raise ValueError("Invalid history format in current_state")
                
                # 2. Append user input if provided
                if payload.user_input:
                    state_dict["history"].append(("user", payload.user_input))
                else:
                     logger.warning("Subsequent turn request received with state but no user_input.")
                
                input_state_dict = state_dict

            except (ValidationError, ValueError) as e:
                logger.error(f"Invalid current_state received: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Invalid current_state provided: {e}")
        else:
            # First turn: Initialize state dict with topic
            logger.info(f"Processing initiation turn. Topic: '{payload.topic}'")
            if payload.user_input:
                logger.warning("Initiation turn received unexpected user_input, ignoring it.")
            input_state_dict = AgentState(topic=payload.topic).model_dump()

        # --- Stream the graph --- 
        logger.info(f"Streaming graph with input state dict keys: {input_state_dict.keys()}")
        config = {"recursion_limit": 15}
        stream = app_graph.astream_events(input_state_dict, config=config, version="v1")
        
        # Get the final state AND the node name associated with it
        final_state_dict, last_node_name = await _get_final_state_from_stream(stream)
        logger.info(f"Graph streaming complete. Last node associated with final state: {last_node_name}")
        logger.debug(f"Final state dict from stream: {final_state_dict}")

        # --- Process Graph Output --- 
        if not final_state_dict:
            logger.error("Graph streaming did not yield a final state dictionary.")
            raise HTTPException(status_code=500, detail="Agent error: Orchestrator stream ended unexpectedly.")
        
        # Validate the final state structure
        try:
             final_state_obj = AgentState(**final_state_dict)
        except ValidationError as e:
            logger.error(f"Graph streaming resulted in invalid final state: {final_state_dict}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent error: Orchestrator stream produced invalid state: {e}")
        
        # --- Determine agent response and final turn status --- 
        agent_output: str
        is_final: bool
        
        # Determine if the graph ended or was interrupted
        # Graph is considered ended if the last node processed was END_CONVERSATION or __end__ tag was seen
        # It's interrupted if the last node processed was likely INITIATE or PROBE (leading to wait_for_input)
        # Note: last_node_name reflects the node whose event held the final state, which might be the node *before* the interrupt.
        graph_likely_ended = last_node_name in ["END_CONVERSATION", "__end__", "CHECK_SUMMARY"] # Check summary can lead directly to END
        # If the last state came from INITIATE or PROBE, it means we were interrupted before wait_for_input
        was_interrupted_before_wait = last_node_name in ["INITIATE", "PROBE"]

        has_question = final_state_obj.current_question is not None
        has_error = final_state_obj.error_message is not None
        has_summary = final_state_obj.summary is not None and "(Summary generation" not in final_state_obj.summary

        if was_interrupted_before_wait and has_question:
             # Interrupted after INITIATE or PROBE, waiting for user input
            agent_output = final_state_obj.current_question
            is_final = False 
            logger.info(f"Intermediate turn (interrupted after {last_node_name}), agent asked a question.")
        else:
            # Graph likely reached END or was interrupted at a different point.
            # Determine final output: Error > Summary > Fallback
            if has_error:
                agent_output = f"An error occurred: {final_state_obj.error_message}"
                is_final = True
                logger.error(f"Agent processing ended with error (last node: {last_node_name}): {agent_output}")
            elif has_summary:
                agent_output = final_state_obj.summary
                is_final = True 
                logger.info(f"Successful completion (last node: {last_node_name}) with summary.")
            # Check if it was interrupted unexpectedly (e.g., after classify_sentiment but before probe/summarize?)
            elif not graph_likely_ended and not has_question:
                logger.error(f"Agent interrupted unexpectedly after {last_node_name} without a question: {final_state_obj}")
                agent_output = "An unexpected state was reached. Please try again."
                is_final = True
                final_state_dict["error_message"] = final_state_dict.get("error_message", f"Agent interrupted unexpectedly after {last_node_name}.")
            else: # Reached END or another terminal state without error or summary
                logger.error(f"Agent invocation ended unexpectedly (last node: {last_node_name}, no Q/Summary/Error): {final_state_obj}")
                agent_output = "An unexpected error occurred. Please try again."
                is_final = True 
                final_state_dict["error_message"] = final_state_dict.get("error_message", f"Agent reached unexpected end state after {last_node_name}.")

        logger.info(f"Final turn status: Is final={is_final}. Agent response snippet: '{agent_output[:100]}...'")
        logger.debug(f"Final state being returned: {final_state_dict}")

        return ReflectionTurnResponse(
            agent_response=agent_output,
            next_state=final_state_dict, 
            is_final_turn=is_final
        )

    except HTTPException: # Re-raise HTTP exceptions
        raise
    except GraphRecursionError as e:
         logger.error(f"Graph recursion limit reached: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Agent error: Conversation complexity limit reached. {e}")
    except Exception as e:
        logger.exception(f"Unhandled error processing reflection turn: {e}") 
        raise HTTPException(status_code=500, detail=f"Internal server error processing reflection: {e}") 