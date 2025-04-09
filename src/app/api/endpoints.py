from fastapi import APIRouter, HTTPException, Body, status
import logging
from typing import Dict, Any, AsyncIterator, Union
import json # Import json for potential state serialization debugging

# Ensure relative imports work correctly
from .models import ReflectionTurnRequest, ReflectionTurnResponse
from ..orchestration.state import AgentState
from ..orchestration.graph import app_graph # Import the compiled graph
from pydantic import ValidationError

router = APIRouter()
logger = logging.getLogger(__name__)

# Nodes that produce output for the user or mark a final state
OUTPUT_NODES = {"initiate", "probe", "summarize", "check_summary"}

async def get_final_state_from_stream(stream: AsyncIterator[Dict[str, Any]], 
                                    input_data: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]: 
    """Helper to consume the stream and get the relevant state dictionary for the *current* turn.
       Breaks after 'probe' to return the question, otherwise captures state after summary nodes.
    """
    probe_state_dict = None # State after probe is hit
    summary_state_dict = None # State after summarize/check_summary (if probe not hit or stream continues)
    
    async for event in stream:
        kind = event.get("event")
        if kind == "on_chain_end": 
            node_name = event.get("name")
            logger.debug(f"Node '{node_name}' ended. Event data keys: {event.get('data', {}).keys()}")
            
            output_data = event.get("data", {}).get("output")

            current_node_state_dict = None
            if isinstance(output_data, AgentState):
                try:
                    current_node_state_dict = output_data.model_dump()
                except Exception as e:
                    logger.error(f"Failed to dump AgentState model from node '{node_name}': {e}")
            elif isinstance(output_data, dict):
                current_node_state_dict = output_data
            # else: Ignore non-state outputs

            if current_node_state_dict is None:
                continue 

            # Capture state specifically after 'probe' and break
            if node_name == "probe":
                probe_state_dict = current_node_state_dict
                logger.debug(f"Captured state dict after node '{node_name}'. Breaking stream consumption.")
                break # Break after probe generates its question
            
            # Capture state after final nodes if probe wasn't hit/broken
            elif node_name in {"summarize", "check_summary"}:
                summary_state_dict = current_node_state_dict
                logger.debug(f"Captured potential final state dict after node '{node_name}'. Will continue stream.")
                # Don't break, let it reach END

    # --- Determine what state to return --- 
    if probe_state_dict is not None:
        # If we broke after probe, return that state
        logger.info("Returning state dict captured after probe.")
        return probe_state_dict
    elif summary_state_dict is not None:
        # If probe didn't run/break, but summary did, return summary state
        logger.info("Returning state dict captured after summarize/check_summary.")
        return summary_state_dict
    else:
        # Fallback if no relevant state was captured
        logger.warning("Stream ended without capturing state dict after probe or summary nodes. Returning original input data as fallback.")
        # ... (fallback logic as before) ...
        if isinstance(input_data, AgentState):
            try:
                return input_data.model_dump()
            except Exception:
                return {}
        elif isinstance(input_data, dict):
            return input_data
        else:
            return {}

@router.post(
    "/turns",
    response_model=ReflectionTurnResponse,
    summary="Process a single turn in a reflection conversation",
    description="Handles both the initiation and subsequent turns... (using streaming)"
)
async def process_turn(payload: ReflectionTurnRequest = Body(...)):
    """Processes a turn using astream_events to capture intermediate state."""
    try:
        input_data: Union[Dict[str, Any], AgentState]
        current_state_provided = payload.current_state is not None

        if current_state_provided:
            logger.info(f"Processing subsequent turn. User input: '{payload.user_input[:50]}...'")
            try:
                state = AgentState(**payload.current_state)
            except ValidationError as e:
                logger.error(f"Invalid current_state received: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid current_state provided: {e}"
                )

            if payload.user_input:
                state.history.append(("user", payload.user_input))
            else:
                logger.warning("Subsequent turn received without user_input.")

            input_data = state

        else:
            logger.info(f"Processing initiation turn. Topic: '{payload.topic}'")
            input_data = {"topic": payload.topic}

        # Stream events from the graph
        config = {"recursion_limit": 10}
        stream = app_graph.astream_events(input_data, config=config, version="v1") # Use v1 events

        # Consume the stream to find the relevant final state for this turn
        # This helper will iterate through and find the state after the last relevant node ran
        final_state_dict = await get_final_state_from_stream(stream, input_data)

        if not final_state_dict:
            logger.error("Graph streaming returned or resulted in an empty state dictionary.")
            raise HTTPException(status_code=500, detail="Agent error: Received empty state from orchestrator stream.")

        try:
             final_state_obj = AgentState(**final_state_dict)
        except ValidationError as e:
                logger.error(f"Graph stream resulted in invalid state dictionary: {final_state_dict}. Error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent error: Orchestrator stream returned invalid state structure: {e}"
                )
        
        # Determine agent response and final turn status
        agent_output: str
        is_final: bool

        # Check conditions in logical order
        has_summary = final_state_obj.summary is not None and "(Summary generation" not in final_state_obj.summary
        needs_correction = final_state_obj.needs_correction
        has_question = final_state_obj.current_question is not None
        has_error = final_state_obj.error_message is not None
        summary_failed = final_state_obj.summary is not None and "(Summary generation" in final_state_obj.summary

        if has_summary and not needs_correction:
            # Case 1: Successful end - valid summary, no correction needed
            agent_output = final_state_obj.summary
            is_final = True
            logger.info("Successful completion with summary.")
        elif has_question:
             # Case 2: Intermediate turn - agent asked a question
             agent_output = final_state_obj.current_question
             is_final = False
             logger.info("Intermediate turn, agent asked a question.")
        elif summary_failed:
             # Case 3: Summary generation failed explicitly
             agent_output = final_state_obj.summary # Report the failure message
             if final_state_obj.error_message:
                 agent_output += f" (Error details: {final_state_obj.error_message})"
             is_final = True # Treat summary failure as final
             logger.warning(f"Summary generation failed: {agent_output}")
        elif has_error:
             # Case 4: Some other error occurred (and no question was asked)
             agent_output = f"An error occurred: {final_state_obj.error_message}"
             is_final = True # Treat other errors as final
             logger.error(f"Agent processing ended with error: {agent_output}")
        elif has_summary and needs_correction:
             # Case 5: Unexpected final state - summary exists but needs correction
             # This shouldn't happen if the graph routes correctly back to summarize.
             logger.error(f"Agent stream ended unexpectedly with summary needing correction: {final_state_obj}")
             agent_output = "An internal error occurred during summary validation."
             is_final = True
             final_state_dict["error_message"] = final_state_dict.get("error_message", "Agent ended unexpectedly with summary needing correction.")
        else:
            # Case 6: Truly unexpected final state (no summary, no question, no error)
            logger.error(f"Agent stream reached unexpected final state: {final_state_obj}")
            agent_output = "An unexpected error occurred. Please try again."
            is_final = True # Treat unexpected states as final
            final_state_dict["error_message"] = final_state_dict.get("error_message", "Agent reached unexpected final state.")

        logger.info(f"Final turn status: Is final={is_final}. Agent response snippet: '{agent_output[:100]}...'")

        return ReflectionTurnResponse(
            agent_response=agent_output,
            next_state=final_state_dict, 
            is_final_turn=is_final
        )

    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unhandled error processing reflection turn: {e}") # Log full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal server error processing reflection: {e}"
        ) 