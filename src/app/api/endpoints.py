from fastapi import APIRouter, HTTPException, Body, status
import logging
from typing import Dict, Any

# Ensure relative imports work correctly
from .models import ReflectionTurnRequest, ReflectionTurnResponse
from ..orchestration.state import AgentState
from ..orchestration.graph import app_graph # Import the compiled graph
from pydantic import ValidationError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/turns",
    response_model=ReflectionTurnResponse,
    summary="Process a single turn in a reflection conversation",
    description="Handles both the initiation and subsequent turns of a reflection conversation based on the provided state."
)
async def process_turn(payload: ReflectionTurnRequest = Body(...)):
    """Processes a turn in the reflection conversation.\n\n    - If `current_state` is null, it\'s an initiation turn using `topic`.\n    - If `current_state` is provided, it\'s a subsequent turn using `user_input`.\n\n    Invokes the underlying LangGraph agent orchestrator.\n    """
    try:
        input_data: Dict[str, Any] | AgentState

        if payload.current_state:
            # Subsequent turn
            logger.info(f"Processing subsequent turn. User input: '{payload.user_input[:50]}...'")
            try:
                # Convert incoming dict state to AgentState object
                state = AgentState(**payload.current_state)
            except ValidationError as e:
                logger.error(f"Invalid current_state received: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid current_state provided: {e}"
                )

            # Add user input to history if provided
            if payload.user_input:
                state.history.append(("user", payload.user_input))
            else:
                # Handle cases where subsequent turn might not have user input
                logger.warning("Subsequent turn received without user_input.")
                # Potentially raise an error or handle based on application logic
                # For now, allow it to proceed, graph nodes should handle state.

            # Pass the state object to ainvoke
            input_data = state

        else:
            # Initiation turn
            logger.info(f"Processing initiation turn. Topic: '{payload.topic}'")
            if payload.user_input:
                logger.warning("user_input provided on initiation turn, it will be ignored.")
            # Pass the initial topic dict to ainvoke
            input_data = {"topic": payload.topic}

        # Invoke the graph
        config = {"recursion_limit": 10} # Set recursion limit for safety
        final_state_dict: Dict[str, Any] = await app_graph.ainvoke(input_data, config=config)

        if not final_state_dict:
            logger.error("Graph invocation returned empty state.")
            raise HTTPException(status_code=500, detail="Agent error: Received empty state from orchestrator.")

        # Convert final state dict back to AgentState object for easier access & validation
        try:
             final_state_obj = AgentState(**final_state_dict)
        except ValidationError as e:
                logger.error(f"Graph returned invalid state dictionary: {final_state_dict}. Error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent error: Orchestrator returned invalid state structure: {e}"
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
             logger.error(f"Agent ended unexpectedly with summary needing correction: {final_state_obj}")
             agent_output = "An internal error occurred during summary validation."
             is_final = True
             final_state_dict["error_message"] = final_state_dict.get("error_message", "Agent ended unexpectedly with summary needing correction.")
        else:
            # Case 6: Truly unexpected final state (no summary, no question, no error)
            logger.error(f"Agent reached unexpected final state: {final_state_obj}")
            agent_output = "An unexpected error occurred. Please try again."
            is_final = True # Treat unexpected states as final
            final_state_dict["error_message"] = final_state_dict.get("error_message", "Agent reached unexpected final state.")

        logger.info(f"Final turn status: Is final={is_final}. Agent response snippet: '{agent_output[:100]}...'")

        return ReflectionTurnResponse(
            agent_response=agent_output,
            next_state=final_state_dict, # Pass the raw dict back
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