from fastapi import APIRouter, HTTPException, Body, status
import logging
from typing import Dict, Any, Union, Optional

# Ensure relative imports work correctly
from .models import ReflectionTurnRequest, ReflectionTurnResponse
from ..orchestration.state import AgentState
from ..orchestration.graph import app_graph # Import the compiled graph
from ..orchestration.graph import initiate as initiate_node # Re-import initiate
from pydantic import ValidationError

router = APIRouter()
logger = logging.getLogger(__name__)

# Remove OUTPUT_NODES if no longer used by stream logic
# OUTPUT_NODES = {"initiate", "probe", "summarize", "check_summary"} 

# Remove the entire get_final_state_from_stream function
# async def get_final_state_from_stream(...) -> Dict[str, Any]: ...

@router.post(
    "/turns",
    response_model=ReflectionTurnResponse,
    summary="Process a single turn in a reflection conversation",
    description="Handles reflection turns using graph invocation for subsequent turns."
)
async def process_turn(payload: ReflectionTurnRequest = Body(...)):
    """Manually handles initiation, invokes graph for subsequent turns."""
    try:
        current_state_provided = payload.current_state is not None
        
        if current_state_provided:
            # --- Subsequent Turn Processing ---
            logger.info(f"Processing subsequent turn. User input: '{payload.user_input[:50]}...'")
            try:
                state = AgentState(**payload.current_state)
            except ValidationError as e:
                logger.error(f"Invalid current_state received: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid current_state provided: {e}")
            
            if payload.user_input:
                state.history.append(("user", payload.user_input))
                # Don't clear question here
            else:
                logger.warning("Subsequent turn request received with state but no user_input.")
            
            input_data = state
            
            # Invoke the graph - it runs until a terminal node (wait_for_input or END)
            logger.info("Invoking graph for subsequent turn...")
            config = {"recursion_limit": 15} 
            final_state_dict = await app_graph.ainvoke(input_data, config=config) 
            logger.info(f"Graph invocation complete. Final state dict: {final_state_dict}")

        else:
            # --- First Turn (Initiation - Manual) --- 
            logger.info(f"Processing initiation turn. Topic: '{payload.topic}'")
            if payload.user_input:
                logger.warning("Initiation turn received unexpected user_input, ignoring it.")
            
            # Manually call initiate node logic
            initial_graph_input = AgentState(topic=payload.topic, probe_count=0) # Ensure probe_count starts at 0
            try:
                final_state_obj = await initiate_node(initial_graph_input)
                final_state_dict = final_state_obj.model_dump()
                logger.info(f"Manual initiation complete. Generated question: {final_state_obj.current_question}")
            except Exception as e:
                 logger.exception(f"Error during manual initiation: {e}")
                 raise HTTPException(status_code=500, detail=f"Agent error during initiation: {e}")

            # For the first turn, it cannot be final yet
            is_final = False 
            agent_output = final_state_obj.current_question
            
            # Return the response immediately after initiation
            return ReflectionTurnResponse(
                agent_response=agent_output,
                next_state=final_state_dict, 
                is_final_turn=is_final
            )

        # --- Post-Graph Processing (Only for subsequent turns) --- 
        if not final_state_dict:
            logger.error("Graph invocation returned an empty state dictionary.")
            raise HTTPException(status_code=500, detail="Agent error: Received empty state from orchestrator.")

        # Validate the final state structure
        try:
             final_state_obj = AgentState(**final_state_dict)
        except ValidationError as e:
                logger.error(f"Graph invocation resulted in invalid state dictionary: {final_state_dict}. Error: {e}")
                raise HTTPException(status_code=500, detail=f"Agent error: Orchestrator returned invalid state structure: {e}")

        # --- Determine agent response and final turn status --- 
        agent_output: str
        is_final: bool

        has_question = final_state_obj.current_question is not None
        has_error = final_state_obj.error_message is not None # Check error first
        has_summary = final_state_obj.summary is not None and "(Summary generation" not in final_state_obj.summary
        summary_failed = final_state_obj.summary is not None and "(Summary generation" in final_state_obj.summary

        # Logic: Prioritize Question > Error > Summary > Summary Failure
        if has_question:
            agent_output = final_state_obj.current_question
            is_final = False 
            logger.info("Intermediate turn, agent asked a question.")
        elif has_error: # Handle errors before summary
            agent_output = f"An error occurred: {final_state_obj.error_message}"
            is_final = True # Treat errors as final if no question generated
            logger.error(f"Agent processing ended with error: {agent_output}")
        elif has_summary:
            agent_output = final_state_obj.summary
            is_final = True 
            logger.info("Successful completion with summary.")
        elif summary_failed:
            agent_output = final_state_obj.summary
            if final_state_obj.error_message and final_state_obj.error_message not in agent_output:
                agent_output += f" (Error details: {final_state_obj.error_message})"
            is_final = True 
            logger.warning(f"Summary generation failed: {agent_output}")
        # Removed the redundant error check here
        else:
            logger.error(f"Agent invocation reached unexpected final state (no Q/Summary/Error): {final_state_obj}")
            agent_output = "An unexpected error occurred. Please try again."
            is_final = True 
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
        logger.exception(f"Unhandled error processing reflection turn: {e}") 
        raise HTTPException(status_code=500, detail=f"Internal server error processing reflection: {e}") 