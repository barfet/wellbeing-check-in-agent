from fastapi import APIRouter, HTTPException, Body, status
import logging
from typing import Dict, Any, Union

# Ensure relative imports work correctly
from .models import ReflectionTurnRequest, ReflectionTurnResponse
from ..orchestration.state import AgentState
from ..orchestration.graph import app_graph # Import the compiled graph
# Import initiate function to manually call it for the first turn
from ..orchestration.graph import initiate as initiate_node 
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
    description="Handles both the initiation and subsequent turns using a single graph invocation per turn." # Update description
)
async def process_turn(payload: ReflectionTurnRequest = Body(...)):
    """Processes a turn. Manually handles initiation, invokes graph for subsequent turns."""
    try:
        current_state_provided = payload.current_state is not None

        if current_state_provided:
            # --- Subsequent Turn Processing ---
            logger.info(f"Processing subsequent turn. User input: '{payload.user_input[:50]}...'")
            try:
                # Validate and load the provided state
                state = AgentState(**payload.current_state)
            except ValidationError as e:
                logger.error(f"Invalid current_state received: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid current_state provided: {e}"
                )
            
            if payload.user_input:
                # Add the new user input to the history
                state.history.append(("user", payload.user_input))
                # Clear the previous agent question as it's now been answered
                state.current_question = None 
            else:
                # Handle cases where state is provided but no new input (e.g., client error?)
                logger.warning("Subsequent turn request received with state but no user_input.")
                # Let's proceed with the existing state, graph might handle it or error.

            input_data = state
            
            # Invoke the graph - it runs until it suspends (e.g., needs input) or finishes
            logger.info("Invoking graph for subsequent turn...")
            config = {"recursion_limit": 10} 
            final_state_dict = await app_graph.ainvoke(input_data, config=config) 
            logger.info("Graph invocation complete.")

        else:
            # --- First Turn (Initiation) --- 
            logger.info(f"Processing initiation turn. Topic: '{payload.topic}'")
            if payload.user_input:
                logger.warning("Initiation turn received unexpected user_input, ignoring it.")
            
            # Manually create the initial state by calling the initiate node logic
            initial_graph_input = AgentState(topic=payload.topic)
            try:
                # Run the initiate node function directly
                final_state_obj = await initiate_node(initial_graph_input)
                # Convert the resulting AgentState object to a dictionary
                final_state_dict = final_state_obj.model_dump()
                logger.info(f"Initiation complete. Generated question: {final_state_obj.current_question}")
            except Exception as e:
                 logger.exception(f"Error during manual initiation: {e}")
                 raise HTTPException(status_code=500, detail=f"Agent error during initiation: {e}")

            # For the first turn, the graph hasn't fully run, so it cannot be final yet
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
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent error: Orchestrator returned invalid state structure: {e}"
                )
        
        # Determine agent response and final turn status based on the final state from the graph
        agent_output: str
        is_final: bool

        # Check conditions based on the state *after* the graph run
        has_summary = final_state_obj.summary is not None and "(Summary generation" not in final_state_obj.summary
        has_question = final_state_obj.current_question is not None
        has_error = final_state_obj.error_message is not None
        summary_failed = final_state_obj.summary is not None and "(Summary generation" in final_state_obj.summary

        # **Prioritize responding with a question if one was generated**
        if has_question:
             # Case 1 (Priority): Intermediate turn - graph generated a question.
             # Even if a summary was also generated in this run, we respond with the question first.
             agent_output = final_state_obj.current_question
             is_final = False
             logger.info("Intermediate turn, agent asked a question.")
        elif has_summary: 
            # Case 2: Successful end - valid summary generated and no question asked.
            agent_output = final_state_obj.summary
            is_final = True
            logger.info("Successful completion with summary.")
        elif summary_failed:
             # Case 3: Summary generation failed explicitly, and no question asked.
             agent_output = final_state_obj.summary # Report the failure message
             if final_state_obj.error_message:
                 if final_state_obj.error_message not in agent_output:
                    agent_output += f" (Error details: {final_state_obj.error_message})"
             is_final = True # Treat summary failure as final for this interaction cycle
             logger.warning(f"Summary generation failed: {agent_output}")
        elif has_error:
             # Case 4: Some other error occurred, and no question was asked.
             agent_output = f"An error occurred: {final_state_obj.error_message}"
             is_final = True # Treat other errors as final
             logger.error(f"Agent processing ended with error: {agent_output}")
        else:
            # Case 5: Unexpected final state (no summary, no question, no error)
            logger.error(f"Agent invocation reached unexpected final state: {final_state_obj}")
            agent_output = "An unexpected error occurred. Please try again."
            is_final = True # Treat unexpected states as final
            final_state_dict["error_message"] = final_state_dict.get("error_message", "Agent reached unexpected final state.")

        logger.info(f"Final turn status: Is final={is_final}. Agent response snippet: '{agent_output[:100]}...'")

        # Return the agent's response and the complete final state dictionary
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