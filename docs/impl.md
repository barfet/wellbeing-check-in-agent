**Epic 1: Project Setup & Foundational Backend**

* **Goal:** Establish the project structure, core dependencies, basic runnable application, containerization, and CI foundation.
* **Value:** Provides a stable base for subsequent feature development and ensures code quality from the start.

**Tasks:**

1.  **Task 1.1: Initialize Project Structure & Version Control**
    * **Description:** Set up the Git repository. Initialize the project using Poetry for dependency management. Create the standard Python project layout (`src/`, `tests/`, etc.). Add configuration files (`pyproject.toml`, `.gitignore`, `README.md`).
    * **Acceptance Criteria:**
        * Git repository exists on the hosting platform (e.g., GitHub).
        * `pyproject.toml` is present and configured for the project (name, description, Python version >=3.10).
        * Directory structure (`src/app`, `tests/`) is created.
        * `.gitignore` file suitable for Python projects is present.
        * Poetry environment can be successfully installed (`poetry install`).
    * **Test Cases:** Manual verification of file structure and successful `poetry install`.

2.  **Task 1.2: Implement Basic FastAPI Application Shell**
    * **Description:** Create the main FastAPI application instance (`src/app/main.py`). Add a simple health check endpoint (`/health`) to verify the app is running. Configure basic settings loading (e.g., using Pydantic's `BaseSettings`).
    * **Guidance:**
        ```python
        # src/app/main.py
        from fastapi import FastAPI
        # from .config import settings # Example settings import

        app = FastAPI(title="Reflective Learning Agent API", version="1.0.0")

        @app.get("/health", tags=["Infrastructure"])
        async def health_check():
            return {"status": "OK"}

        # Add other routers later: app.include_router(...)
        ```
    * **Acceptance Criteria:**
        * FastAPI application runs locally using `uvicorn`.
        * GET request to `/health` returns `{"status": "OK"}` with HTTP status 200.
    * **Test Cases:** Unit test for the `/health` endpoint using `pytest` and `httpx.AsyncClient`. Manual test via browser/curl.

3.  **Task 1.3: Dockerize the FastAPI Application**
    * **Description:** Create a `Dockerfile` using multi-stage builds to create an optimized, runnable container image for the application. Include a `.dockerignore` file.
    * **Guidance:** Use a Python base image, install dependencies using Poetry export, copy application code, expose port, set CMD/ENTRYPOINT to run Uvicorn.
    * **Acceptance Criteria:**
        * `Dockerfile` and `.dockerignore` exist in the project root.
        * `docker build .` command successfully builds the image without errors.
        * `docker run <image_name>` starts the container, and the `/health` endpoint is accessible from the host.
    * **Test Cases:** Manual build and run of the Docker image, verifying `/health` endpoint access.

4.  **Task 1.4: Setup Basic CI Pipeline (Lint & Test)**
    * **Description:** Create a GitHub Actions workflow (`.github/workflows/ci.yml`) that triggers on push/pull_request to the main branch. The workflow should checkout code, set up Python and Poetry, install dependencies, run linters (`ruff`), and run tests (`pytest`).
    * **Acceptance Criteria:**
        * `ci.yml` file exists and defines the lint and test jobs.
        * Workflow successfully runs automatically on pull requests/pushes to `main`.
        * Workflow fails if linting errors occur.
        * Workflow fails if any tests fail.
    * **Test Cases:** Create a PR with linting errors -> verify CI fails. Create a PR with failing tests -> verify CI fails. Create a PR with clean code/passing tests -> verify CI passes.

---

**Epic 2: Core Agent Orchestration (LangGraph)**

* **Goal:** Define the state management model and the basic LangGraph structure that will orchestrate the agent interactions.
* **Value:** Establishes the core logic flow and state handling mechanism for the conversational agent, directly addressing TDD specifications.

**Tasks:**

1.  **Task 2.1: Define AgentState Model**
    * **Description:** Create the `AgentState` data structure as a Pydantic model or TypedDict, matching the definition in the TDD (Section 5). This model will be passed between LangGraph nodes.
    * **Guidance:**
        ```python
        # src/app/orchestration/state.py
        from typing import List, Tuple, Optional
        from pydantic import BaseModel, Field

        class AgentState(BaseModel):
            topic: Optional[str] = None
            history: List[Tuple[str, str]] = Field(default_factory=list) # [(speaker, utterance)]
            current_question: Optional[str] = None
            summary: Optional[str] = None
            needs_correction: bool = False
            error_message: Optional[str] = None
            # Potentially add invocation metadata if needed later
        ```
    * **Acceptance Criteria:**
        * `AgentState` model is defined in `src/app/orchestration/state.py`.
        * Model includes all fields specified in TDD Section 5.
        * Model can be imported and instantiated correctly.
    * **Test Cases:** Unit tests validating model creation, default values, and type correctness.

2.  **Task 2.2: Implement Basic LangGraph Structure**
    * **Description:** Create the main graph definition file (`src/app/orchestration/graph.py`). Instantiate `StatefulGraph` using the `AgentState`. Define dummy functions for each agent node (`initiate`, `probe`, `summarize`, `check_summary`) and add them to the graph. Define the basic (non-conditional) edges and entry/exit points.
    * **Guidance:** Use `langgraph.graph.StatefulGraph`, `graph.add_node`, `graph.add_edge`, `graph.set_entry_point`, `graph.set_finish_point`. Dummy nodes should just print their name and pass the state through.
    * **Acceptance Criteria:**
        * `StatefulGraph` instance is created using `AgentState`.
        * Dummy functions representing agent nodes are defined and added to the graph.
        * Basic edges connecting the dummy nodes in the main sequence are defined.
        * Entry and finish points are set.
        * The graph object compiles successfully (`graph.compile()`).
    * **Test Cases:** Unit test verifying graph compilation. Integration test (within code) invoking the compiled graph with an initial state and verifying it passes through the dummy nodes in the expected sequence.

---

**Epic 3: Agent Implementation - Dialogue Flow**

* **Goal:** Implement the core conversational agents (Initiator, Prober) that handle the beginning of the user interaction, including LLM integration.
* **Value:** Brings the initial part of the conversation to life, directly addressing Alex's need for guided questioning (PRD FR2).

**Tasks:**

1.  **Task 3.1: Implement LLM Client Wrapper**
    * **Description:** Create a service/class (`src/app/services/llm_client.py`) to abstract interactions with the chosen LLM provider (e.g., OpenAI). It should handle API key loading from environment variables, prompt formatting (basic initially), API calls, response parsing, and basic error handling (timeouts, retries).
    * **Guidance:** Use `openai` library, `python-dotenv`, implement an async method `get_completion(prompt: str) -> str`. Load API key in constructor/init.
    * **Acceptance Criteria:**
        * `LLMClient` class/module exists.
        * Loads API Key correctly from environment variable (e.g., `OPENAI_API_KEY`).
        * `get_completion` method successfully calls the LLM API (e.g., `openai.ChatCompletion.acreate`) and returns the text response.
        * Basic error handling (e.g., `try...except` for API errors, potential simple retry) is implemented.
    * **Test Cases:** Unit tests mocking the `openai` library calls to test logic without actual API calls. Integration test (requires API key configured) making a real call and verifying a response. Test error handling paths (mock API error response).

2.  **Task 3.2: Implement Initiator Agent Node Logic**
    * **Description:** Replace the dummy `initiate` node function with real logic. It should take the `AgentState`, determine the initial question (using the `topic` if provided, otherwise a generic prompt), potentially call the LLM Client if needed (though likely just uses predefined questions for MVP), and update the `AgentState` with the `current_question` and add it to `history`.
    * **Guidance:** Focus on simple predefined questions for MVP based on `state['topic']`.
    * **Acceptance Criteria:**
        * Initiator function correctly generates an initial question string.
        * Updates `state['current_question']` with the generated question.
        * Adds the agent's question turn to `state['history']`.
        * Returns the updated `AgentState`.
    * **Test Cases:** Unit tests for the agent function with different initial states (topic vs no topic). Integration test verifying this node functions correctly within the compiled LangGraph.

3.  **Task 3.3: Implement Prober Agent Node Logic**
    * **Description:** Replace the dummy `probe` node function. It should take the `AgentState`, formulate a prompt for the LLM using the conversation `history` (especially the last user input), call the `LLMClient` to generate *one* relevant follow-up question (per MVP scope), update `AgentState` with the new `current_question`, and add turns to `history`.
    * **Guidance:** The prompt should instruct the LLM to ask a relevant open-ended question based on the prior turn, focusing on "what went well?", "what was challenging?", "tell me more about X?".
    * **Acceptance Criteria:**
        * Prober function correctly extracts relevant context from `state['history']`.
        * Calls `LLMClient` with an appropriate prompt.
        * Updates `state['current_question']` with the LLM-generated question.
        * Adds the user's last input and the agent's new question to `state['history']`.
        * Returns the updated `AgentState`.
    * **Test Cases:** Unit tests mocking `LLMClient` to verify prompt construction and state updates. Integration test within LangGraph (requires API key). Test with different simulated user inputs in history.

---

**Epic 4: Agent Implementation - Summarization & Correction**

* **Goal:** Implement the agents responsible for summarizing the conversation and performing the basic quality check, completing the core reflection loop.
* **Value:** Delivers the key output for Alex (the summary) and implements the basic self-correction pattern (PRD FR3, FR4).

**Tasks:**

1.  **Task 4.1: Implement Summarizer Agent Node Logic**
    * **Description:** Replace the dummy `summarize` node function. It takes the `AgentState`, prepares a prompt for the LLM using the full conversation `history`, calls the `LLMClient` to generate a concise summary, and updates the `AgentState` with the `summary`.
    * **Guidance:** Prompt should instruct LLM: "Summarize the key points, challenges, and learnings from the following conversation: [history]".
    * **Acceptance Criteria:**
        * Summarizer function uses `state['history']`.
        * Calls `LLMClient` with a summarization prompt.
        * Updates `state['summary']` with the LLM-generated summary.
        * Returns the updated `AgentState`.
    * **Test Cases:** Unit tests mocking `LLMClient` to verify prompt and state update. Integration test within LangGraph. Test with varying lengths of conversation history.

2.  **Task 4.2: Implement Corrector Agent Node Logic**
    * **Description:** Replace the dummy `check_summary` node function. It takes `AgentState`, prepares a prompt for the LLM including the generated `summary` and `history` (or topic), asking for a basic coherence check (e.g., "Is this summary relevant? YES/NO"). It parses the LLM response and updates the `needs_correction` flag in `AgentState`.
    * **Guidance:** Keep the check very simple for MVP as specified in PRD/TDD. Parse a simple YES/NO response.
    * **Acceptance Criteria:**
        * Corrector function uses `state['summary']` and context.
        * Calls `LLMClient` with a validation prompt.
        * Correctly parses the LLM's YES/NO response.
        * Sets `state['needs_correction']` to `True` if response indicates poor quality, `False` otherwise.
        * Returns the updated `AgentState`.
    * **Test Cases:** Unit tests mocking `LLMClient` for both YES and NO responses, verifying `needs_correction` flag is set correctly. Integration test within LangGraph.

3.  **Task 4.3: Implement Conditional Correction Logic in Graph**
    * **Description:** Implement the conditional edge logic in the LangGraph definition (`src/app/orchestration/graph.py`). Define a function that checks the `needs_correction` flag in the state after the `check_summary` node runs. Add conditional edges routing to `summarize` if `True` or to the finish node if `False`. Ensure the `needs_correction` flag is reset if looping back to `summarize`.
    * **Guidance:** Use `graph.add_conditional_edges`. The condition function takes `AgentState` and returns the name of the next node.
    * **Acceptance Criteria:**
        * Conditional logic function correctly checks `state['needs_correction']`.
        * `graph.add_conditional_edges` is configured correctly linking `check_summary` to `summarize` or `END` based on the condition.
        * The state (`needs_correction`) is appropriately reset if looping back.
    * **Test Cases:** Integration test running the graph, forcing `needs_correction` to `True` -> verify it loops back to `summarize`. Run again forcing `False` -> verify it proceeds to `END`.

---

**Epic 5: API Implementation**

* **Goal:** Expose the agent orchestration logic via a clean, stateless RESTful API as defined in the TDD.
* **Value:** Allows clients (like the Streamlit UI or other services) to interact with the reflection agent, enabling the user journey.

**Tasks:**

1.  **Task 5.1: Define API Request/Response Models**
    * **Description:** Define the Pydantic models (`ReflectionTurnRequest`, `ReflectionTurnResponse`) in `src/app/api/models.py` corresponding to the API design in TDD Section 6. Ensure `AgentStateModel` is consistent with the internal `AgentState`.
    * **Acceptance Criteria:**
        * Pydantic models for request and response bodies are defined.
        * Models accurately reflect the structure specified in the TDD.
    * **Test Cases:** Unit tests validating the Pydantic models.

2.  **Task 5.2: Implement `/reflections/turns` Endpoint**
    * **Description:** Create the API endpoint logic in FastAPI (`src/app/api/endpoints.py`). It should accept the `ReflectionTurnRequest`, handle the logic for initiation vs. subsequent turns (based on presence of `current_state`), invoke the compiled LangGraph orchestrator, and return the `ReflectionTurnResponse`.
    * **Guidance:** Use FastAPI's dependency injection to get the compiled graph. Handle the state passing: receive `current_state` dict, convert to `AgentState` object, invoke graph, convert resulting `AgentState` back to dict for `next_state` in response.
        ```python
        # src/app/api/endpoints.py (simplified)
        from fastapi import APIRouter, HTTPException, Body
        # ... import models, AgentState, compiled_graph

        router = APIRouter()

        @router.post("/turns", response_model=ReflectionTurnResponse)
        async def process_turn(payload: ReflectionTurnRequest = Body(...)):
            if payload.current_state:
                # Subsequent turn
                state = AgentState(**payload.current_state)
                # Add user_input to history if present
                if payload.user_input:
                     state.history.append(("user", payload.user_input))
            else:
                # Initiation turn
                state = AgentState(topic=payload.topic)

            try:
                # Invoke the graph - may need adjustments based on sync/async graph execution
                final_state_dict = {}
                async for event in compiled_graph.astream(state):
                     # Process streaming events if needed, get final state
                     # For simplicity now, assume invoke gives final state:
                     pass # Replace with actual streaming/invocation logic
                # final_state = await compiled_graph.ainvoke(state) # Or invoke
                # final_state_dict = final_state.dict() # Assuming pydantic state

                # Placeholder until graph invocation is finalized
                final_state = state # Replace with actual result
                agent_output = final_state.current_question or final_state.summary or "Error processing."
                is_final = final_state.summary is not None and not final_state.needs_correction # Example condition
                final_state_dict = final_state.model_dump() if isinstance(final_state, BaseModel) else final_state

            except Exception as e:
                # Log error with Sentry/logging
                raise HTTPException(status_code=500, detail=f"Error processing reflection: {e}")

            return ReflectionTurnResponse(
                agent_response=agent_output,
                next_state=final_state_dict,
                is_final_turn=is_final
            )
        ```
    * **Acceptance Criteria:**
        * `POST /api/v1/reflections/turns` endpoint is implemented.
        * Endpoint correctly deserializes request, handles initiation vs subsequent turns.
        * Invokes the compiled LangGraph orchestrator with the appropriate state.
        * Serializes the resulting state and agent response into the response model.
        * Handles basic exceptions during orchestration and returns appropriate HTTP errors.
    * **Test Cases:** API integration tests using `pytest` and `httpx.AsyncClient`. Test initiation call. Test a sequence of calls simulating a conversation. Test state is correctly passed and updated between calls. Test error scenarios (invalid input, orchestration error).

---

**Epic 6: Observability**

* **Goal:** Integrate essential logging and error tracking for visibility into the application's behavior and health.
* **Value:** Crucial for debugging during development, monitoring in testing/deployment, and understanding user interactions.

**Tasks:**

1.  **Task 6.1: Configure Structured Logging**
    * **Description:** Configure Python's `logging` module to output logs in a structured format (JSON). Add log statements at key points: API request start/end, agent node entry/exit, LLM calls initiated/completed, errors encountered. Ensure logs are easily parseable.
    * **Guidance:** Use libraries like `structlog` or configure standard logging with `JSONFormatter`. Log relevant context (e.g., session ID if introduced, agent name).
    * **Acceptance Criteria:**
        * Logging is configured and outputs JSON formatted logs to console/stdout.
        * Key application events are logged with appropriate levels (INFO, WARNING, ERROR).
    * **Test Cases:** Run the application, perform API calls, and manually inspect log output for correct format and content.

2.  **Task 6.2: Integrate Sentry Error Tracking**
    * **Description:** Add the `sentry-sdk` dependency with the FastAPI integration. Initialize Sentry using the DSN from environment variables. Ensure unhandled exceptions in the FastAPI application are automatically captured and sent to Sentry.
    * **Guidance:** Follow `sentry-sdk` documentation for FastAPI integration. Initialize early in the application startup.
    * **Acceptance Criteria:**
        * `sentry-sdk` is added as a dependency.
        * Sentry is initialized using `SENTRY_DSN` environment variable.
        * Unhandled exceptions in API endpoints are visible in the Sentry dashboard.
    * **Test Cases:** Manually trigger an unhandled exception in an API endpoint (e.g., `raise Exception("Test Sentry")`). Verify the error appears in the configured Sentry project.

---

**Epic 7: Minimal Interface & Deployment Prep**

* **Goal:** Provide a basic user interface for testing/demos (optional) and finalize the application packaging and documentation for running/deployment.
* **Value:** Facilitates user testing and prepares the application for potential deployment, fulfilling PRD/TDD requirements.

**Tasks:**

1.  **Task 7.1: (Optional) Build Basic Streamlit UI**
    * **Description:** Create a simple Streamlit application (`streamlit_app.py`) that interacts with the deployed/running API backend. It should allow users to input a topic (optional), start a session, send messages turn-by-turn, and display agent responses and the final summary. Manage state within the Streamlit session.
    * **Guidance:** Use Streamlit's session state to store the `next_state` received from the API. Use `st.text_input`, `st.button`, `st.chat_message`.
    * **Acceptance Criteria:**
        * Streamlit app runs locally.
        * Users can initiate a reflection via the UI.
        * Users can send messages and receive agent responses turn-by-turn.
        * The final summary is displayed.
        * App correctly calls the backend API endpoints.
    * **Test Cases:** Manual end-to-end testing using the Streamlit UI, covering the full reflection flow.

2.  **Task 7.2: Finalize Dockerfile & Deployment Documentation**
    * **Description:** Review and optimize the `Dockerfile` (e.g., ensure non-root user, minimize layers). Update the `README.md` with clear instructions on how to build and run the Docker container locally, including necessary environment variables (`OPENAI_API_KEY`, `SENTRY_DSN`). Add basic steps/guidance for deploying to Google Cloud Run.
    * **Acceptance Criteria:**
        * Dockerfile is optimized and follows best practices.
        * README provides clear, accurate instructions for local build/run via Docker.
        * README includes a section outlining basic steps for Cloud Run deployment (e.g., gcloud commands for build, deploy).
    * **Test Cases:** Follow README instructions to build and run locally. Review Cloud Run deployment steps for clarity and theoretical correctness.
