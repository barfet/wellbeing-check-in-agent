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

---

**Epic 8: Advanced Reflection Capabilities**

* **Goal:** Enhance the Reflective Learning Agent with deeper conversational abilities, improved summary quality, sentiment awareness, and actionable outcomes.
* **Value:** Moves beyond basic reflection to provide a more insightful, empathetic, and impactful experience for the user (Alex), increasing the tool's value for learning and performance improvement.

---

**Task 8.1: Implement Adaptive Multi-Turn Probing Dialogue**

* **Story:** As Alex, I want the agent to ask me multiple relevant follow-up questions about my experience before summarizing, so that I can explore my thoughts more deeply without needing to manually continue the conversation over many separate turns.
* **Description:** This feature enhances the `probe` phase. Instead of moving directly to `summarize` after one probe question, the agent will assess if further probing is beneficial and loop back to ask more questions until the reflection seems sufficiently developed. The decision to continue can be based on conversation length or an LLM-based assessment.
    * **Technical Approach:**
        1.  **Modify Graph:** Introduce a conditional edge after the `probe` node.
        2.  **Condition Function:** Create `should_continue_probing(state: AgentState) -> str`.
            * This function checks criteria like `len(state.history) < MIN_TURNS` (e.g., 5-7 total turns).
            * Optionally, add an LLM call within this function: Prompt="Based on this history [history], is the reflection sufficiently detailed for a meaningful summary? YES/NO".
            * The function returns `"probe"` to continue or `"summarize"` to proceed.
        3.  **State Update:** Add `probe_count: int = 0` to `AgentState`. Increment in the `probe` node. Add a max limit check in `should_continue_probing` to prevent infinite loops (e.g., if `probe_count >= MAX_PROBES`, force "summarize").
        4.  **Probe Node Update:** Ensure the `probe` node's prompt uses sufficient history to ask varied, non-repetitive questions during loops.
        5.  **Graph Edges:** Add edge `probe` -> `probe` (if condition met) and `probe` -> `summarize` (if condition met).
* **Acceptance Criteria:**
    * AC1: A condition function `should_continue_probing` exists and uses state (e.g., history length, probe count) to decide the next node.
    * AC2: The graph includes a conditional edge after `probe` routing to either `probe` or `summarize` based on `should_continue_probing`.
    * AC3: The graph execution successfully loops back to the `probe` node at least once if the continuation condition is met.
    * AC4: The graph includes a mechanism (e.g., max `probe_count`) to prevent infinite probing loops, eventually forcing routing to `summarize`.
    * AC5: The `probe` node generates contextually relevant, non-identical questions during loops (verified via logging/testing).
    * AC6: The API `/turns` response continues to return the `current_question` with `is_final_turn=False` during probe loops.
* **Test Cases:**
    * Unit Test: Test `should_continue_probing` logic with different `AgentState` inputs (varying history length, probe count). If using LLM check, mock it.
    * Integration Test: Mock LLM responses for `probe`. Test the graph flow ensuring it loops correctly based on mock state conditions (e.g., short history -> loop, long history -> summarize, max probes -> summarize).
    * API Test: Simulate a conversation via the API. Provide short answers initially, verifying the API returns multiple probe questions. Then provide longer answers, verifying it eventually proceeds to summary.

---

**Task 8.2: Implement Advanced Summary Self-Correction Loop**

* **Story:** As Alex, I want the agent to rigorously check its own summary for accuracy and relevance against our conversation and retry generating it if needed, so that I receive a high-quality, trustworthy summary.
* **Description:** This enhances the `check_summary` node and the surrounding logic to allow multiple attempts at generating a satisfactory summary, using more specific feedback.
    * **Technical Approach:**
        1.  **State Update:** Add `correction_attempts: int = 0` and `correction_feedback: Optional[str] = None` to `AgentState`.
        2.  **`check_summary` Node Prompt:** Modify the prompt to ask for evaluation against specific criteria (e.g., "Does summary capture challenge? Success? Learning?") and request *specific feedback* if criteria are not met. E.g., "If NO, briefly state what is missing or inaccurate."
        3.  **`check_summary` Node Logic:**
            * Parse LLM response. If criteria met (e.g., response starts "YES"), set `needs_correction=False`, clear `correction_feedback`.
            * If criteria not met, set `needs_correction=True`, parse and store the feedback in `state.correction_feedback`.
        4.  **Routing Function (`route_after_summary_check`):**
            * Check `state.needs_correction`.
            * If `True` AND `state.correction_attempts < MAX_CORRECTION_ATTEMPTS` (e.g., 2): Return `"summarize"`.
            * Otherwise (False OR max attempts reached): Return `END`.
        5.  **`summarize` Node Logic:**
            * Increment `state.correction_attempts` if `state.needs_correction` was True upon entry (or manage attempts in routing).
            * Check if `state.correction_feedback` exists. If so, add it to the summarization prompt: "...Please regenerate the summary addressing the following feedback: [feedback]".
            * Clear `state.correction_feedback` after using it.
        6.  **Final State Handling:** If max attempts are reached and correction is still needed, ensure `error_message` reflects this (e.g., "Summary failed validation after multiple attempts."). The API response logic should handle this state.
* **Acceptance Criteria:**
    * AC1: `check_summary` prompt requests feedback based on specific criteria.
    * AC2: `check_summary` node parses feedback and stores it in `AgentState.correction_feedback` upon failure.
    * AC3: `AgentState` includes `correction_attempts`, which is incremented correctly during the loop.
    * AC4: Routing logic correctly loops back to `summarize` based on `needs_correction` and `correction_attempts` < max.
    * AC5: Routing logic correctly routes to `END` when summary is approved or max attempts are reached.
    * AC6: `summarize` node prompt incorporates `correction_feedback` when regenerating.
    * AC7: If max correction attempts are reached, the final state/API response indicates the summary may be suboptimal (e.g., via `error_message`).
* **Test Cases:**
    * Unit Test: Test `check_summary` parsing of feedback. Test `summarize` prompt generation with feedback. Test `route_after_summary_check` for multi-attempt logic.
    * Integration Test: Mock LLM responses for `check_summary` to simulate: approval, rejection->retry->approval, rejection->retry->rejection->max attempts. Verify graph flow and state updates (`feedback`, `attempts`).
    * API Test: Simulate a conversation where the initial summary (mocked) fails validation, verify the graph loops (may require inspecting logs or intermediate states if API only returns final result), and check the final output (either corrected summary or error state).

---

**Task 8.3: Integrate Sentiment Analysis to Influence Probing**

* **Story:** As Alex, I want the agent to react more empathetically to my expressed feelings, perhaps by asking slightly different follow-up questions based on whether I sound positive or negative, so that the conversation feels more natural and supportive.
* **Description:** Introduces sentiment analysis of the user's input to subtly adapt the tone or focus of the subsequent probe question.
    * **Technical Approach:**
        1.  **New Node:** Create an `analyze_sentiment(state: AgentState, llm_client: LLMClient) -> AgentState` node.
        2.  **Node Logic:**
            * Get the last user utterance from `state.history`.
            * Call `llm_client` with a prompt: "Classify the sentiment of the user's statement: '[utterance]'. Respond only with POSITIVE, NEGATIVE, or NEUTRAL."
            * Parse the response. Handle errors/unclear responses by defaulting to NEUTRAL.
            * Store the result in `AgentState.last_sentiment: Optional[str]`.
        3.  **Graph Update:** Insert this node into the flow *before* `probe`. Edges: `initiate` -> `analyze_sentiment` (or maybe after user input is confirmed?), `analyze_sentiment` -> `probe`. If multi-turn probing exists, the loop might be `probe` -> `analyze_sentiment` -> `probe`. *Decision: Place it reliably after user input is added and before probe runs.* The API logic already adds user input to state before `ainvoke`, so the sequence inside `ainvoke` should start with `analyze_sentiment`. Modify entry point or first edge accordingly. *Revised Graph Flow Idea: `(Entry: Add User Input/Initiate)` -> `analyze_sentiment` -> `probe` -> `(Condition: should_continue_probing)` -> `analyze_sentiment` (if looping) OR `summarize`.*
        4.  **`probe` Node Update:** Modify the prompt generation logic within `probe`. Access `state.last_sentiment`. Add specific instructions based on sentiment:
            * If NEGATIVE: "...The user expressed negativity. Ask a question focusing on understanding the challenge or feeling."
            * If POSITIVE: "...The user expressed positivity. Ask a question exploring the success or positive feeling."
            * If NEUTRAL (or None): Use the standard prompt.
* **Acceptance Criteria:**
    * AC1: New graph node `analyze_sentiment` exists and correctly calls LLM for sentiment classification.
    * AC2: `AgentState` includes `last_sentiment`, updated by `analyze_sentiment` (defaults correctly on error).
    * AC3: The graph structure ensures `analyze_sentiment` runs before `probe` after user input.
    * AC4: `probe` node prompt generation logic dynamically changes based on `state.last_sentiment`.
    * AC5: The change in prompt demonstrably influences the type of question generated by the LLM (verified via logging/testing).
* **Test Cases:**
    * Unit Test: Test `analyze_sentiment` node logic (mock LLM for POSITIVE/NEGATIVE/NEUTRAL/Error). Test `probe` node's prompt generation with different `last_sentiment` values in the input state.
    * Integration Test: Mock LLM calls. Test graph flow ensuring `analyze_sentiment` runs, updates state, and `probe` receives the sentiment. Verify the correct prompt variation is generated in `probe`.
    * API Test (Qualitative): Send API requests with user inputs having strong positive or negative sentiment. Observe the agent's next question – does it seem appropriately tailored? (Requires inspecting agent response).

---

**Task 8.4: Implement Post-Summary Goal-Setting Agent**

* **Story:** As Alex, after reviewing the reflection summary, I want the agent to help me identify one concrete action I can take based on my reflection, so that I can turn insights into practical improvements.
* **Description:** Adds a new conversational phase after summary approval, where the agent prompts the user to define an actionable goal. This requires extending the conversation flow and API handling.
    * **Technical Approach:**
        1.  **New Node:** Create `suggest_goal_step(state: AgentState, llm_client: LLMClient) -> AgentState`.
        2.  **State Update:** Add `actionable_goal: Optional[str] = None` and `goal_setting_active: bool = False` to `AgentState`.
        3.  **`suggest_goal_step` Logic:**
            * Prompt LLM using `summary` and `history`: "Based on this reflection [summary/history], ask the user *one* question to help them define a small, actionable step for the future."
            * Update `state.current_question` with the LLM response.
            * Set `state.goal_setting_active = True`.
            * Add the goal question to `history`.
        4.  **Graph Update:** Change the routing after successful summary check (`route_after_summary_check`). Instead of routing to `END`, route to `suggest_goal_step` if `needs_correction is False`.
        5.  **New Node & Final Step:** Create a simple node `capture_goal(state: AgentState) -> AgentState`. This node *only* runs if `goal_setting_active` is True and the user provides input.
            * Logic: Get the last user utterance (the goal) from history. Store it in `state.actionable_goal`. Add a final confirmation message to history (e.g., "agent: Got it. Goal captured."). Clear `current_question`. Set `goal_setting_active = False`.
            * Add edge `suggest_goal_step` -> `capture_goal` (This seems wrong, `capture_goal` needs user input first).
        6.  **Revised Graph/API Interaction:**
            * `route_after_summary_check` routes approved summaries to `suggest_goal_step`.
            * `suggest_goal_step` sets `goal_setting_active=True` and generates `current_question`. The graph *ends* here for this invocation.
            * API `process_turn` checks the *returned* state: if `goal_setting_active` is True and `current_question` is set, it returns the question with `is_final_turn=False`.
            * The *next* API call from the client will have `goal_setting_active=True` in `current_state` and the user's goal in `user_input`.
            * The graph needs an entry point or logic to detect this state. Add a conditional entry point or modify the main flow: If `state.goal_setting_active` is True upon entry, route directly to `capture_goal`.
            * `capture_goal` runs, updates state (sets goal, clears question), and routes to `END`.
            * API `process_turn` receives the final state from `capture_goal` -> `END`, sets `is_final_turn=True`, and returns a confirmation message.
* **Acceptance Criteria:**
    * AC1: New graph node `suggest_goal_step` exists and prompts LLM for a goal-setting question using summary/history.
    * AC2: Graph routes to `suggest_goal_step` after successful summary validation.
    * AC3: `AgentState` includes `actionable_goal` and `goal_setting_active` fields, updated correctly by `suggest_goal_step`.
    * AC4: The API response for the turn ending with `suggest_goal_step` contains the goal question and `is_final_turn=False`.
    * AC5: A mechanism exists (e.g., conditional entry, new `capture_goal` node) to handle the subsequent API request containing the user's goal.
    * AC6: The user's goal input is stored in `AgentState.actionable_goal`.
    * AC7: The final API response after capturing the goal includes a confirmation and `is_final_turn=True`.
* **Test Cases:**
    * Unit Test: Test `suggest_goal_step` logic (mock LLM). Test `capture_goal` logic. Test conditional graph entry/routing logic for goal phase.
    * Integration Test: Test graph flow showing transition from summary approval to `suggest_goal_step`. Test the goal capture path.
    * API Test: Simulate a conversation through summary approval -> verify goal question is returned (`is_final=False`). Send another request with goal input -> verify final confirmation (`is_final=True`) and check `next_state` contains the captured goal.

---

