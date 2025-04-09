**Technical Design Document: Reflective Learning Agent (MVP v1.0)**

* **Document Status:** Draft
* **Version:** 1.0
* **Date:** April 9, 2025
* **Author:** Principal Software Engineer/Architect
* **Related PRD:** [Link to PRD: Reflective Learning Agent (MVP v1.0)]

**1. Introduction & Overview**

This document details the technical design for the Minimum Viable Product (MVP) of the Reflective Learning Agent, as specified in the corresponding PRD (v1.0). The objective is to implement a backend service that facilitates AI-guided user reflections through a conversational interface. This service will leverage the LangGraph framework for agent orchestration, interact with an external Large Language Model (LLM), and expose a RESTful API for client interaction. The focus is on validating the core technical approach, agentic flow, and integration points within the MVP scope.

**2. Goals & Non-Goals (Technical)**

**2.1. Goals:**

* Implement a containerized Python service exposing a well-defined REST API for managing reflection sessions.
* Utilize LangGraph to orchestrate the stateful conversational flow between defined agents (Initiator, Prober, Summarizer, Corrector).
* Integrate securely and reliably with a configured external LLM API (e.g., OpenAI).
* Implement the core functional requirements (FR1-FR6) from the PRD, including the basic self-correction loop.
* Establish basic observability through structured logging and error tracking (Sentry).
* Ensure the API design is stateless where feasible for simplified MVP deployment and scaling.

**2.2. Non-Goals (for MVP):**

* Implementing user authentication or authorization.
* Persistent storage of conversation history or user data beyond the transient state needed for a single session's API calls.
* High-availability setup or automated scaling beyond basic container orchestration capabilities (e.g., Cloud Run scaling).
* Implementing asynchronous task queues (e.g., Celery, ARQ); direct `async/await` is sufficient for MVP's I/O.
* Advanced prompt engineering optimizations or A/B testing frameworks.
* Building the frontend UI (Streamlit/other); focus is on the backend API.
* Production-grade security hardening (focus on fundamentals like API key handling).

**3. Proposed Architecture**

The system follows a simple service-based architecture:

```mermaid
graph LR
    subgraph Client
        UI[UI / API Client e.g., Streamlit]
    end

    subgraph Backend Service (Docker Container)
        API[API Service - FastAPI]
        Orchestrator[Agent Orchestrator - LangGraph]
        Agents[Agent Modules - Python]
        LLMClient[LLM Client Wrapper]
    end

    subgraph External Services
        LLM[External LLM API e.g., OpenAI]
        Sentry[Sentry Error Tracking]
    end

    UI -- HTTP Requests --> API;
    API -- Invokes --> Orchestrator;
    Orchestrator -- Uses --> Agents;
    Agents -- Uses --> LLMClient;
    LLMClient -- API Calls --> LLM;
    API -- Sends Errors --> Sentry;
    Orchestrator -- Sends Errors --> Sentry;
```

* **API Service (FastAPI):** Entry point for all client requests. Handles request validation, manages interaction flow by invoking the orchestrator, and formats responses. Designed to be stateless regarding conversation history between calls.
* **Agent Orchestrator (LangGraph):** Manages the state machine of the conversation using LangGraph. Defines the nodes (agents) and edges (transitions) based on the conversational logic and agent state.
* **Agents (Python Modules):** Discrete Python functions/classes representing each agent role defined in the PRD (Initiator, Prober, Summarizer, Corrector). They contain the core logic and prompt definitions.
* **LLM Client Wrapper:** A dedicated module abstracting the communication details with the chosen external LLM API. Handles API key management, request formatting, response parsing, and basic error handling/retries.

**4. Component Deep Dive**

* **API Service (FastAPI):**
    * Framework: FastAPI (for async capabilities, Pydantic integration, auto-docs).
    * Responsibilities: Define API endpoints (Section 6), implement request/response models (Pydantic), handle HTTP methods, pass necessary state to/from the orchestrator via API calls, basic exception handling, Sentry integration.
    * Server: Uvicorn.
* **Agent Orchestrator (LangGraph):**
    * Framework: Langchain Core, LangGraph.
    * Responsibilities: Define the `AgentState` graph state object (Section 7), construct the `StatefulGraph`, define agent nodes (`RunnableLambda` or similar), define entry/exit points and conditional edges based on `AgentState` values (e.g., for correction loop), execute graph runs.
* **Agents (Python Modules):**
    * Structure: Separate Python functions (e.g., `run_initiator_agent(state: AgentState)`, `run_probe_agent(state: AgentState)`).
    * Responsibilities: Receive current `AgentState`, prepare prompts using state information (topic, history), invoke `LLM Client`, process LLM response, update `AgentState` with results (e.g., new question, summary, correction flag).
* **LLM Client Wrapper:**
    * Structure: A Python class (e.g., `LLMService`) instantiated with API keys (from environment variables).
    * Responsibilities: Provide methods like `get_completion(prompt: str, max_tokens: int)` or similar. Load API key securely (e.g., using `python-dotenv` locally, environment variables in deployment). Implement simple retry logic for transient LLM API errors. Handle API exceptions gracefully.

**5. Agent Orchestration (LangGraph)**

* **State Definition (`AgentState`):** A TypedDict or Pydantic model containing:
    * `topic: Optional[str]`
    * `history: List[Tuple[str, str]]` # List of (speaker, utterance) pairs, e.g., [("user", "..."), ("agent", "...")]
    * `current_question: Optional[str]`
    * `summary: Optional[str]`
    * `needs_correction: bool = False`
    * `error_message: Optional[str]` # To capture errors during flow
* **Graph Structure:**
    * Nodes: `initiate`, `probe`, `summarize`, `check_summary`, `final_summary_node` (placeholder if needed before end).
    * Entry Point: `initiate`
    * Edges:
        * `initiate` -> `probe`
        * `probe` -> `summarize` (MVP: only one probe)
        * `summarize` -> `check_summary`
        * `check_summary` -> (Conditional):
            * If `needs_correction == True`: -> `summarize` (Reset `needs_correction` flag before re-entering summarize)
            * If `needs_correction == False`: -> `final_summary_node` (or END)
    * END Node: Represents completion of the graph execution for a turn.

**6. API Design (RESTful)**

* **Base URL:** `/api/v1`
* **State Management:** Conversation state (`AgentState`) will be passed back and forth between the client and server in API calls to keep the backend stateless regarding sessions.

* **Endpoint:** `POST /reflections/turns`
    * **Purpose:** Process a single turn in the reflection dialogue. Handles both initiation and subsequent turns.
    * **Request Body:**
        ```json
        {
          "current_state": Optional[Dict], // Full AgentState from previous turn, null for initiation
          "user_input": Optional[str], // User's response for this turn, null for initiation
          "topic": Optional[str] // Provided only on initiation if user specifies a topic
        }
        ```
    * **Response Body (Success - 200 OK):**
        ```json
        {
          "agent_response": str, // The agent's question or final summary/message
          "next_state": Dict, // The updated AgentState to be sent in the next request
          "is_final_turn": bool // True if this is the last turn (summary presented)
        }
        ```
    * **Response Body (Error - 4xx/5xx):** Standard FastAPI error response schema.
    * **Pydantic Models:** Define `ReflectionTurnRequest`, `ReflectionTurnResponse`, and `AgentStateModel` mirroring the `AgentState` TypedDict.

**7. Data Model**

* **Primary Model:** `AgentState` (defined in Section 5) - used transiently within graph execution and for API state transfer.
* **Persistence:** No database persistence in MVP. State exists only within the lifecycle of graph execution and API request/response cycles.

**8. Technology Stack**

* **Language:** Python (>= 3.10)
* **Web Framework:** FastAPI
* **Orchestration:** Langchain Core, LangGraph
* **Data Validation:** Pydantic
* **LLM Integration:** `openai` library (or relevant library for chosen LLM provider)
* **API Server:** Uvicorn
* **Containerization:** Docker, Docker Compose (for local dev)
* **CI/CD:** GitHub Actions
* **Error Tracking:** Sentry SDK (`sentry-sdk[fastapi]`)
* **Environment Mgmt:** `python-dotenv` (for local dev)
* **Testing:** `pytest`, `httpx` (for API testing)

**9. Infrastructure & Deployment (MVP)**

* **Containerization:** A `Dockerfile` will be created using best practices (e.g., multi-stage builds, non-root user).
* **Local Development:** Use `docker-compose.yml` to manage the service and potentially related tools (if any). Document setup using `.env` for API keys.
* **Deployment Target:** Google Cloud Run.
    * Build container image using Cloud Build (triggered by GitHub Actions ideally).
    * Push image to Google Artifact Registry.
    * Configure Cloud Run service (CPU/memory allocation, concurrency, environment variables for `LLM_API_KEY`, `SENTRY_DSN`).
* **CI Workflow (GitHub Actions):**
    * Trigger: On push/PR to `main` branch.
    * Jobs:
        * Lint (`ruff check .`, `ruff format --check .`)
        * Test (`pytest`)
        * (Optional on merge to main) Build and Push Docker image to GAR.
        * (Optional on merge to main) Deploy to Cloud Run.

**10. Observability (MVP)**

* **Logging:**
    * Use Python's standard `logging` module.
    * Configure FastAPI to use a structured logging format (e.g., JSON) via middleware or configuration.
    * Log key events: API request received/responded, graph execution start/end, agent node entry/exit, LLM request/response (potentially token counts, duration), errors.
    * Log level configurable via environment variable.
* **Error Tracking:**
    * Integrate `sentry-sdk` with the FastAPI integration.
    * Initialize with DSN from environment variable.
    * Unhandled exceptions will be automatically captured. Manually capture significant errors (e.g., LLM API failures after retries).

**11. Security Considerations (MVP)**

* **API Keys:** LLM API keys and Sentry DSN must be loaded from environment variables, not hardcoded. Use secret management in Cloud Run.
* **Input Validation:** Leverage Pydantic for strict request body validation in FastAPI to prevent basic injection attacks.
* **Dependencies:** Use a tool like `pip-audit` or GitHub Dependabot (manual check for MVP) to identify known vulnerabilities in dependencies.
* **Rate Limiting:** Not implemented at the application level for MVP. Rely on Cloud Run's inherent limits or configure limits if deploying behind a gateway later.
* **DoS:** Basic protection provided by Cloud Run infrastructure. No specific application-level DoS mitigation.

**12. Scalability & Performance Considerations (MVP)**

* **Concurrency:** FastAPI's async model handles concurrent I/O-bound operations (LLM calls) effectively.
* **Statelessness:** Passing state via the API allows for simple horizontal scaling (multiple container instances on Cloud Run) without needing shared state management for MVP.
* **Bottlenecks:** LLM latency is the primary performance bottleneck. LLM Client should implement sensible timeouts. Potential LLM provider rate limits need consideration.
* **Payload Size:** Passing full `AgentState` via API increases network payload. Monitor size; if it becomes excessive (unlikely in MVP), alternative state management (e.g., Redis) would be needed.

**13. Future Considerations (Technical Hooks)**

* **State Management:** The API design using `current_state`/`next_state` can be adapted. Future work could involve storing state in Redis (for active sessions) or MongoDB (for historical analysis), requiring changes primarily in the API service layer to load/save state instead of passing it.
* **Async Tasks:** For longer-running agent steps (beyond simple LLM calls), the architecture allows integrating task queues (Celery, ARQ) invoked by agent modules.
* **Agent Modularity:** Keeping agents as distinct functions/modules allows for easier addition or modification of steps in the LangGraph definition.
* **Database Integration:** Adding a database (e.g., MongoDB) for storing summaries or user profiles would primarily impact the API service and potentially add new agents for data interaction.

**14. Open Questions & Risks**

* **LLM Prompt Robustness:** Significant effort may be required in prompt tuning to ensure reliable question generation, summarization, and correction across diverse user inputs. Risk: Unpredictable/poor quality agent responses.
* **LangGraph State Complexity:** Ensuring the `AgentState` and conditional logic in LangGraph remain manageable as features are added. Risk: Increased debugging difficulty.
* **API State Payload Size:** Need to monitor the size of the `AgentState` JSON as conversation history grows. Risk: Potential performance issues or request size limits if conversations become extremely long.
* **LLM Cost Management:** Monitor token usage during development and testing.

**15. Alternatives Considered**

* **Server-Side Session State:** Using Redis or in-memory storage for session state. Rejected for MVP to prioritize statelessness and simplify initial deployment.
* **Alternative Orchestration:** Custom state machine implementation. Rejected to leverage LangGraph's features and align with desired skill development.
* **Synchronous Framework (e.g., Flask):** Rejected due to FastAPI's superior handling of async I/O, crucial for efficient LLM interaction.
