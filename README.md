# Reflective Learning Agent

AI Agent for guided reflection using LangGraph and LLMs.

## Project Status

*(Add brief status, e.g., MVP complete, In Development)*

## Setup

1.  **Python:** Ensure you have Python >= 3.11 installed.
2.  **Clone:** Clone the repository: `git clone <repository_url>`
3.  **Navigate:** `cd wellbeing-check-in-agent`
4.  **Environment:** Create a virtual environment: `python3 -m venv .venv`
5.  **Activate:** Activate the virtual environment: `source .venv/bin/activate` (Linux/macOS) or `.\\venv\\Scripts\\activate` (Windows)
6.  **Install:** Install dependencies: `pip install -r requirements.txt`
7.  **(Optional) Dev Install:** Install development dependencies: `pip install -r requirements-dev.txt`

## Environment Variables

This application requires certain environment variables to be set, primarily for accessing external services like the LLM.

1.  Create a file named `.env` in the project root directory.
2.  Add the following required variable:

    ```dotenv
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

3.  **(Optional) API Base URL:** If running the Streamlit UI (`streamlit_app.py`) separately from the backend and the backend is not at `http://localhost:8000`, set:
    ```dotenv
    API_BASE_URL="http://your_backend_host:port"
    ```

4.  **(Optional) Sentry DSN:** If Sentry integration is added (Epic 6):
    ```dotenv
    SENTRY_DSN="your_sentry_dsn_here"
    ```

**Note:** The application uses `python-dotenv` to automatically load variables from the `.env` file when running locally (e.g., via `uvicorn` or `streamlit run`). When running via Docker, these variables need to be passed explicitly.

## Usage

### Running Locally (Backend API)

Ensure your `.env` file is created and variables are set.
Activate the virtual environment (`source .venv/bin/activate`) and run:

```bash
uvicorn src.app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
You can access the auto-generated documentation at `http://localhost:8000/docs`.

### Running Locally (Streamlit UI)

Ensure the backend API is running (either locally via `uvicorn` or via Docker).
Ensure your `.env` file is created.
Activate the virtual environment (`source .venv/bin/activate`) and run:

```bash
streamlit run streamlit_app.py
```

### Running with Docker

1.  **Build the Image:**
    ```bash
    docker build -t reflective-learning-agent .
    ```

2.  **Run the Container:** You *must* pass the `OPENAI_API_KEY` environment variable to the container.
    ```bash
    # Replace your_openai_api_key_here with your actual key
    docker run -d -p 8000:8000 --name reflective-agent --env OPENAI_API_KEY="your_openai_api_key_here" reflective-learning-agent
    ```
    *   `-d`: Run in detached mode (background).
    *   `-p 8000:8000`: Map port 8000 on the host to port 8000 in the container.
    *   `--name reflective-agent`: Assign a name to the container.
    *   `--env OPENAI_API_KEY=...`: Pass the required environment variable.

3.  **Access:** The API will be available at `http://localhost:8000`.
4.  **Stop:** `docker stop reflective-agent`
5.  **Remove:** `docker rm reflective-agent`

## Deployment (Google Cloud Run - Basic Guide)

This provides basic steps for deploying the container to Google Cloud Run.

**Prerequisites:**
*   `gcloud` CLI installed and configured.
*   Google Cloud project created with billing enabled.
*   Artifact Registry API enabled.
*   Cloud Run API enabled.

**Steps:**

1.  **Set Project ID:**
    ```bash
    export PROJECT_ID="your-gcp-project-id"
    gcloud config set project $PROJECT_ID
    ```

2.  **(Optional) Create Artifact Registry Repository:** (If you don't have one)
    ```bash
    export REGION="your-chosen-region" # e.g., us-central1
    export REPO_NAME="reflective-agent-repo"
    gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repository for Reflective Agent"
    ```

3.  **Build and Push Docker Image:**
    ```bash
    # Format: REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/IMAGE_NAME:TAG
    export IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/reflective-learning-agent:latest"
    gcloud builds submit --tag $IMAGE_NAME .
    ```
    (This uses Cloud Build to build the image and push it to Artifact Registry)

4.  **Deploy to Cloud Run:**
    ```bash
    export SERVICE_NAME="reflective-learning-agent"

    # Get your OpenAI API Key (ensure it's handled securely, consider Secret Manager)
    export OPENAI_API_KEY_VALUE="your_openai_api_key_here"

    gcloud run deploy $SERVICE_NAME \\
        --image=$IMAGE_NAME \\
        --platform=managed \\
        --region=$REGION \\
        --allow-unauthenticated \\
        --set-env-vars="OPENAI_API_KEY=${OPENAI_API_KEY_VALUE}" \\
        --port=8000
        # Add --update-secrets for Secret Manager integration
        # Add other env vars like SENTRY_DSN if needed: --set-env-vars="OPENAI_API_KEY=...,SENTRY_DSN=..."
    ```
    *   `--allow-unauthenticated`: Allows public access. Adjust as needed.
    *   `--set-env-vars`: Set environment variables. **Important:** For production, manage secrets like API keys securely using Google Secret Manager instead of passing them directly.
    *   `--port=8000`: Specify the container port.

5.  **Access:** Cloud Run will provide a service URL upon successful deployment.

## Development

*(Details about running tests, linters, etc., can be added here)*

```bash
# Run unit tests
pytest tests/

# Run linter (Ruff)
ruff check src/ tests/

# Format code (Ruff)
ruff format src/ tests/
```