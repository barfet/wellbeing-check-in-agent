# Reflective Learning Agent

AI Agent for guided reflection using LangGraph and LLMs.

## Setup

1. Ensure you have Python 3.11 installed.
2. Clone the repository.
3. Create a virtual environment: `python3 -m venv .venv`
4. Activate the virtual environment: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. (Optional) Install development dependencies: `pip install -r requirements-dev.txt`

## Usage

*(Details to be added)*

### Running Locally

Activate the virtual environment (`source .venv/bin/activate`) and run:

```bash
uvicorn src.app.main:app --reload
```

### Running with Docker

1. Build the image: `docker build -t reflective-learning-agent .`
2. Run the container: `docker run -p 8000:8000 reflective-learning-agent`