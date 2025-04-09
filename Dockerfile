# Stage 1: Build environment with Poetry
FROM python:3.11-slim as builder

WORKDIR /app

# Install required tools and Poetry
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy only files necessary for dependency installation
COPY pyproject.toml /app/
RUN touch README.md

# Install dependencies
# Don't install the project itself
# Skip development dependencies
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --without dev

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set the FastAPI app module
    APP_MODULE="app.main:app"

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv .venv

# Copy the application code
COPY ./src/app ./app

# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app.main:app"] 