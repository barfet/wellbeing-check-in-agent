# Stage 1: Build environment
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (if any, e.g., for C extensions - none needed now)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy requirements files
COPY requirements.txt ./

# Install runtime dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set the FastAPI app module
    APP_MODULE="app.main:app" \
    # Add virtual environment to PATH
    PATH="/opt/venv/bin:$PATH"

# Create a non-root user and group
RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid appgroup appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY ./src/app ./app

# Change ownership of the app directory and venv to the non-root user
# Ensure the user can write to logs or temp files if needed (adjust permissions accordingly)
RUN chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /opt/venv 

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "app.main:app"] 