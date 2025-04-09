import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app instance
# Use absolute import path based on project structure
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
