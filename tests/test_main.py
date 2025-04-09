import pytest
from httpx import AsyncClient

# Import the FastAPI app instance
# Use absolute import path based on project structure
from src.app.main import app


@pytest.mark.asyncio
async def test_health_check():
    """Test the /health endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
