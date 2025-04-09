import pytest

# Import the FastAPI app instance
# Use absolute import path based on structure: src/app/main.py
from app.main import app
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check():
    """Test the /health endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
