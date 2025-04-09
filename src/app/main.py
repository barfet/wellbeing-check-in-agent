from fastapi import FastAPI
# from .config import settings # Example settings import
from .api import endpoints as reflection_endpoints # Import the new router
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Reflective Learning Agent API", 
    version="1.0.0",
    description="API for interacting with the Reflective Learning Agent."
    )

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """Check if the application is running."""
    logger.info("Health check endpoint called.")
    return {"status": "OK"}

# Include the reflection agent router
app.include_router(
    reflection_endpoints.router, 
    prefix="/api/v1/reflections", 
    tags=["Reflections"]
    )

# Add other routers later if needed: app.include_router(...)
logger.info("FastAPI application configured and routers included.")
