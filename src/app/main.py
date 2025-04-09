from fastapi import FastAPI

app = FastAPI(title="Reflective Learning Agent API", version="0.1.0")


@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "OK"}


# Add other routers later: app.include_router(...)
