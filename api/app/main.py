"""
FastAPI backend for Machine Vision MCP.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.app.routers import analysis, health, upload, claude

# Create necessary directories
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Machine Vision MCP API",
    description="API for computer vision tasks using scikit-image",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(claude.router, prefix="/api", tags=["claude"])

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Machine Vision MCP API is running",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("api.app.main:app", host="0.0.0.0", port=8000, reload=True)