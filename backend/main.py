"""
Ultra Doc-Intelligence API
Author: Development Team
Last Updated: Feb 2024

Quick project I put together for analyzing logistics docs.
Uses RAG + Gemini for Q&A, plus some basic extraction logic.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router
from config import get_settings

# grab env vars first
load_dotenv()
settings = get_settings()

# spin up the app
app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="Logistics document Q&A and data extraction tool",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - wide open for dev, would lock down in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# wire up the routes
app.include_router(router, prefix="/api", tags=["Documents"])


@app.get("/")
async def root():
    """Basic info about the API"""
    return {
        "app": "Ultra Doc-Intelligence",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Quick health check - useful for monitoring"""
    return {
        "status": "ok",
        "llm_ready": bool(settings.gemini_api_key)
    }


# for running directly with python main.py
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
