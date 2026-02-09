"""
Ultra Doc-Intelligence API
Author: Development Team
Last Updated: Feb 2024

Quick project I put together for analyzing logistics docs.
Uses RAG + Gemini for Q&A, plus some basic extraction logic.
"""

import os
from dotenv import load_dotenv

# Load env vars from the same directory as this file
# This must happen BEFORE importing config or other modules that use settings
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import get_settings
from core.error_handling import setup_logging

settings = get_settings()

# Configure logging before initializing components
setup_logging()

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
    allow_origins=[
        "https://doc-ai-frontend.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
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
    from core.error_handling import get_gemini_circuit_breaker, get_ollama_circuit_breaker
    
    # Get circuit breaker states
    gemini_cb = get_gemini_circuit_breaker() if settings.llm_mode == "online" else None
    ollama_cb = get_ollama_circuit_breaker() if settings.llm_mode == "offline" else None
    
    health_status = {
        "status": "ok",
        "llm_mode": settings.llm_mode,
        "llm_configured": bool(settings.gemini_api_key) if settings.llm_mode == "online" else True,
    }
    
    # Add circuit breaker status
    if gemini_cb:
        health_status["gemini_circuit_breaker"] = gemini_cb.state.value
    if ollama_cb:
        health_status["ollama_circuit_breaker"] = ollama_cb.state.value
    
    return health_status


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
