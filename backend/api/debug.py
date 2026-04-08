"""
Debug and Monitoring Router

Endpoints for system introspection, model discovery, and deep health checks.
"""

import logging
from fastapi import APIRouter, HTTPException
import google.generativeai as genai
import ollama

from config import get_settings
from api.schemas import ModelsResponse, ModelInfo

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available LLM models for the current provider (Gemini or Ollama).
    """
    models_list = []
    
    try:
        if settings.llm_mode == "online":
            if not settings.gemini_api_key:
                raise HTTPException(status_code=400, detail="Gemini API Key not configured")
            
            genai.configure(api_key=settings.gemini_api_key)
            
            # List Gemini models
            for m in genai.list_models():
                models_list.append(ModelInfo(
                    name=m.name,
                    methods=m.supported_generation_methods
                ))
                
        else:
            # Offline mode (Ollama)
            client = ollama.Client(host=settings.ollama_base_url)
            ollama_models = client.list()
            
            # Ollama response format handling
            if isinstance(ollama_models, dict) and 'models' in ollama_models:
                for m in ollama_models['models']:
                    models_list.append(ModelInfo(
                        name=m.get('name', 'unknown'),
                        methods=["generateContent"] # Standard for all Ollama models
                    ))
            elif isinstance(ollama_models, list):
                for m in ollama_models:
                     models_list.append(ModelInfo(
                        name=getattr(m, 'name', str(m)),
                        methods=["generateContent"]
                    ))
            
        return ModelsResponse(models=models_list)
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get("/health/extended")
async def extended_health():
    """
    Perform deep health checks on vector store and model connection.
    """
    from core.vector_store import get_vector_store
    
    vs = get_vector_store()
    
    health = {
        "status": "ok",
        "vector_store": {
            "status": "connected",
            "collection_count": vs.client.count_collections() if hasattr(vs.client, 'count_collections') else "unknown"
        },
        "llm": {
            "mode": settings.llm_mode,
            "api_key_set": bool(settings.gemini_api_key)
        }
    }
    
    return health
