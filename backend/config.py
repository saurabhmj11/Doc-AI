"""Configuration and environment variables for Ultra Doc-Intelligence."""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    gemini_api_key: str = ""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Document processing
    max_file_size_mb: int = 10
    allowed_extensions: list[str] = ["pdf", "docx", "txt"]
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval settings
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    similarity_threshold: float = 0.15  # Lowered for better retrieval
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.35
    grounding_coverage_threshold: float = 0.6
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
