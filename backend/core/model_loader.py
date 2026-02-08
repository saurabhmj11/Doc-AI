"""
Shared Model Loader

Singleton pattern for loading expensive ML models once.
"""

from sentence_transformers import SentenceTransformer
from config import get_settings

settings = get_settings()

# Global singleton instance
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create shared embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded!")
    return _embedding_model
