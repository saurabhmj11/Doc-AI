"""
Shared Model Loader

Lazy loading for expensive ML models to prevent blocking server startup.
All imports are deferred until first use.
"""

from config import get_settings

settings = get_settings()

# Global singleton instance
_embedding_model = None


def get_embedding_model():
    """Get or create shared embedding model singleton with lazy imports."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        # Lazy import - only load sentence_transformers when needed
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded!")
    return _embedding_model
