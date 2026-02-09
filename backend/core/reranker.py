"""
Reranker Module

Cross-encoder based reranking to improve retrieval quality.
Reranks retrieved chunks using a cross-encoder model for better relevance scoring.
"""

import logging
from typing import Optional, List, Dict, Any

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval quality.
    
    Uses a cross-encoder model to rerank retrieved chunks based on their
    actual relevance to the query, providing better scoring than bi-encoder
    cosine similarity.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Fast inference (~10ms per query-doc pair)
    - Good performance on retrieval tasks
    - Lightweight (80MB)
    """
    
    def __init__(self):
        """Initialize reranker with lazy loading."""
        self._model = None
        self.model_name = settings.reranker_model
        self.batch_size = settings.reranker_batch_size
        logger.info(f"Reranker initialized (model will load on first use: {self.model_name})")
    
    @property
    def model(self):
        """Lazy-load cross-encoder model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}", exc_info=True)
                # Set to False to indicate failure
                self._model = False
        
        # Return None if loading failed
        return self._model if self._model is not False else None
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on cross-encoder scores.
        
        Args:
            query: The user's question
            chunks: List of chunk dicts with 'text' field
            top_k: Number of top chunks to return (default: all)
            
        Returns:
            Reranked list of chunks with added 'rerank_score' field
        """
        if not chunks:
            logger.debug("No chunks to rerank")
            return chunks
        
        # Check if model is available
        if self.model is None:
            logger.warning("Cross-encoder model not available, returning original order")
            return chunks
        
        try:
            # Prepare query-document pairs
            pairs = [[query, chunk['text']] for chunk in chunks]
            
            # Get cross-encoder scores
            logger.debug(f"Reranking {len(chunks)} chunks with query: {query[:50]}...")
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Add rerank scores to chunks and preserve original scores
            reranked_chunks = []
            for chunk, score in zip(chunks, scores):
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = float(score)
                # Preserve original similarity score if it exists
                if 'similarity_score' in chunk:
                    chunk_copy['original_similarity_score'] = chunk['similarity_score']
                reranked_chunks.append(chunk_copy)
            
            # Sort by rerank score (descending)
            reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top-k if specified
            if top_k is not None:
                reranked_chunks = reranked_chunks[:top_k]
            
            logger.info(
                f"Reranked {len(chunks)} chunks, returning top {len(reranked_chunks)}. "
                f"Top score: {reranked_chunks[0]['rerank_score']:.3f}"
            )
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            logger.warning("Falling back to original chunk order")
            return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "model_name": self.model_name,
            "model_loaded": self._model is not None and self._model is not False,
            "batch_size": self.batch_size
        }


# Singleton instance
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get or create reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
