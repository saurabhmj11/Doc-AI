"""
Confidence Scorer Module

Multi-signal confidence scoring for RAG answers.
"""

import numpy as np
from config import get_settings
from core.model_loader import get_embedding_model

settings = get_settings()


class ConfidenceScorer:
    """
    Multi-signal confidence scoring for RAG answers.
    
    Signals:
    - Retrieval similarity (30%): How similar top chunks are to query
    - Chunk agreement (25%): Consistency across retrieved chunks
    - Answer coverage (25%): How much of answer is grounded in context
    - Query-answer relevance (20%): Semantic similarity of Q-A pair
    """
    
    WEIGHT_RETRIEVAL = 0.30
    WEIGHT_CHUNK_AGREEMENT = 0.25
    WEIGHT_ANSWER_COVERAGE = 0.25
    WEIGHT_QA_RELEVANCE = 0.20
    
    def __init__(self):
        self.embedding_model = get_embedding_model()  # Shared singleton
    
    def compute_confidence(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[dict],
        context: str
    ) -> tuple[float, dict]:
        """
        Compute overall confidence score for an answer.
        
        Args:
            query: User's question
            answer: Generated answer
            retrieved_chunks: List of retrieved chunks with similarity scores
            context: Combined context text used for generation
            
        Returns:
            Tuple of (confidence_score, breakdown_dict)
        """
        # Compute individual signals
        retrieval_score = self._compute_retrieval_score(retrieved_chunks)
        chunk_agreement_score = self._compute_chunk_agreement(retrieved_chunks)
        coverage_score = self._compute_answer_coverage(answer, context)
        qa_relevance_score = self._compute_qa_relevance(query, answer)
        
        # Weighted combination
        confidence = (
            self.WEIGHT_RETRIEVAL * retrieval_score +
            self.WEIGHT_CHUNK_AGREEMENT * chunk_agreement_score +
            self.WEIGHT_ANSWER_COVERAGE * coverage_score +
            self.WEIGHT_QA_RELEVANCE * qa_relevance_score
        )
        
        breakdown = {
            "retrieval_similarity": round(retrieval_score, 3),
            "chunk_agreement": round(chunk_agreement_score, 3),
            "answer_coverage": round(coverage_score, 3),
            "qa_relevance": round(qa_relevance_score, 3),
            "weights": {
                "retrieval": self.WEIGHT_RETRIEVAL,
                "chunk_agreement": self.WEIGHT_CHUNK_AGREEMENT,
                "coverage": self.WEIGHT_ANSWER_COVERAGE,
                "qa_relevance": self.WEIGHT_QA_RELEVANCE
            }
        }
        
        return round(confidence, 3), breakdown
    
    def _compute_retrieval_score(self, chunks: list[dict]) -> float:
        """Compute score based on retrieval similarity of top chunks."""
        if not chunks:
            return 0.0
        
        # Weight top chunks more heavily
        scores = [chunk.get("similarity_score", 0) for chunk in chunks[:3]]
        if not scores:
            return 0.0
        
        # Weighted average (first chunk counts most)
        weights = [0.5, 0.3, 0.2][:len(scores)]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights[:len(scores)])
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _compute_chunk_agreement(self, chunks: list[dict]) -> float:
        """
        Compute how consistent retrieved chunks are with each other.
        High agreement = chunks support similar information.
        """
        if len(chunks) < 2:
            return 1.0  # Single chunk always agrees with itself
        
        # Get embeddings using semantic similarity task type for direct comparison
        texts = [chunk.get("text", "") for chunk in chunks[:5]]
        embeddings = self.embedding_model.encode(
            texts, 
            convert_to_numpy=True,
            task_type="semantic_similarity"
        )
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Average agreement
        avg_sim = np.mean(similarities)
        
        # Scale to 0-1 (assuming agreement > 0.3 is good)
        return min(1.0, max(0.0, (avg_sim + 0.2) / 1.2))
    
    def _compute_answer_coverage(self, answer: str, context: str) -> float:
        """
        Compute what fraction of the answer is grounded in the context.
        Uses token-level matching with some fuzzy matching.
        """
        if not answer or not context:
            return 0.0
        
        # Tokenize (simple word-level)
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'and',
                      'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 
                      'out', 'off', 'over', 'under', 'again', 'further', 'once',
                      'this', 'that', 'these', 'those', 'it', 'its', "it's"}
        
        answer_tokens = answer_tokens - stop_words
        
        if not answer_tokens:
            return 1.0  # All stop words = trivially covered
        
        # Count matched tokens
        matched = answer_tokens.intersection(context_tokens)
        coverage = len(matched) / len(answer_tokens)
        
        return coverage
    
    def _compute_qa_relevance(self, query: str, answer: str) -> float:
        """Compute semantic similarity between question and answer."""
        if not query or not answer:
            return 0.0
        
        # Use semantic similarity task type for Q&A comparison
        embeddings = self.embedding_model.encode(
            [query, answer], 
            convert_to_numpy=True,
            task_type="semantic_similarity"
        )
        
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Answers shouldn't be too similar to questions (that would be repetition)
        # Optimal is moderate similarity (0.3-0.7)
        if similarity > 0.7:
            return 0.7 + (1 - similarity) * 0.5  # Penalize too similar
        elif similarity < 0.2:
            return similarity * 2  # Penalize too dissimilar
        else:
            return min(1.0, similarity * 1.3)  # Reward moderate similarity
    
    def get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= settings.high_confidence_threshold:
            return "high"
        elif score >= settings.low_confidence_threshold:
            return "medium"
        else:
            return "low"


# Singleton instance
_confidence_scorer = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create confidence scorer singleton."""
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer
