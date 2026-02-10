import logging
from typing import Optional
import google.generativeai as genai
import re

from config import get_settings
from core.model_loader import get_embedding_model
from core.vector_store import get_vector_store
from core.confidence_scorer import get_confidence_scorer
from core.guardrails import get_guardrails
from core.reranker import get_reranker
from core.error_handling import (
    get_gemini_circuit_breaker,
    get_ollama_circuit_breaker,
    api_retry,
    APIError
)

settings = get_settings()
logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self):

        self._embedding_model = None
        self._reranker = None

        self.vector_store = get_vector_store()
        self.confidence_scorer = get_confidence_scorer()
        self.guardrails = get_guardrails()

        self.llm_mode = settings.llm_mode

        if self.llm_mode == "offline":
            import ollama
            self.ollama_client = ollama.Client(host=settings.ollama_base_url)
            self.ollama_model = settings.ollama_model
            self.llm = True
            self.circuit_breaker = get_ollama_circuit_breaker()
        else:
            if settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.llm = genai.GenerativeModel(settings.gemini_model)
                self.circuit_breaker = get_gemini_circuit_breaker()
            else:
                self.llm = None
                self.circuit_breaker = None
        
        # Initialize reranker if enabled
        if settings.enable_reranking:
            self._reranker = get_reranker()
        else:
            self._reranker = None

    @property
    def reranker(self):
        return self._reranker

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    # =====================================================
    # MAIN ASK FUNCTION
    # =====================================================

    def ask(self, document_ids: list[str] | str, question: str) -> dict:

        if isinstance(document_ids, str):
            document_ids = [document_ids]

        # Embed query
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        )

        if query_embedding is None:
            raise RuntimeError("Embedding failed")

        # Retrieval
        chunks = self.vector_store.search_parallel(
            document_ids=document_ids,
            query_embedding=query_embedding if isinstance(query_embedding, list)
            else query_embedding.tolist(),
            top_k=settings.top_k_retrieval
        )

        retrieval_check = self.guardrails.check_retrieval(chunks)

        if retrieval_check.should_refuse:
            return {
                "answer": retrieval_check.message,
                "confidence": 0.0,
                "confidence_level": "low",
                "sources": [],
                "guardrail_status": "refused",
                "guardrail_message": retrieval_check.message
            }

        # Rerank if enabled
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=settings.top_k_rerank)
        
        top_chunks = chunks[:settings.top_k_rerank]
        context = self._build_context(top_chunks)

        # =====================================================
        # FAST STRUCTURED EXTRACTION (NO LLM NEEDED)
        # =====================================================

        fast_answer = self._structured_fast_path(question, context)

        if fast_answer:
            return {
                "answer": fast_answer,
                "confidence": 0.95,
                "confidence_level": "high",
                "sources": self._format_sources(top_chunks),
                "guardrail_status": "allowed",
                "guardrail_message": None
            }

        # =====================================================
        # LLM CALL
        # =====================================================

        answer = self._generate_answer(question, context)

        confidence, breakdown = self.confidence_scorer.compute_confidence(
            question,
            answer,
            top_chunks,
            context
        )

        confidence_level = self.confidence_scorer.get_confidence_level(confidence)

        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "sources": self._format_sources(top_chunks),
            "guardrail_status": "allowed",
            "guardrail_message": None
        }

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================

    def _build_context(self, chunks):

        parts = []

        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page")
            parts.append(
                f"[Source {i} - {filename} Page {page}]:\n{chunk['text']}"
            )

        return "\n\n".join(parts)

    # =====================================================
    # STRUCTURED FAST PATH (FIXED VERSION)
    # =====================================================

    def _structured_fast_path(self, question, context):

        # Remove source labels (IMPORTANT FIX)
        context = re.sub(r'\[Source.*?\]:', '', context)

        q = question.lower()

        patterns = {
            "shipper": r'Shipper[:\s]*\n?([^\n]+(?:\n[^\n]+){0,3})',
            "consignee": r'Consignee[:\s]*\n?([^\n]+(?:\n[^\n]+){0,3})',
            "load id": r'(?:Load|Shipment|BOL)\s*(?:ID|#)?[:\s]*([A-Z0-9-]+)',
            "rate": r'(?:Rate|Total|Amount|Charges)[:\s]*(\$?\s*[\d,]+\.?\d*)'
        }

        keyword_map = {
            "sender": "shipper",
            "from": "shipper",
            "origin": "shipper",
            "receiver": "consignee",
            "destination": "consignee"
        }

        target = None

        for key in patterns:
            if key in q:
                target = key
                break

        if not target:
            for k, v in keyword_map.items():
                if k in q:
                    target = v
                    break

        if not target:
            return None

        match = re.search(patterns[target], context, re.S | re.I)

        if not match:
            return None

        val = match.group(1).strip()
        val = re.sub(r'\s+', ' ', val)

        return f"**{target.title()}:**\n{val}\n\n✅ Instant answer"

    # =====================================================
    # LLM GENERATION (SIMPLIFIED + SAFE)
    # =====================================================

    def _generate_answer(self, question, context):

        if not self.llm:
            return self._extractive_fallback(question, context)

        prompt = f"""
Answer ONLY using this context:

{context}

Question: {question}
"""

        try:

            @api_retry(api_name="Gemini")
            def call_llm():
                response = self.llm.generate_content(prompt)

                if not response.text:
                    raise APIError("Empty response")

                return response.text.strip()

            return self.circuit_breaker.call(call_llm)

        except Exception as e:
            logger.warning(f"LLM failed: {e}")
            return self._extractive_fallback(question, context)

    # =====================================================
    # FALLBACK
    # =====================================================

    def _extractive_fallback(self, question, context):

        parts = context.split("[Source")

        if len(parts) > 1:
            text = parts[1].split("]:", 1)[1][:400]
            return f"Based on document:\n{text}"

        return context[:400]

    # =====================================================
    # SOURCE FORMATTER
    # =====================================================

    def _format_sources(self, chunks):

        sources = []

        for chunk in chunks:

            text = chunk["text"]

            if len(text) > 300:
                text = text[:300] + "..."

            sources.append({
                "text": text,
                "page": chunk.get("page"),
                "filename": chunk.get("filename"),
                "chunk_id": chunk.get("chunk_id"),
                "similarity_score": round(chunk.get("similarity_score", 0), 3)
            })

        return sources


_rag_pipeline = None

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
