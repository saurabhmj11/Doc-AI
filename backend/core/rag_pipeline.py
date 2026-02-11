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
        # STRUCTURED EXTRACTION ROUTING (CRITICAL FIX)
        # =====================================================

        # Check for structured keywords
        keywords = ["shipper", "consignee", "rate", "weight", "date", "load", "shipment"]
        is_structured_query = any(k in question.lower() for k in keywords)

        if is_structured_query:
            # Try fast path first
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

    # =====================================================
    # STRUCTURED FAST PATH (FIXED VERSION)
    # =====================================================

    def _extract_pickup_block(self, context):
        # Extract pickup block safely using structural anchors
        match = re.search(
            r'Pickup\s*\n(.+?)(?=\n\s*Delivery|\n\s*Drop|\n\s*Consignee|\Z)',
            context,
            re.S | re.I
        )

        if not match:
            return None

        block = match.group(1).strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]

        # Filter out common noise
        cleaned_lines = []
        for line in lines:
            if any(x in line for x in ["S.No", "Commodity", "Weight", "Quantity", "Shipping Date"]):
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)

        return None

    def _extract_delivery_block(self, context):
        # Extract delivery block safely
        match = re.search(
            r'(?:Delivery|Drop)\s*\n(.+?)(?=\n\s*Rate|\n\s*Total|\n\s*Notes|\Z)',
            context,
            re.S | re.I
        )

        if not match:
            return None

        block = match.group(1).strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]

        cleaned_lines = []
        for line in lines:
            if any(x in line for x in ["S.No", "Commodity", "Weight", "Quantity", "Delivery Date"]):
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)
            
        return None

    def _structured_fast_path(self, question, context):

        # Remove source labels to prevent pollution
        context = re.sub(r'\[Source.*?\]:', '', context)
        q = question.lower()

        # 1. SHIPPER / ORIGIN
        if any(k in q for k in ["shipper", "sender", "from", "origin"]):
            val = self._extract_pickup_block(context)
            if val and len(val) < 200: # Length guard
                return f"**Shipper:**\n{val}\n\n✅ Instant answer"

        # 2. CONSIGNEE / DESTINATION
        if any(k in q for k in ["consignee", "receiver", "to", "destination"]):
            val = self._extract_delivery_block(context)
            if val and len(val) < 200:
                return f"**Consignee:**\n{val}\n\n✅ Instant answer"

        # 3. GENERIC REGEX FALLBACK FOR OTHER FIELDS
        patterns = {
            "load id": r'(?:Load|Shipment|BOL)\s*(?:ID|#)?[:\s]*([A-Z0-9-]+)',
            "rate": r'(?:Rate|Total|Amount|Charges)[:\s]*(\$?\s*[\d,]+\.?\d*)',
            "weight": r'(?:Weight|Gross)\s*[:\s]*([\d,]+\s*(?:lbs|kg)?)',
            "date": r'(?:Date|Pickup Date|Delivery Date)\s*[:\s]*(\d{2}[-/]\d{2}[-/]\d{4})'
        }

        for key, pattern in patterns.items():
            if key in q:
                match = re.search(pattern, context, re.I)
                if match:
                    val = match.group(1).strip()
                    return f"**{key.title()}:**\n{val}\n\n✅ Instant answer"

        return None

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
        
        # Clean context to just text
        clean_text = re.sub(r'\[Source.*?\]:', '', context).strip()
        
        # Split by double newline to get first paragraph/header block
        first_block = clean_text.split("\n\n")[0]
        
        # Strict length limit to avoid dumping entire doc
        if len(first_block) > 300:
            first_block = first_block[:300] + "..."
            
        return f"Based on document:\n{first_block}"

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
