"""
RAG Pipeline - the heart of the Q&A system

This is where the magic happens - takes a question, finds relevant 
chunks from the doc, and asks Gemini to answer based on what we found.

I added some fallbacks in case the LLM is down or unavailable.
"""

import os
import logging
from typing import Optional
import google.generativeai as genai

from config import get_settings
from core.model_loader import get_embedding_model
from core.vector_store import get_vector_store
from core.confidence_scorer import get_confidence_scorer
from core.guardrails import get_guardrails, GuardrailResult
from core.error_handling import (
    get_gemini_circuit_breaker,
    get_ollama_circuit_breaker,
    api_retry,
    APIError
)

settings = get_settings()
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG logic. Pretty standard flow:
    query -> embed -> retrieve -> context -> LLM -> answer
    
    Plus confidence scoring and guardrails to catch hallucinations.
    """
    
    # Tried a bunch of prompts, this one works best for logistics docs
    SYSTEM_PROMPT = """You are a document assistant for logistics paperwork. Answer questions using ONLY the provided context.

Rules:
- Only use info from the context below
- If something isn't there, say "This information is not found in the document"
- Don't make stuff up or guess
- Quote exact text when you can
- Be direct and concise

For numbers, dates, and names - use exactly what's in the document."""

    def __init__(self):
        # Lazy load embedding model - don't call get_embedding_model() here
        self._embedding_model = None
        self._reranker = None  # Lazy load reranker
        self.vector_store = get_vector_store()
        self.confidence_scorer = get_confidence_scorer()
        self.guardrails = get_guardrails()
        
        # set up LLM based on mode
        self.llm_mode = settings.llm_mode
        
        if self.llm_mode == "offline":
            import ollama
            self.ollama_client = ollama.Client(host=settings.ollama_base_url)
            self.ollama_model = settings.ollama_model
            self.llm = True # marker that we are ready
            self.circuit_breaker = get_ollama_circuit_breaker()
            logger.info(f"RAG Pipeline initialized in OFFLINE mode (Model: {self.ollama_model})")
        else:
            # set up gemini if we have a key
            if settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.llm = genai.GenerativeModel(settings.gemini_model)
                self.circuit_breaker = get_gemini_circuit_breaker()
                logger.info(f"RAG Pipeline initialized in ONLINE mode ({settings.gemini_model})")
            else:
                self.llm = None
                self.circuit_breaker = None
                logger.warning("RAG Pipeline initialized WITHOUT LLM (no API key)")
    
    @property
    def embedding_model(self):
        """Lazy-load embedding model on first use."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model
    
    @property
    def reranker(self):
        """Lazy-load reranker on first use."""
        if self._reranker is None and settings.enable_reranking:
            from core.reranker import get_reranker
            self._reranker = get_reranker()
            logger.info("Reranker loaded for RAG pipeline")
        return self._reranker
    
    def ask(self, document_ids: list[str] | str, question: str) -> dict:
        """
        Main entry point - ask a question about one or more docs.
        Returns answer with confidence score and source citations.
        """
        # handle single doc ID for backward compatibility
        if isinstance(document_ids, str):
            document_ids = [document_ids]
            
        # embed the question (using default task_type for compatibility with indexed chunks)
        query_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        
        # find similar chunks across all documents (parallel search)
        chunks = self.vector_store.search_parallel(
            document_ids=document_ids,
            query_embedding=query_embedding.tolist(),
            top_k=settings.top_k_retrieval
        )
        
        # quick check - did we find anything relevant?
        retrieval_check = self.guardrails.check_retrieval(chunks)
        if retrieval_check.should_refuse:
            return {
                "answer": retrieval_check.message,
                "confidence": 0.0,
                "confidence_level": "low",
                "sources": [],
                "guardrail_status": retrieval_check.status,
                "guardrail_message": retrieval_check.message
            }
        
        # Rerank chunks if enabled
        reranked = False
        if settings.enable_reranking and self.reranker:
            try:
                logger.debug(f"Reranking {len(chunks)} retrieved chunks")
                chunks = self.reranker.rerank(question, chunks, top_k=settings.top_k_rerank)
                reranked = True
                logger.info(f"Reranking completed, using top {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Reranking failed, using original order: {e}")
                reranked = False
        
        # build context from best chunks
        top_chunks = chunks[:settings.top_k_rerank]
        context = self._build_context(top_chunks)
        
        # ask the LLM
        if self.llm_mode == "offline":
            answer = self._generate_answer_ollama(question, context)
        else:
            answer = self._generate_answer(question, context)
        
        # score how confident we are
        confidence, breakdown = self.confidence_scorer.compute_confidence(
            query=question,
            answer=answer,
            retrieved_chunks=top_chunks,
            context=context
        )
        
        # final guardrail checks
        guardrail_result = self.guardrails.run_all_checks(
            chunks=top_chunks,
            answer=answer,
            context=context,
            confidence=confidence,
            confidence_breakdown=breakdown
        )
        
        # package up the response
        if guardrail_result.should_refuse:
            return {
                "answer": guardrail_result.message,
                "confidence": confidence,
                "confidence_level": self.confidence_scorer.get_confidence_level(confidence),
                "sources": self._format_sources(top_chunks),
                "guardrail_status": guardrail_result.status,
                "guardrail_message": guardrail_result.message
            }
        
        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": self.confidence_scorer.get_confidence_level(confidence),
            "sources": self._format_sources(top_chunks),
            "guardrail_status": guardrail_result.status,
            "guardrail_message": guardrail_result.message,
            "reranked": reranked  # Add reranking status
        }
    
    def _build_context(self, chunks: list[dict]) -> str:
        """Stitch chunks together into a context string with source attribution"""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', 'Unknown Document')
            page = f", Page {chunk['page']}" if chunk.get('page') else ""
            parts.append(f"[Source {i} - {filename}{page}]:\n{chunk['text']}")
        return "\n\n".join(parts)
    
    def _generate_answer_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama with retry and circuit breaker."""
        if not self.llm:
            logger.error("Ollama not initialized")
            if settings.enable_extractive_fallback:
                return self._extractive_fallback(question, context)
            return "Service temporarily unavailable. Please try again later."
             
        prompt = f"""You are an AI logistics assistant. Answer the question based ONLY on the context provided.
Context:
{context}

Question: {question}

Answer:"""

        try:
            # Use circuit breaker for API call
            def _call_ollama():
                return self.ollama_client.chat(model=self.ollama_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
            
            response = self.circuit_breaker.call(_call_ollama)
            return response['message']['content']
            
        except APIError as e:
            # Circuit breaker is open
            logger.warning(f"Ollama API circuit breaker open: {str(e)}")
            if settings.enable_extractive_fallback:
                return self._extractive_fallback(question, context)
            return "Service temporarily unavailable due to high error rate. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}", exc_info=True)
            if settings.enable_extractive_fallback:
                logger.info("Using extractive fallback due to Ollama error")
                return self._extractive_fallback(question, context)
            return "An error occurred while processing your question. Please try again."

    def _generate_answer(self, question: str, context: str) -> str:
        """Call Gemini API with retry logic, circuit breaker, and fallback."""
        if not self.llm:
            logger.warning("Gemini API key not configured")
            if settings.enable_extractive_fallback:
                return self._extractive_fallback(question, context)
            return "Service configuration error. Please contact support."
        
        # Improved Prompt
        prompt = f"""You are an AI logistics assistant analyzing documents.
Strictly answer based ONLY on the provided Context below.
If the answer is not in the context, say "I cannot find the answer in the document."
Do NOT repeat the entire document. Be concise.

Context:
{context}

User Question: {question}

Answer:"""
        
        try:
            # Wrapper function for circuit breaker and retry
            @api_retry(api_name="Gemini", max_attempts=settings.retry_attempts)
            def _call_gemini():
                # Use initialized model instance
                print("\n--- DEBUG: CALLING LLM ---")
                response = self.llm.generate_content(prompt)
                print(f"--- DEBUG: RAW RESPONSE: {response.text[:100]}... ---")
                
                if not response.text:
                    logger.warning("Gemini response was blocked or safety filtered")
                    raise APIError("Response blocked by safety filters")
                    
                return response.text.strip()
            
            # Call through circuit breaker
            answer = self.circuit_breaker.call(_call_gemini)
            logger.debug(f"Gemini API call successful for question: {question[:50]}...")
            return answer
            
        except APIError as e:
            # Circuit breaker is open
            logger.warning(f"Gemini API circuit breaker open: {str(e)}")
            if "429" in str(e) or "ResourceExhausted" in str(e):
                return "⚠️ API Rate Limit Exceeded. Please wait a minute before asking again. (Gemini Free Tier Quota)"
            
            if settings.enable_extractive_fallback:
                logger.info("Using extractive fallback due to circuit breaker")
                return self._extractive_fallback(question, context)
            return "Service temporarily unavailable. Please try again in a moment."
            
        except Exception as e:
            # All retries exhausted or other error
            logger.error(f"Gemini API error after retries: {type(e).__name__}: {str(e)}", exc_info=True)
            
            if "429" in str(e) or "quota" in str(e).lower():
                 return "⚠️ API Rate Limit Exceeded. Please wait a minute before asking again. (Gemini Free Tier Quota)"
            
            if settings.enable_extractive_fallback:
                logger.info("Using extractive fallback due to Gemini API error")
                return self._extractive_fallback(question, context)
            
            return "An error occurred while processing your question. Please try again."
    
    def _extractive_fallback(self, question: str, context: str) -> str:
        """Enhanced extractive fallback - uses structured extractor + regex for specific fields."""
        logger.warning(f"FALLBACK TRIGGERED | question={question[:120]}")
        logger.info("Using extractive fallback method")
        
        if not context:
            return "I couldn't find relevant information in the document to answer your question."
            
        # 1. Try structured extraction for specific entity questions
        # This bypasses the LLM completely for common fields
        try:
            from core.structured_extractor import get_structured_extractor
            extractor = get_structured_extractor()
            
            # Map keywords to fields
            q_lower = question.lower()
            field_map = {
                "shipper": "shipper",
                "sender": "shipper",
                "from": "shipper",
                "consignee": "consignee",
                "receiver": "consignee",
                "destination": "consignee",
                "to": "consignee",
                "rate": "rate",
                "cost": "rate", 
                "price": "rate",
                "amount": "rate",
                "total": "rate",
                "shipment id": "shipment_id",
                "load id": "shipment_id",
                "bol": "shipment_id",
                "tracking": "shipment_id",
                "reference": "shipment_id",
                "pickup": "pickup_datetime",
                "delivery": "delivery_datetime",
                "carrier": "carrier_name",
                "trucking": "carrier_name",
                "weight": "weight",
                "equipment": "equipment_type",
                "trailer": "equipment_type"
            }
            
            target_field = None
            for key, field in field_map.items():
                if key in q_lower:
                    target_field = field
                    break
            
            if target_field:
                logger.debug(f"Attempting structured extraction for field: {target_field}")
                # Use offline extraction (Regex only) to avoid LLM issues
                data, confidence = extractor.extract_offline(context)
                
                value = getattr(data, target_field, None)
                if value:
                    return f"Based on the document, the {target_field.replace('_', ' ')} is **{value}**.\n\n(Source: extracted from document text)"
                    
        except Exception as e:
            logger.warning(f"Structured fallback failed: {e}")
        
        # 2. Standard fallback (chunk excerpt)
        # Split context by sources
        parts = context.split("[Source")
        
        if len(parts) > 1:
            # Get the first source (most relevant)
            first_source = parts[1].split("]:", 1)
            if len(first_source) > 1:
                source_info = first_source[0].strip()
                text = first_source[1].strip()
                
                # Limit text length but try to keep complete sentences
                if len(text) > 500:
                    # Find last period within 500 chars
                    truncated = text[:500]
                    last_period = truncated.rfind('.')
                    if last_period > 200:  # At least 200 chars
                        text = truncated[:last_period + 1]
                    else:
                        text = truncated + "..."
                
                return f"""Based on the document, here's the most relevant information I found:

{text}

(Note: This is a direct excerpt from the document. For a more detailed answer, please try again when the API quota resets.)"""
        
        # Fallback if parsing fails
        return "I found some information but couldn't process it properly. Please try again or rephrase your question."
    
    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        """Format source citations for the response"""
        sources = []
        for chunk in chunks:
            text = chunk["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            
            source = {
                "text": text,
                "page": chunk.get("page"),
                "filename": chunk.get("filename"),
                "chunk_id": chunk["chunk_id"],
                "similarity_score": round(chunk.get("similarity_score", 0), 3)
            }
            
            # Add rerank score if available
            if "rerank_score" in chunk:
                source["rerank_score"] = round(chunk["rerank_score"], 3)
            
            sources.append(source)
        return sources


# keep one instance around
_rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
