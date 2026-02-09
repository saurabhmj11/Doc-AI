"""
RAG Pipeline - the heart of the Q&A system

This is where the magic happens - takes a question, finds relevant 
chunks from the doc, and asks Gemini to answer based on what we found.

I added some fallbacks in case the LLM is down or unavailable.
"""

import os
from typing import Optional
import google.generativeai as genai

from config import get_settings
from core.model_loader import get_embedding_model
from core.vector_store import get_vector_store
from core.confidence_scorer import get_confidence_scorer
from core.guardrails import get_guardrails, GuardrailResult

settings = get_settings()


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
        self.vector_store = get_vector_store()
        self.confidence_scorer = get_confidence_scorer()
        self.guardrails = get_guardrails()
        
        # set up gemini if we have a key
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.llm = None
    
    @property
    def embedding_model(self):
        """Lazy-load embedding model on first use."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model
    
    def ask(self, document_ids: list[str] | str, question: str) -> dict:
        """
        Main entry point - ask a question about one or more docs.
        Returns answer with confidence score and source citations.
        """
        # handle single doc ID for backward compatibility
        if isinstance(document_ids, str):
            document_ids = [document_ids]
            
        # embed the question for higher relevance using correct task type
        query_embedding = self.embedding_model.encode(
            question, 
            convert_to_numpy=True,
            task_type="retrieval_query"
        )
        
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
        
        # build context from best chunks
        top_chunks = chunks[:settings.top_k_rerank]
        context = self._build_context(top_chunks)
        
        # ask the LLM
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
            "guardrail_message": guardrail_result.message
        }
    
    def _build_context(self, chunks: list[dict]) -> str:
        """Stitch chunks together into a context string with source attribution"""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', 'Unknown Document')
            page = f", Page {chunk['page']}" if chunk.get('page') else ""
            parts.append(f"[Source {i} - {filename}{page}]:\n{chunk['text']}")
        return "\n\n".join(parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Call the LLM with our context. Returns specific error if fails."""
        if not self.llm:
            return "Error: Gemini API key not configured. Check backend logs."
        
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
            # Use generation config
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            if not response.text:
                return "The answer was blocked or safety filtered."
                
            return response.text.strip()
            
        except Exception as e:
            # Return the actual error to debug!
            return f"Error from Gemini API: {str(e)}"
    
    def _extractive_fallback(self, question: str, context: str) -> str:
        """Backup plan - just return the top chunk text"""
        if context:
            parts = context.split("[Source")
            if len(parts) > 1:
                first = parts[1].split("]:", 1)
                if len(first) > 1:
                    return first[1].strip()[:500] + "..."
        return "Unable to generate answer. Please check the document directly."
    
    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        """Format source citations for the response"""
        sources = []
        for chunk in chunks:
            text = chunk["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            sources.append({
                "text": text,
                "page": chunk.get("page"),
                "filename": chunk.get("filename"),
                "chunk_id": chunk["chunk_id"],
                "similarity_score": round(chunk.get("similarity_score", 0), 3)
            })
        return sources


# keep one instance around
_rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
