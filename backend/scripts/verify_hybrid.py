import sys
import os
import asyncio
import logging
from pathlib import Path

# Add backend to python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.config import get_settings
from backend.core.rag_pipeline import RAGPipeline
from backend.core.document_processor import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    # 1. Setup Document with Tricky Text
    doc_id = "hybrid_test_doc"
    tricky_text = """
    Bill of Lading 9245 Laguna Springs Dr., Suite#200, Elk Grove, CA, 95758-7991 
    Mailing Address: 1250 Broadway, New York, New York, 10001 
    Phone: (844) 850-3391 Fax: 916-209-6669 
    After hour Contact: usdev@ultraship.ai 
    Load ID LD53657 Ship Date 02-08-2026 09:00 Delivery Date 02-08-2026 09:00 PO Number Pickup 112233ABC 
    Freight Charges Collect COD Prepaid Shipper Consignee 
    1. AAA , Los Angeles International Airport (LAX), World Way, Los Angeles, CA, USA 
    1. (Note: This is a direct excerpt from the document. For a more detailed answer, please try again when the API quota resets.)
    """
    
    print("\n--- Creating Manual Chunk ---")
    chunk = DocumentChunk(
        chunk_id="h1", 
        document_id=doc_id, 
        text=tricky_text, 
        page=1, 
        chunk_index=0
    )
    
    try:
        print("Embedding chunk (requires API)...")
        # We need this to work for retrieval to work
        embeddings = rag.embedding_model.encode([chunk.text], convert_to_numpy=True)
        chunk.embedding = embeddings[0].tolist()
        chunks = [chunk]
        
        print("Adding to Vector Store...")
        rag.vector_store.add_document(doc_id, chunks, "hybrid_test.pdf", "pdf", "hybrid_test.pdf")
        
        # 2. Test Fast Path (Normal Operation)
        print("\n--- TEST 1: Fast Path Check ---")
        question = "Who is the shipper?"
        result = rag.ask([doc_id], question)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        if "Instant answer" in str(result['answer']):
            print("✅ Fast Path Triggered Successfully!")
        else:
            print("❌ Fast Path NOT Triggered!")

        # 3. Test Offline/Failure Fallback
        print("\n--- TEST 2: LLM Failure Fallback ---")
        # Simulate LLM failure by invalidating the model
        original_llm = rag.llm
        rag.llm = None # Force "Gemini API key not configured" path which logs warning and should hit structured logic
        
        # Start a different question that might NOT be in fast path patterns directly 
        # BUT "Shipper" is, so let's try another one.
        # Actually user said: "Works when Gemini quota finished ... Structured field answers ... always correct"
        # So we test the same question to see if it works WITHOUT LLM instance
        
        result_fail = rag.ask([doc_id], question)
        
        print(f"Answer (No LLM): {result_fail['answer']}")
        if "Fast Structured Match" in str(result_fail['answer']):
             print("✅ Fast Path Fallback Triggered Successfully!")
        elif "Shipper" in result_fail['answer'] and "AAA" in result_fail['answer']:
             print("✅ Structured Fallback Triggered!")
        else:
             print("❌ Fallback Failed!")
             
        # Restore logic if needed (not needed since script ends)
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
