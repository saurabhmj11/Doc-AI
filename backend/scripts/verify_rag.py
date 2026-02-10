import sys
import os
import asyncio
from pathlib import Path

# Add backend to python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.config import get_settings
from backend.core.rag_pipeline import RAGPipeline
from backend.core.document_processor import DocumentChunk

async def main():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    # --- TEST LLM CONNECTION DIRECTLY ---
    print("\n--- TESTING LLM CONNECTION ---")
    try:
        response = rag.llm.generate_content("Hello, can you hear me?")
        print(f"LLM Response: {response.text}")
        print("--- LLM CONNECTION OK ---\n")
    except Exception as e:
        print(f"--- LLM CONNECTION FAILED: {e} ---")
        return
    
    doc_id = "test_doc_manual"
    print("Creating manual chunk...")
    chunk = DocumentChunk(
        chunk_id="1", 
        document_id=doc_id, 
        text="The shipper is Global Electronics Inc. Located at 123 Tech Park.", 
        page=1, 
        chunk_index=0
    )
    
    print("Embedding chunk...")
    # chunks = rag.embedding_model.embed_chunks([chunk])
    # Fix: use encode directly
    embeddings = rag.embedding_model.encode([chunk.text], convert_to_numpy=True)
    chunk.embedding = embeddings[0].tolist()
    chunks = [chunk]
    
    print("Adding to Vector Store...")
    from backend.core.vector_store import get_vector_store
    vs = get_vector_store()
    vs.add_document(doc_id, chunks, "test.txt", "txt", "test.txt")
    
    print("Asking 'Who is the shipper?'...")
    res = rag.ask([doc_id], "Who is the shipper?")
    
    print("\n--- RESULT ---")
    print("Answer:", res['answer'])
    print("Confidence:", res['confidence'])
    print("Status:", res.get('guardrail_status'))

if __name__ == "__main__":
    asyncio.run(main())
