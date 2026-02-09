
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
from backend.core.document_processor import DocumentProcessor
from backend.core.rag_pipeline import RAGPipeline
from backend.core.vector_store import get_vector_store

async def main():
    print("Initializing components...")
    
    try:
        processor = DocumentProcessor()
        rag = RAGPipeline()
        vector_store = get_vector_store()
    except Exception as e:
        print(f"Error initializing: {e}")
        return
    
    # Files to process
    files = [
        "sample_docs/BOL53657_billoflading.pdf",
    ]
    
    doc_ids = []

    for fpath in files:
        full_path = Path(fpath).resolve()
        if not full_path.exists():
            print(f"Skipping {fpath} (not found)")
            continue
            
        print(f"\nProcessing {full_path.name}...")
        try:
            # Note: DocumentProcessor uses model_loader which uses "models/gemini-embedding-001" now
            doc_id, chunks = processor.process_file(str(full_path), "pdf")
            print(f" - Generated {len(chunks)} chunks.")
            if chunks and chunks[0].embedding:
                print(f" - Sample embedding: {chunks[0].embedding[:5]}...")
            else:
                print(" - WARNING: Chunks have no embeddings or are empty.")
            
            # Add to Vector Store
            vector_store.add_document(
                document_id=doc_id,
                chunks=chunks,
                filename=full_path.name,
                file_type="pdf",
                file_path=str(full_path)
            )
            print(f" - Added to Vector Store as {doc_id}")
            doc_ids.append(doc_id)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            import traceback
            traceback.print_exc()

    if not doc_ids:
        print("No documents processed. Exiting.")
        return

    # Ask Question
    question = "Who is the shipper?"

    if doc_ids:
        # Debug Embeddings
        # Debug Embeddings
        import numpy as np
        emb = chunks[0].embedding
        norm = np.linalg.norm(emb)
        print(f" - Embedding Norm: {norm:.4f}")
        
        # Debug Search Directly
        print("\n--- Debug Search ---")
        q_emb = rag.embedding_model.encode(question, convert_to_numpy=True)
        results = vector_store.search(doc_ids[0], q_emb, top_k=3)
        print(f"Direct Search Results: {results}")
        if results:
             print(f"Top 3 Scores: {[r['similarity_score'] for r in results]}")
        else:
             print("Direct Search returned NO results.")

    # Ask Question
    question = "Who is the shipper?"
    print(f"\n\nAsking: '{question}' across {len(doc_ids)} documents...")
    
    try:
        # Use our updated RAG pipeline
        result = rag.ask(document_ids=doc_ids, question=question)
        
        print("\n--- RAG Result ---")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']*100:.1f}% ({result['confidence_level']})")
        print(f"Status: {result.get('guardrail_status', 'unknown')}")
        
    except Exception as e:
        print(f"Error asking question: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
