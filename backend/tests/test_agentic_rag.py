"""
Test Agentic RAG (RAG 2.0) Flow
Verifies Query Planner -> Dual Retrieval -> Grounding Graph -> Answer
"""

import sys
import os
import asyncio
from pathlib import Path

# Add backend to python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "backend"))

from dotenv import load_dotenv
load_dotenv("backend/.env")

from core.document_processor import DocumentProcessor
from core.rag_pipeline import RAGPipeline, QueryIntent
from core.vector_store import get_vector_store
import logging

# Configure logging to see our new logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_agentic_rag")

async def main():
    print("Initializing Agentic RAG components...")
    
    try:
        processor = DocumentProcessor()
        rag = RAGPipeline()
        vector_store = get_vector_store()
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Check components
    print(f"Planner initialized: {rag.planner is not None}")
    print(f"Extractor initialized: {rag.extractor is not None}")
    
    # Process a file
    fpath = "sample_docs/bill_of_lading_sample.txt"
    full_path = Path(fpath).resolve()
    
    if not full_path.exists():
        print(f"Error: {fpath} not found")
        return

    print(f"\nProcessing {full_path.name}...")
    doc_id, chunks = processor.process_file(str(full_path), "txt")
    vector_store.add_document(doc_id, chunks, full_path.name, "txt", str(full_path))
    
    # Test 1: ENTITY_LOOKUP Intent
    q1 = "Who is the shipper?"
    print(f"\n--- Test Case 1: '{q1}' ---")
    
    # Verify Planner directly first
    plan = rag.planner.plan(q1)
    print(f"Direct Plan Check: Intent={plan.intent}, Entities={plan.target_entities}")
    if plan.intent == QueryIntent.ENTITY_LOOKUP:
        print("PASS: Planner correctly identified ENTITY_LOOKUP")
    else:
        print(f"FAIL: Planner failed: Got {plan.intent}")

    # Verify RAG Flow
    response = rag.ask(doc_id, q1)
    print(f"Answer: {response['answer']}")
    print(f"Sources: {len(response['sources'])}")
    
    # Test 2: GENERAL Intent (Should NOT trigger extraction)
    q2 = "Summarize this document."
    print(f"\n--- Test Case 2: '{q2}' ---")
    
    # Verify Planner
    plan = rag.planner.plan(q2)
    print(f"Direct Plan Check: Intent={plan.intent}")
    
    response = rag.ask(doc_id, q2)
    print(f"Answer: {response['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
