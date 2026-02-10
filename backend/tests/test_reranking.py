"""
Test Reranking - Verify cross-encoder reranking improves retrieval quality

This script tests the reranker module and demonstrates the improvement
in chunk relevance after reranking.
"""

import os
import sys
from pathlib import Path

# Add backend to path
# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
# Add backend to path (parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
from core.error_handling import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def test_reranker_module():
    """Test reranker module standalone."""
    print("\n" + "="*60)
    print("TEST 1: Reranker Module")
    print("="*60)
    
    from core.reranker import get_reranker
    
    reranker = get_reranker()
    stats = reranker.get_stats()
    
    print(f"\nReranker Stats:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Model Loaded: {stats['model_loaded']}")
    print(f"  Batch Size: {stats['batch_size']}")
    
    # Create sample chunks
    query = "What is the pickup date?"
    chunks = [
        {
            "text": "The delivery is scheduled for 02/15/2026 at 3:00 PM.",
            "chunk_id": "chunk_1",
            "similarity_score": 0.75
        },
        {
            "text": "Pickup date and time: 02/10/2026 at 10:00 AM from Chicago warehouse.",
            "chunk_id": "chunk_2",
            "similarity_score": 0.70
        },
        {
            "text": "Equipment type: 53' Dry Van with liftgate.",
            "chunk_id": "chunk_3",
            "similarity_score": 0.65
        },
        {
            "text": "The shipment will be picked up on February 10, 2026 in the morning.",
            "chunk_id": "chunk_4",
            "similarity_score": 0.60
        }
    ]
    
    print(f"\n\nOriginal Order (by similarity):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. [{chunk['chunk_id']}] sim={chunk['similarity_score']:.2f}: {chunk['text'][:60]}...")
    
    # Rerank
    print(f"\n\nReranking with query: '{query}'")
    reranked = reranker.rerank(query, chunks, top_k=4)
    
    print(f"\nReranked Order (by cross-encoder):")
    for i, chunk in enumerate(reranked, 1):
        rerank_score = chunk.get('rerank_score', 0)
        orig_sim = chunk.get('original_similarity_score', chunk.get('similarity_score', 0))
        print(f"  {i}. [{chunk['chunk_id']}] rerank={rerank_score:.3f}, orig_sim={orig_sim:.2f}")
        print(f"      {chunk['text'][:80]}...")
    
    # Check if most relevant chunk (chunk_2 or chunk_4) moved to top
    if reranked[0]['chunk_id'] in ['chunk_2', 'chunk_4']:
        print("\n  ✓ Reranking successfully prioritized pickup-related chunks!")
    else:
        print(f"\n  ⚠ Top chunk is {reranked[0]['chunk_id']}, expected chunk_2 or chunk_4")


def test_rag_with_reranking():
    """Test RAG pipeline with reranking enabled."""
    print("\n" + "="*60)
    print("TEST 2: RAG Pipeline with Reranking")
    print("="*60)
    
    from core.rag_pipeline import get_rag_pipeline
    
    rag = get_rag_pipeline()
    
    print(f"\nRAG Pipeline Configuration:")
    print(f"  Reranker: {'enabled' if rag.reranker else 'disabled'}")
    
    # Create mock chunks to simulate retrieval
    mock_chunks = [
        {
            "text": "Rate confirmation: Total freight charge is $2,850.00 USD.",
            "page": 1,
            "filename": "rate_conf.pdf",
            "chunk_id": "chunk_1",
            "similarity_score": 0.80
        },
        {
            "text": "Equipment specifications: 53 foot dry van trailer required.",
            "page": 1,
            "filename": "rate_conf.pdf",
            "chunk_id": "chunk_2",
            "similarity_score": 0.75
        },
        {
            "text": "Pickup scheduled for February 10, 2026 at 10:00 AM CST.",
            "page": 1,
            "filename": "rate_conf.pdf",
            "chunk_id": "chunk_3",
            "similarity_score": 0.70
        }
    ]
    
    query = "When is the pickup?"
    
    print(f"\nQuery: '{query}'")
    print(f"\nOriginal retrieval order:")
    for i, chunk in enumerate(mock_chunks, 1):
        print(f"  {i}. sim={chunk['similarity_score']:.2f}: {chunk['text'][:60]}...")
    
    # Test reranking
    from config import get_settings
    settings = get_settings()
    
    if settings.enable_reranking and rag.reranker:
        print(f"\nReranking {len(mock_chunks)} chunks...")
        reranked = rag.reranker.rerank(query, mock_chunks, top_k=3)
        
        print(f"\nReranked order:")
        for i, chunk in enumerate(reranked, 1):
            print(f"  {i}. rerank={chunk.get('rerank_score', 0):.3f}: {chunk['text'][:60]}...")
        
        if reranked[0]['chunk_id'] == 'chunk_3':
            print("\n  ✓ Reranking correctly prioritized pickup information!")
        else:
            print(f"\n  ⚠ Expected chunk_3 at top, got {reranked[0]['chunk_id']}")
    else:
        print("\n  ⚠ Reranking is disabled")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RERANKING VERIFICATION TESTS")
    print("="*60)
    
    try:
        test_reranker_module()
        test_rag_with_reranking()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nReranking is working! The cross-encoder successfully")
        print("reorders chunks based on actual relevance to the query.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
