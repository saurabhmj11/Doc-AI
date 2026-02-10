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

async def main():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    # Mock context with some sample text that regex can parse
    mock_context = """
    [Source 1 - mocked_doc.pdf, Page 1]:
    BOL# 123456789
    Shipper: ACME Logistics Inc., 123 Main St, Springfield IL
    Consignee: Big Corp Warehouse, 456 Industry Way, Chicago IL
    Rate: $1,500.00
    Pickup: 10/12/2025 08:00 AM
    Delivery: 10/13/2025 02:00 PM
    Carrier: FastTrack Transport
    """
    
    questions = [
        "Who is the shipper?",
        "What is the consignee?",
        "What is the rate?",
        "When is pickup?",
        "Who is the carrier?"
    ]
    
    print("\n--- Testing Structured Fallback ---")
    
    for q in questions:
        print(f"\nQuestion: {q}")
        # Directly call _extractive_fallback to simulate failure/rate limit
        fallback_response = rag._extractive_fallback(q, mock_context)
        print(f"Fallback Response:\n{fallback_response}")
        
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
