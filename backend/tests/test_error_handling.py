"""
Test Error Handling - Manual verification of API error handling

This script simulates various API failure scenarios to verify
the error handling implementation.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
# Add backend to path (parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
from core.error_handling import setup_logging, CircuitBreaker, APIError

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def test_circuit_breaker():
    """Test circuit breaker pattern."""
    print("\n" + "="*60)
    print("TEST 1: Circuit Breaker Pattern")
    print("="*60)
    
    cb = CircuitBreaker(failure_threshold=3, timeout=5, name="test_cb")
    
    def failing_function():
        raise Exception("Simulated API failure")
    
    def working_function():
        return "Success!"
    
    # Test failures
    print("\n1. Testing failures to open circuit...")
    for i in range(5):
        try:
            result = cb.call(failing_function)
        except Exception as e:
            print(f"   Attempt {i+1}: {type(e).__name__}: {str(e)[:50]}")
    
    # Test that circuit is open
    print("\n2. Verifying circuit is OPEN...")
    try:
        cb.call(working_function)
        print("   ❌ FAILED: Circuit should be open!")
    except APIError as e:
        print(f"   ✓ Circuit is OPEN: {str(e)[:60]}")
    
    # Test recovery
    print("\n3. Waiting for timeout and testing recovery...")
    import time
    time.sleep(6)  # Wait for timeout
    
    try:
        result = cb.call(working_function)
        print(f"   ✓ Circuit recovered: {result}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print(f"\n   Final state: {cb.state.value}")


def test_rag_pipeline_fallback():
    """Test RAG pipeline with API failure."""
    print("\n" + "="*60)
    print("TEST 2: RAG Pipeline Extractive Fallback")
    print("="*60)
    
    from core.rag_pipeline import RAGPipeline
    
    # Create mock chunks
    mock_chunks = [
        {
            "text": "The shipment will be picked up on 02/10/2026 at 10:00 AM from Chicago, IL.",
            "page": 1,
            "filename": "test_doc.pdf",
            "chunk_id": "test_1",
            "similarity_score": 0.85
        }
    ]
    
    rag = RAGPipeline()
    
    # Build context
    context = rag._build_context(mock_chunks)
    
    # Test extractive fallback
    print("\n1. Testing extractive fallback method...")
    result = rag._extractive_fallback("When is pickup?", context)
    print(f"\n   Fallback Result:\n   {result[:200]}...")
    
    if "02/10/2026" in result or "Chicago" in result:
        print("\n   ✓ Extractive fallback is working correctly")
    else:
        print("\n   ❌ Fallback might not be extracting correctly")


def test_structured_extractor_fallback():
    """Test structured extractor with regex fallback."""
    print("\n" + "="*60)
    print("TEST 3: Structured Extractor Regex Fallback")
    print("="*60)
    
    from core.structured_extractor import StructuredExtractor
    
    # Sample document text
    sample_doc = """
    RATE CONFIRMATION
    
    Load #: RC-2026-001
    BOL #: BOL123456
    
    Shipper: ABC Logistics Inc, 123 Main St, Chicago, IL 60601
    Consignee: XYZ Distribution Center, 456 Oak Ave, Dallas, TX 75201
    
    Pickup Date: 02/10/2026 10:00 AM
    Delivery Date: 02/12/2026 2:00 PM
    
    Equipment: 53' Dry Van
    Mode: TL
    Weight: 42,000 lbs
    Rate: $2,850.00 USD
    
    Carrier: Premium Freight Lines
    """
    
    extractor = StructuredExtractor()
    
    print("\n1. Testing regex-based extraction...")
    shipment, confidence = extractor._fallback_extraction(sample_doc)
    
    print(f"\n   Extracted Data:")
    print(f"   - Shipment ID: {shipment.shipment_id}")
    print(f"   - Shipper: {shipment.shipper}")
    print(f"   - Consignee: {shipment.consignee}")
    print(f"   - Pickup: {shipment.pickup_datetime}")
    print(f"   - Delivery: {shipment.delivery_datetime}")
    print(f"   - Equipment: {shipment.equipment_type}")
    print(f"   - Mode: {shipment.mode}")
    print(f"   - Weight: {shipment.weight}")
    print(f"   - Rate: ${shipment.rate} {shipment.currency}")
    print(f"   - Carrier: {shipment.carrier_name}")
    print(f"   - Confidence: {confidence:.2f}")
    
    # Count extracted fields
    fields = [
        shipment.shipment_id, shipment.shipper, shipment.consignee,
        shipment.pickup_datetime, shipment.delivery_datetime,
        shipment.equipment_type, shipment.mode, shipment.weight,
        shipment.rate, shipment.carrier_name
    ]
    extracted_count = sum(1 for f in fields if f is not None)
    
    print(f"\n   Extracted {extracted_count}/10 fields")
    
    if extracted_count >= 7:
        print("   ✓ Regex extraction is working well")
    else:
        print("   ⚠ Some fields might not be extracted correctly")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ERROR HANDLING VERIFICATION TESTS")
    print("="*60)
    
    try:
        test_circuit_breaker()
        test_rag_pipeline_fallback()
        test_structured_extractor_fallback()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nCheck the logs at: logs/api_errors.log")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
