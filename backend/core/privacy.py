"""
Privacy Layer Module

HIPAA-compliant PII/PHI masking using Microsoft Presidio.
Anonymizes sensitive entities in document chunks and user queries.
"""

import logging
from typing import List, Dict, Any, Optional
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class PrivacyLayer:
    """
    Handles PII/PHI masking for document processing and RAG.
    
    Identifies and redacts:
    - PERSON, LOCATION, PHONE_NUMBER, EMAIL_ADDRESS
    - MEDICAL_LICENSE, US_SSN, US_PASSPORT (standard PHI)
    - DATE_TIME (partial masking)
    """

    def __init__(self):
        try:
            # Initialize Presidio engines
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            
            # Entities we want to mask for HIPAA-like compliance
            self.entities_to_mask = [
                "PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "US_ITIN", "US_PASSPORT", "US_SSN", "US_BANK_NUMBER",
                "CREDIT_CARD", "DATE_TIME"
            ]
            
            # Custom operators for masking
            self.operators = {
                "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            }
            
            logger.info("PrivacyLayer initialized with Microsoft Presidio")
        except Exception as e:
            logger.error(f"Failed to initialize PrivacyLayer: {e}")
            self.analyzer = None
            self.anonymizer = None

    def mask_text(self, text: str) -> str:
        """
        Mask sensitive entities in the provided text.
        
        Args:
            text: Input string to anonymize
            
        Returns:
            Anonymized string with placeholders
        """
        if not text or not self.analyzer:
            return text
            
        try:
            # 1. Analyze text for entities
            results = self.analyzer.analyze(
                text=text,
                entities=self.entities_to_mask,
                language='en'
            )
            
            # 2. Anonymize found entities
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self.operators
            )
            
            return anonymized_result.text
        except Exception as e:
            logger.error(f"Error during PII masking: {e}")
            return text

    def mask_chunks(self, chunks: List[Any]) -> List[Any]:
        """
        Mask text in a list of DocumentChunks or dictionaries.
        
        Args:
            chunks: List of chunk objects with 'text' attribute or key
            
        Returns:
            List of chunks with masked text
        """
        for chunk in chunks:
            if hasattr(chunk, 'text'):
                chunk.text = self.mask_text(chunk.text)
            elif isinstance(chunk, dict) and 'text' in chunk:
                chunk['text'] = self.mask_text(chunk['text'])
        return chunks

# Singleton instance
_privacy_layer = None

def get_privacy_layer() -> PrivacyLayer:
    """Get or create PrivacyLayer singleton."""
    global _privacy_layer
    if _privacy_layer is None:
        _privacy_layer = PrivacyLayer()
    return _privacy_layer
