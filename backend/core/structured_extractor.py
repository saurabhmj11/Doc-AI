"""
Structured Extractor Module

Extract structured shipment data from logistics documents.
"""

import json
from typing import Optional
import google.generativeai as genai

from config import get_settings
from api.schemas import ShipmentData

settings = get_settings()


class StructuredExtractor:
    """
    Extract structured shipment data from logistics documents.
    
    Uses LLM with structured output prompt to extract:
    - shipment_id, shipper, consignee
    - pickup_datetime, delivery_datetime
    - equipment_type, mode, rate, currency
    - weight, carrier_name
    """
    
    EXTRACTION_PROMPT = """You are a logistics document data extractor. Extract the following fields from the document.

FIELDS TO EXTRACT:
1. shipment_id - Any shipment/load/order ID or reference number
2. shipper - The shipping company/sender name and address
3. consignee - The receiving company/destination name and address  
4. pickup_datetime - Pickup date and time (use ISO 8601 format if possible: YYYY-MM-DDTHH:MM:SS)
5. delivery_datetime - Delivery date and time (use ISO 8601 format if possible)
6. equipment_type - Type of equipment/trailer (e.g., "53' Dry Van", "Reefer", "Flatbed")
7. mode - Transportation mode (e.g., "TL", "LTL", "Intermodal", "Air")
8. rate - The total rate/cost as a number only (no currency symbol)
9. currency - Currency of the rate (e.g., "USD", "CAD", "EUR")
10. weight - Total weight with unit (e.g., "42,000 lbs", "19,000 kg")
11. carrier_name - Name of the carrier/trucking company

RULES:
- Return ONLY a valid JSON object
- Use null for any field not found in the document
- Do not guess or invent values - only extract what's explicitly stated
- For dates, try to use ISO 8601 format when possible
- For rate, extract only the numeric value (e.g., 2500.00 not "$2,500.00")

DOCUMENT:
{document_text}

Return ONLY valid JSON (no markdown, no explanation):"""

    def __init__(self):
        # Configure Gemini
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.llm = None
    
    def extract(self, document_text: str) -> tuple[ShipmentData, float]:
        """
        Extract structured data from document.
        
        Args:
            document_text: Full document text
            
        Returns:
            Tuple of (ShipmentData, confidence_score)
        """
        if not self.llm:
            return self._fallback_extraction(document_text)
        
        prompt = self.EXTRACTION_PROMPT.format(document_text=document_text[:8000])  # Limit context
        
        try:
            response = self.llm.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
            json_text = json_text.strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Create ShipmentData with validation
            shipment = ShipmentData(
                shipment_id=data.get("shipment_id"),
                shipper=data.get("shipper"),
                consignee=data.get("consignee"),
                pickup_datetime=data.get("pickup_datetime"),
                delivery_datetime=data.get("delivery_datetime"),
                equipment_type=data.get("equipment_type"),
                mode=data.get("mode"),
                rate=self._parse_rate(data.get("rate")),
                currency=data.get("currency"),
                weight=data.get("weight"),
                carrier_name=data.get("carrier_name")
            )
            
            # Calculate confidence based on fields found
            confidence = self._calculate_confidence(shipment)
            
            return shipment, confidence
            
        except json.JSONDecodeError as e:
            return self._fallback_extraction(document_text)
        except Exception as e:
            return self._fallback_extraction(document_text)
    
    def _parse_rate(self, rate_value) -> Optional[float]:
        """Parse rate value to float."""
        if rate_value is None:
            return None
        
        if isinstance(rate_value, (int, float)):
            return float(rate_value)
        
        if isinstance(rate_value, str):
            # Remove currency symbols and commas
            cleaned = rate_value.replace("$", "").replace(",", "").replace(" ", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _calculate_confidence(self, shipment: ShipmentData) -> float:
        """Calculate extraction confidence based on fields found."""
        fields = [
            shipment.shipment_id,
            shipment.shipper,
            shipment.consignee,
            shipment.pickup_datetime,
            shipment.delivery_datetime,
            shipment.equipment_type,
            shipment.mode,
            shipment.rate,
            shipment.currency,
            shipment.weight,
            shipment.carrier_name
        ]
        
        found_count = sum(1 for f in fields if f is not None)
        total_count = len(fields)
        
        # Base confidence on percentage of fields found
        # Weight important fields more heavily
        important_fields = [
            shipment.shipper,
            shipment.consignee,
            shipment.pickup_datetime,
            shipment.rate
        ]
        important_found = sum(1 for f in important_fields if f is not None)
        
        base_confidence = found_count / total_count
        important_bonus = (important_found / len(important_fields)) * 0.2
        
        return min(1.0, base_confidence + important_bonus)
    
    def _fallback_extraction(self, document_text: str) -> tuple[ShipmentData, float]:
        """Fallback extraction using regex patterns."""
        import re
        
        shipment = ShipmentData()
        
        # Try to extract common patterns
        # Rate patterns
        rate_patterns = [
            r'\$[\d,]+\.?\d*',
            r'Rate[:\s]+\$?[\d,]+\.?\d*',
            r'Total[:\s]+\$?[\d,]+\.?\d*'
        ]
        for pattern in rate_patterns:
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                rate_str = re.sub(r'[^\d.]', '', match.group())
                try:
                    shipment.rate = float(rate_str)
                    shipment.currency = "USD"
                    break
                except ValueError:
                    pass
        
        # Date patterns
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        dates = re.findall(date_pattern, document_text)
        if len(dates) >= 1:
            shipment.pickup_datetime = dates[0]
        if len(dates) >= 2:
            shipment.delivery_datetime = dates[1]
        
        # Weight patterns
        weight_pattern = r'(\d{1,3}(?:,\d{3})*)\s*(lbs?|kg|pounds?|kilograms?)'
        weight_match = re.search(weight_pattern, document_text, re.IGNORECASE)
        if weight_match:
            shipment.weight = f"{weight_match.group(1)} {weight_match.group(2)}"
        
        confidence = self._calculate_confidence(shipment)
        return shipment, confidence


# Singleton instance
_extractor: Optional[StructuredExtractor] = None


def get_structured_extractor() -> StructuredExtractor:
    """Get or create structured extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = StructuredExtractor()
    return _extractor
