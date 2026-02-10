"""
Query Planner Module

Responsibility: Deconstruct user queries into actionable intents and parameters.
This is the "Brain" of the Agentic RAG system (RAG 2.0).
"""

import logging
import json
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import google.generativeai as genai

from config import get_settings
from core.error_handling import get_gemini_circuit_breaker, api_retry, APIError

settings = get_settings()
logger = logging.getLogger(__name__)

class QueryIntent(str, Enum):
    ENTITY_LOOKUP = "ENTITY_LOOKUP"       # asking for specific field/fact
    SUMMARIZATION = "SUMMARIZATION"       # asking for summary
    COMPARISON = "COMPARISON"             # asking to compare 2+ things
    GENERAL = "GENERAL"                   # catch-all

class QueryPlan(BaseModel):
    intent: QueryIntent
    target_entities: List[str] = []       # e.g. ["shipper", "pickup_date"]
    reasoning: str = ""

class QueryPlanner:
    """
    Analyzes user query to determine the best retrieval strategy.
    """
    
    PLANNING_PROMPT = """You are the complications logic layer of a RAG system.
    Analyze the user's question and determine the intent.
    
    INTENTS:
    - ENTITY_LOOKUP: User is asking for a specific fact, name, date, amount, or ID. (e.g., "Who is the shipper?", "What is the rate?", "When is pickup?")
    - SUMMARIZATION: User wants an overview. (e.g., "Summarize this load", "What is this document?")
    - COMPARISON: User is comparing two things. (e.g., "Is the rate higher than last time?")
    - GENERAL: Greeting, chit-chat, or unclear.

    Available Entity Types: [shipment_id, shipper, consignee, pickup_datetime, delivery_datetime, equipment_type, mode, rate, currency, weight, carrier_name]

    OUTPUT JSON FORMAT:
    {{
        "intent": "ENTITY_LOOKUP",
        "target_entities": ["shipper"],
        "reasoning": "User asked for the shipper identity."
    }}

    QUESTION: {question}
    
    Return ONLY valid JSON:"""

    def __init__(self):
        self.llm_mode = settings.llm_mode
        if settings.gemini_api_key:
             genai.configure(api_key=settings.gemini_api_key)
             # Use Flash for speed in planning
             self.llm = genai.GenerativeModel('models/gemini-flash-latest') 
             self.circuit_breaker = get_gemini_circuit_breaker()
             logger.info("Query Planner initialized (Gemini Flash)")
        else:
             self.llm = None
             logger.warning("Query Planner initialized WITHOUT LLM")

    def plan(self, question: str) -> QueryPlan:
        """
        Generate a retrieval plan for the question.
        """
        if not self.llm:
            return QueryPlan(intent=QueryIntent.GENERAL, reasoning="No LLM available")

        try:
            prompt = self.PLANNING_PROMPT.format(question=question[:500])
            
            @api_retry(api_name="QueryPlanner", max_attempts=2)
            def _call_planner():
                response = self.llm.generate_content(prompt)
                return response.text.strip()
            
            json_text = self.circuit_breaker.call(_call_planner)
            
            # Robust JSON extraction
            import re
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from QueryPlanner: {json_text}")
                return QueryPlan(intent=QueryIntent.GENERAL, reasoning="JSON Parse Error")
            
            # Normalize intent
            intent_str = data.get("intent", "GENERAL").upper()
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"Invalid intent '{intent_str}', defaulting to GENERAL")
                intent = QueryIntent.GENERAL

            return QueryPlan(
                intent=intent,
                target_entities=data.get("target_entities", []),
                reasoning=data.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            # Fallback
            return QueryPlan(intent=QueryIntent.GENERAL, reasoning="Planning failed")
