"""
Guardrails - the safety net for our RAG answers

Three main checks:
1. Did we find relevant chunks at all?
2. Is the confidence score acceptable?
3. Is the answer actually grounded in the context?

If any of these fail, we either refuse to answer or add a warning.
Better to say "I don't know" than make stuff up.
"""

from dataclasses import dataclass
from typing import Optional
from config import get_settings

settings = get_settings()


@dataclass
class GuardrailResult:
    """What the guardrails decided"""
    status: str  # passed, warning, refused
    should_refuse: bool
    message: Optional[str] = None


class Guardrails:
    """
    Catches bad answers before they reach the user.
    
    Tuned these thresholds through trial and error - might need 
    adjustment for different doc types.
    """
    
    def __init__(self):
        self.similarity_threshold = settings.similarity_threshold
        self.confidence_threshold = settings.low_confidence_threshold
        self.grounding_threshold = settings.grounding_coverage_threshold
    
    def check_retrieval(self, chunks: list[dict]) -> GuardrailResult:
        """
        First check - did we actually find relevant content?
        If top chunks have low similarity, the doc probably doesn't 
        have what we're looking for.
        """
        if not chunks:
            return GuardrailResult(
                status="low_retrieval",
                should_refuse=True,
                message="I couldn't find any relevant information in the document for this question."
            )
        
        # check the best match
        top_score = chunks[0].get("similarity_score", 0)
        
        if top_score < self.similarity_threshold:
            return GuardrailResult(
                status="low_retrieval",
                should_refuse=True,
                message="The question doesn't seem to match the document content well. Please try rephrasing or verify the document contains this information."
            )
        
        return GuardrailResult(status="passed", should_refuse=False)
    
    def check_confidence(self, confidence: float) -> GuardrailResult:
        """
        Is the confidence score acceptable?
        Low confidence = uncertain answer = warning or refusal
        """
        if confidence < self.confidence_threshold:
            return GuardrailResult(
                status="low_confidence",
                should_refuse=True,
                message="I'm not confident enough in this answer to provide it. The information may not be clearly stated in the document."
            )
        
        # medium confidence gets a warning but we still answer
        if confidence < settings.high_confidence_threshold:
            return GuardrailResult(
                status="medium_confidence",
                should_refuse=False,
                message="This answer has moderate confidence. Please verify against the source document."
            )
        
        return GuardrailResult(status="passed", should_refuse=False)
    
    def check_grounding(self, answer: str, context: str, coverage: float) -> GuardrailResult:
        """
        Is the answer actually based on the context?
        If most of the answer words aren't in the context, something's fishy.
        """
        if coverage < self.grounding_threshold:
            return GuardrailResult(
                status="poor_grounding",
                should_refuse=False,  # warn but don't refuse
                message="Some parts of this answer may not be directly supported by the document."
            )
        
        return GuardrailResult(status="passed", should_refuse=False)
    
    def run_all_checks(
        self,
        chunks: list[dict],
        answer: str,
        context: str,
        confidence: float,
        confidence_breakdown: dict
    ) -> GuardrailResult:
        """
        Run the full gauntlet of checks.
        Returns the most severe result.
        """
        # confidence is the main gate
        confidence_check = self.check_confidence(confidence)
        if confidence_check.should_refuse:
            return confidence_check
        
        # check if answer is grounded
        coverage = confidence_breakdown.get("answer_coverage", 0)
        grounding_check = self.check_grounding(answer, context, coverage)
        
        # if there's any warning, return that
        if grounding_check.status != "passed":
            return grounding_check
        if confidence_check.status != "passed":
            return confidence_check
        
        # all good
        return GuardrailResult(status="passed", should_refuse=False)


# singleton
_guardrails: Optional[Guardrails] = None

def get_guardrails() -> Guardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
