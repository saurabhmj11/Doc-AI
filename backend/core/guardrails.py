from dataclasses import dataclass
from typing import Optional
import threading

from config import get_settings

settings = get_settings()


@dataclass
class GuardrailResult:

    status: str
    should_refuse: bool
    message: Optional[str] = None


class Guardrails:

    def __init__(self):

        self.similarity_threshold = settings.similarity_threshold
        self.confidence_threshold = settings.low_confidence_threshold
        self.grounding_threshold = settings.grounding_coverage_threshold

        self._lock = threading.Lock()

    # =====================================================
    # RETRIEVAL CHECK
    # =====================================================

    def check_retrieval(self, chunks):

        if not chunks:
            return GuardrailResult(
                status="no_retrieval",
                should_refuse=True,
                message="No relevant content found in the document."
            )

        top_score = chunks[0].get("similarity_score", 0)

        # FIX: downgrade refusal → warning
        if top_score < self.similarity_threshold:
            return GuardrailResult(
                status="weak_retrieval",
                should_refuse=False,
                message="Low similarity match — answer may be incomplete."
            )

        return GuardrailResult("passed", False)

    # =====================================================
    # CONFIDENCE CHECK
    # =====================================================

    def check_confidence(self, confidence):

        if confidence < self.confidence_threshold:

            return GuardrailResult(
                status="low_confidence",
                should_refuse=False,
                message="Low confidence — please verify against document."
            )

        if confidence < settings.high_confidence_threshold:

            return GuardrailResult(
                status="medium_confidence",
                should_refuse=False,
                message="Moderate confidence — verify details."
            )

        return GuardrailResult("passed", False)

    # =====================================================
    # GROUNDING CHECK
    # =====================================================

    def check_grounding(self, answer, chunks, coverage):

        if not answer or len(answer.strip()) < 3:
            return GuardrailResult(
                status="empty_answer",
                should_refuse=True,
                message="Unable to generate a reliable answer."
            )

        retrieval_score = chunks[0].get("similarity_score", 0) if chunks else 0

        combined = (coverage + retrieval_score) / 2

        if combined < self.grounding_threshold:

            return GuardrailResult(
                status="poor_grounding",
                should_refuse=False,
                message="Some parts may not be fully supported by the document."
            )

        return GuardrailResult("passed", False)

    # =====================================================
    # MASTER CHECK
    # =====================================================

    def run_all_checks(self, chunks, answer, context, confidence, breakdown):

        retrieval_check = self.check_retrieval(chunks)
        if retrieval_check.should_refuse:
            return retrieval_check

        coverage = breakdown.get("answer_coverage", 0)

        grounding_check = self.check_grounding(answer, chunks, coverage)
        if grounding_check.should_refuse:
            return grounding_check

        confidence_check = self.check_confidence(confidence)

        if grounding_check.status != "passed":
            return grounding_check

        if confidence_check.status != "passed":
            return confidence_check

        return GuardrailResult("passed", False)


# =====================================================
# THREAD SAFE SINGLETON
# =====================================================

_guardrails = None
_guardrails_lock = threading.Lock()

def get_guardrails():

    global _guardrails

    with _guardrails_lock:

        if _guardrails is None:
            _guardrails = Guardrails()

    return _guardrails
