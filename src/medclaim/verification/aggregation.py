"""Deterministic aggregation of independently verified atomic claims."""

from __future__ import annotations

import math

from .models import AtomicClaimResult, VerificationResult

COMPONENT_VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
VERDICT_PHRASES = {
    "SUPPORTS": "is supported by the indexed evidence",
    "REFUTES": "is contradicted by the indexed evidence",
    "NOT_ENOUGH_INFO": "could not be established from the available evidence",
}
ORDINALS = ("first", "second", "third", "fourth")
RESEARCH_LIMITATION = "Research use only; this result is not medical advice, diagnosis, or treatment guidance."


class AggregationError(Exception):
    """Raised when component outputs cannot be aggregated safely."""


def _validate_component(item: AtomicClaimResult) -> None:
    if item.verdict not in COMPONENT_VERDICTS:
        raise AggregationError(f"AGGREGATION_INVALID_VERDICT: Unsupported verdict {item.verdict!r}.")
    if (
        isinstance(item.confidence, bool)
        or not isinstance(item.confidence, (int, float))
        or not math.isfinite(float(item.confidence))
        or not 0 <= item.confidence <= 1
    ):
        raise AggregationError("AGGREGATION_INVALID_VERDICT: Component confidence must be from zero to one.")


def aggregate_component_results(
    component_results: list[AtomicClaimResult],
    *,
    decomposition_warnings: list[str] | None = None,
    prompt_version: str | None = None,
) -> VerificationResult:
    if not isinstance(component_results, list) or not component_results or len(component_results) > 4:
        raise AggregationError("AGGREGATION_INVALID_VERDICT: One to four component results are required.")
    for item in component_results:
        _validate_component(item)
    verdicts = {item.verdict for item in component_results}
    verdict = component_results[0].verdict if len(verdicts) == 1 else "MIXED"
    confidence = min(float(item.confidence) for item in component_results)
    evidence_used: list[str] = []
    for item in component_results:
        for passage_id in item.evidence_used:
            if passage_id not in evidence_used:
                evidence_used.append(passage_id)
    if len(component_results) == 1:
        explanation = component_results[0].explanation
    else:
        count_word = {2: "two", 3: "three", 4: "four"}[len(component_results)]
        sentences = [f"This claim contains {count_word} verifiable components."]
        sentences.extend(
            f"The {ORDINALS[index]} component {VERDICT_PHRASES[item.verdict]}."
            for index, item in enumerate(component_results)
        )
        sentences.append(f"The overall result is {verdict}.")
        explanation = " ".join(sentences)
    limitations = [RESEARCH_LIMITATION]
    for item in component_results:
        for limitation in item.limitations:
            if limitation not in limitations:
                limitations.append(limitation)
    warnings = list(decomposition_warnings or [])
    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        evidence_used=evidence_used,
        component_results=list(component_results),
        limitations=limitations,
        technical_metadata={
            "decomposition_warnings": warnings,
            "decomposition_prompt_version": prompt_version,
            "confidence_note": "Confidence is an estimate, not a clinical probability.",
        },
        raw_confidence=confidence,
    )
