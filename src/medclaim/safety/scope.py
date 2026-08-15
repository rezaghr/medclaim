"""Deterministic safety routing before claim verification."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

SCOPE_RULE_VERSION = "safety-scope-v4"
ScopeCategory = Literal[
    "GENERAL_CLAIM",
    "INVALID_INPUT",
    "PERSONAL_DIAGNOSIS",
    "TREATMENT_REQUEST",
    "DOSAGE_REQUEST",
    "MEDICATION_CHANGE_REQUEST",
    "EMERGENCY_PERSONAL_REQUEST",
]
MANDATORY_SAFETY_DISCLAIMER = (
    "MedClaimRAG verifies textual claims against a limited indexed corpus. "
    "It is an educational research prototype. It is not a doctor, diagnostic "
    "system, treatment recommender, or emergency service. Its results are not medical advice."
)


class ScopeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["VERIFY", "LIMIT_SCOPE", "EMERGENCY_LIMITED_RESPONSE"]
    category: ScopeCategory
    message: str | None
    rule_version: str


_PERSONAL = re.compile(r"\b(i|i'm|im|me|my|mine|we|our|someone with me)\b", re.I)
_EMERGENCY = re.compile(
    r"\b(can(?:not|'t) breathe|chest pain|overdos(?:e|ed)|severe bleeding|"
    r"unconscious|suicid(?:e|al)|anaphylaxis|stroke|heart attack|medical emergency)\b",
    re.I,
)
_DOSAGE = re.compile(r"\b(dosage|dose|how many (?:mg|pills)|how much .*\btake\b)\b", re.I)
_MEDICATION_CHANGE = re.compile(
    r"\b(should|can|may)\s+i\s+(stop|discontinue|change|switch|reduce|increase|skip)\b.*"
    r"\b(medication|medicine|prescription|drug|dose|treatment)?\b",
    re.I,
)
_DIAGNOSIS = re.compile(
    r"\b(diagnose me|what (?:illness|disease|condition) (?:do )?i have|"
    r"do i have [a-z]|what is wrong with me|my symptoms mean)\b",
    re.I,
)
_TREATMENT = re.compile(
    r"\b(what (?:medication|medicine|drug|treatment) should i (?:take|use)|"
    r"how should i treat|treat my|recommend (?:a )?(?:medication|treatment))\b",
    re.I,
)


def _limited(category: ScopeCategory, message: str) -> ScopeDecision:
    return ScopeDecision(
        action="LIMIT_SCOPE",
        category=category,
        message=message,
        rule_version=SCOPE_RULE_VERSION,
    )


def route_scope(text: str) -> ScopeDecision:
    """Limit unsafe personal requests and allow other claims to reach retrieval."""
    if not isinstance(text, str) or not text.strip():
        return _limited(
            "INVALID_INPUT",
            "Enter a non-empty textual claim to verify.",
        )
    value = " ".join(text.split())
    if _PERSONAL.search(value) and _EMERGENCY.search(value):
        return ScopeDecision(
            action="EMERGENCY_LIMITED_RESPONSE",
            category="EMERGENCY_PERSONAL_REQUEST",
            message=(
                "This service cannot assess emergencies. Contact local emergency services or a "
                "qualified medical professional now; do not rely on claim verification for care."
            ),
            rule_version=SCOPE_RULE_VERSION,
        )
    if _DOSAGE.search(value) and (_PERSONAL.search(value) or "should" in value.casefold()):
        return _limited(
            "DOSAGE_REQUEST",
            "MedClaimRAG cannot provide personalized dosage instructions.",
        )
    if _MEDICATION_CHANGE.search(value):
        return _limited(
            "MEDICATION_CHANGE_REQUEST",
            "MedClaimRAG cannot advise changing prescribed treatment; consult a qualified clinician.",
        )
    if _DIAGNOSIS.search(value):
        return _limited(
            "PERSONAL_DIAGNOSIS",
            "MedClaimRAG cannot diagnose symptoms or rank possible diseases.",
        )
    if _TREATMENT.search(value):
        return _limited(
            "TREATMENT_REQUEST", "MedClaimRAG cannot recommend medication or treatment."
        )
    # Medical relevance is deliberately not guessed from a keyword allowlist. The
    # retriever and evidence gate are the source of truth for corpus coverage, so
    # unfamiliar terminology and phrasing can proceed and abstain naturally when
    # the indexed evidence does not support them.
    return ScopeDecision(
        action="VERIFY",
        category="GENERAL_CLAIM",
        message=None,
        rule_version=SCOPE_RULE_VERSION,
    )
