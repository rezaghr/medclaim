"""Configurable evidence-sufficiency gate for reranked passages."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

GateStatus = Literal["PROCEED", "ABSTAIN"]
GateReason = Literal[
    "RELEVANCE_REQUIREMENTS_MET",
    "NO_CANDIDATES",
    "TOP_SCORE_BELOW_THRESHOLD",
    "TOO_FEW_RELEVANT_PASSAGES",
    "TOO_FEW_UNIQUE_DOCUMENTS",
]


class EvidenceGateError(Exception):
    """Raised for controlled evidence-gate configuration or input errors."""


@dataclass(frozen=True)
class EvidenceGateConfiguration:
    version: str
    enabled: bool
    minimum_score: float
    minimum_relevant_passages: int = 1
    minimum_unique_documents: int = 1
    score_field: str = "reranker_score"

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            self._invalid("version must be a non-empty string")
        if not isinstance(self.enabled, bool):
            self._invalid("enabled must be a boolean")
        if (
            isinstance(self.minimum_score, bool)
            or not isinstance(self.minimum_score, (int, float))
            or not math.isfinite(float(self.minimum_score))
        ):
            self._invalid("minimum_score must be finite")
        for name, value in (
            ("minimum_relevant_passages", self.minimum_relevant_passages),
            ("minimum_unique_documents", self.minimum_unique_documents),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                self._invalid(f"{name} must be a positive integer")
        if not isinstance(self.score_field, str) or not self.score_field.strip():
            self._invalid("score_field must be a non-empty string")

    @staticmethod
    def _invalid(message: str) -> None:
        raise EvidenceGateError(f"GATE_CONFIG_INVALID: {message}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGateDecision:
    status: GateStatus
    reason: GateReason
    threshold: float
    score_field: str
    top_score: float | None
    relevant_passage_count: int
    unique_document_count: int
    evidence_passage_ids: list[str]
    gate_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceGate:
    def __init__(self, configuration: EvidenceGateConfiguration) -> None:
        self.configuration = configuration

    def evaluate(
        self, candidates: list[dict[str, Any]], final_evidence_k: int | None = None
    ) -> tuple[EvidenceGateDecision, list[dict[str, Any]]]:
        if not isinstance(candidates, list):
            raise EvidenceGateError("GATE_SCORE_FIELD_MISSING: Candidates must be a list.")
        if final_evidence_k is not None and (
            not isinstance(final_evidence_k, int)
            or isinstance(final_evidence_k, bool)
            or final_evidence_k < 1
        ):
            raise EvidenceGateError("GATE_CONFIG_INVALID: final_evidence_k must be positive.")
        limit = final_evidence_k or len(candidates)
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("passage_id"), str
            ) or not isinstance(candidate.get("document_id"), str):
                raise EvidenceGateError("GATE_SCORE_FIELD_MISSING: Candidate provenance is invalid.")

        if not self.configuration.enabled:
            selected = [dict(candidate) for candidate in candidates[:limit]]
            return self._decision("PROCEED", "RELEVANCE_REQUIREMENTS_MET", None, selected), selected
        if not candidates:
            return self._decision("ABSTAIN", "NO_CANDIDATES", None, []), []

        scored: list[tuple[dict[str, Any], float]] = []
        for candidate in candidates:
            field = self.configuration.score_field
            if field not in candidate:
                raise EvidenceGateError(
                    f"GATE_SCORE_FIELD_MISSING: Passage {candidate.get('passage_id')!r} "
                    f"is missing {field!r}."
                )
            raw_score = candidate[field]
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
            ):
                raise EvidenceGateError(
                    f"GATE_SCORE_FIELD_MISSING: Passage {candidate.get('passage_id')!r} "
                    f"has an invalid {field}."
                )
            scored.append((candidate, float(raw_score)))
        top_score = max(score for _, score in scored)
        relevant = [
            dict(candidate)
            for candidate, score in scored
            if score >= self.configuration.minimum_score
        ]
        selected = relevant[:limit]
        if top_score < self.configuration.minimum_score:
            return self._decision("ABSTAIN", "TOP_SCORE_BELOW_THRESHOLD", top_score, relevant), []
        if len(relevant) < self.configuration.minimum_relevant_passages:
            return self._decision("ABSTAIN", "TOO_FEW_RELEVANT_PASSAGES", top_score, relevant), []
        unique_documents = {candidate["document_id"] for candidate in relevant}
        if len(unique_documents) < self.configuration.minimum_unique_documents:
            return self._decision("ABSTAIN", "TOO_FEW_UNIQUE_DOCUMENTS", top_score, relevant), []
        return self._decision(
            "PROCEED", "RELEVANCE_REQUIREMENTS_MET", top_score, selected
        ), selected

    def decide(self, candidates: list[dict[str, Any]]) -> EvidenceGateDecision:
        return self.evaluate(candidates)[0]

    def _decision(
        self,
        status: GateStatus,
        reason: GateReason,
        top_score: float | None,
        relevant: list[dict[str, Any]],
    ) -> EvidenceGateDecision:
        return EvidenceGateDecision(
            status=status,
            reason=reason,
            threshold=float(self.configuration.minimum_score),
            score_field=self.configuration.score_field,
            top_score=top_score,
            relevant_passage_count=len(relevant),
            unique_document_count=len({item.get("document_id") for item in relevant}),
            evidence_passage_ids=[str(item["passage_id"]) for item in relevant],
            gate_version=self.configuration.version,
        )
