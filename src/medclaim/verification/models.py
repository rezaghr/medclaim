"""Structured verification results shared by runtime and UI callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from medclaim.evidence_gate.gate import EvidenceGateDecision


@dataclass(frozen=True)
class AtomicClaimResult:
    component_id: str
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence_used: list[str]
    gate_decision: EvidenceGateDecision
    limitations: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_candidates: list[dict[str, Any]] = field(default_factory=list)
    model_input_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    confidence: float
    explanation: str
    evidence_used: list[str]
    component_results: list[AtomicClaimResult] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    technical_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
