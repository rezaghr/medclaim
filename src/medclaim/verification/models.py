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
        value = asdict(self)
        value["gate_decision"] = self.gate_decision.to_dict()
        return value


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    confidence: float
    explanation: str
    evidence_used: list[str]
    component_results: list[AtomicClaimResult] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    technical_metadata: dict[str, Any] = field(default_factory=dict)
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    confidence_method: str = "raw"
    calibrator_version: str | None = None
    confidence_warning: str = "Uncalibrated model confidence estimate."

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence_used": list(self.evidence_used),
            "component_results": [item.to_dict() for item in self.component_results],
            "limitations": list(self.limitations),
            "technical_metadata": dict(self.technical_metadata),
            "raw_confidence": (
                self.confidence if self.raw_confidence is None else self.raw_confidence
            ),
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_method": self.confidence_method,
            "calibrator_version": self.calibrator_version,
            "confidence_warning": self.confidence_warning,
        }
