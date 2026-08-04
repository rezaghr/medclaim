"""Per-component retrieval, gating, verification, and aggregation."""

from __future__ import annotations

import math
import time
from typing import Any

from medclaim.decomposition.decomposer import ClaimDecomposer, DecompositionError
from medclaim.evidence_gate.gate import EvidenceGate
from medclaim.explanation.validation import ExplanationValidator, validate_with_one_correction

from .aggregation import COMPONENT_VERDICTS, aggregate_component_results
from .models import AtomicClaimResult, VerificationResult

ABSTENTION_EXPLANATION = "The indexed evidence is not sufficiently relevant to verify this claim."
ABSTENTION_LIMITATIONS = [
    "The evidence corpus may not cover the claim.",
    "Weak topical similarity is not treated as support or contradiction.",
]
RETRIEVAL_RESULT_FIELDS = (
    "rank",
    "passage_id",
    "document_id",
    "dataset",
    "text",
    "bm25_score",
    "dense_score",
    "rrf_score",
    "reranker_score",
    "reranker_rank",
    "pre_rerank_rank",
)


class ComponentVerificationError(Exception):
    """Raised when one component cannot complete the verification contract."""


def _retrieval_trace(
    retrieval: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = {
        key: value
        for key, value in retrieval.items()
        if key not in {"results", "candidate_results"}
    }
    traced_candidates = [
        {key: item[key] for key in RETRIEVAL_RESULT_FIELDS if key in item}
        for item in candidates
        if isinstance(item, dict)
    ]
    return metadata, traced_candidates


class VerificationPipeline:
    def __init__(
        self,
        retriever: Any,
        verifier: Any,
        evidence_gate: EvidenceGate,
        decomposer: ClaimDecomposer | None = None,
        decomposition_mode: str = "auto",
        final_evidence_k: int = 5,
        explanation_validator: ExplanationValidator | None = None,
    ) -> None:
        if decomposition_mode not in {"off", "auto", "always"}:
            raise DecompositionError(f"DECOMPOSITION_INVALID_MODE: Unsupported mode {decomposition_mode!r}.")
        if not isinstance(final_evidence_k, int) or isinstance(final_evidence_k, bool) or not 1 <= final_evidence_k <= 100:
            raise ComponentVerificationError("COMPONENT_VERIFICATION_FAILED: final_evidence_k must be from 1 to 100.")
        self.retriever = retriever
        self.verifier = verifier
        self.evidence_gate = evidence_gate
        self.decomposer = decomposer or ClaimDecomposer(None)
        self.decomposition_mode = decomposition_mode
        self.final_evidence_k = final_evidence_k
        self.explanation_validator = explanation_validator or ExplanationValidator()

    def verify(self, claim: str, request_id: str) -> VerificationResult:
        if not isinstance(claim, str) or not claim.strip() or not isinstance(request_id, str) or not request_id.strip():
            raise ComponentVerificationError("COMPONENT_VERIFICATION_FAILED: Claim and request ID must be non-empty.")
        outcome = self.decomposer.decompose(claim.strip(), self.decomposition_mode)
        components = [
            self._verify_component(
                atomic.text,
                f"{request_id}:component:{atomic.index}",
            )
            for atomic in outcome.decomposition.atomic_claims
        ]
        result = aggregate_component_results(
            components,
            decomposition_warnings=outcome.warnings,
            prompt_version=outcome.prompt_version,
        )
        metadata = dict(result.technical_metadata)
        metadata.update(
            {
                "decomposition_mode": self.decomposition_mode,
                "decomposition_attempted": outcome.attempted,
                "is_compound": outcome.decomposition.is_compound,
            }
        )
        return VerificationResult(
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            evidence_used=result.evidence_used,
            component_results=result.component_results,
            limitations=result.limitations,
            technical_metadata=metadata,
            raw_confidence=result.raw_confidence,
            calibrated_confidence=result.calibrated_confidence,
            confidence_method=result.confidence_method,
            calibrator_version=result.calibrator_version,
            confidence_warning=result.confidence_warning,
        )

    def _verify_component(self, claim: str, component_id: str) -> AtomicClaimResult:
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        try:
            retrieval = self.retriever.search(claim, top_k=self.final_evidence_k)
            candidates = retrieval.get("results")
            if not isinstance(candidates, list):
                raise ValueError("Retriever output has no results list")
        except Exception as exc:
            raise ComponentVerificationError(f"COMPONENT_VERIFICATION_FAILED: Retrieval failed for {component_id}: {exc}.") from exc
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        trace_candidates = retrieval.get("candidate_results", candidates)
        if not isinstance(trace_candidates, list):
            raise ComponentVerificationError(
                f"COMPONENT_VERIFICATION_FAILED: Retrieval trace failed for {component_id}."
            )
        retrieval_metadata, retrieved_candidates = _retrieval_trace(
            retrieval, trace_candidates
        )
        gate_started = time.perf_counter()
        decision, selected = self.evidence_gate.evaluate(candidates, self.final_evidence_k)
        gate_ms = (time.perf_counter() - gate_started) * 1000
        if decision.status == "ABSTAIN":
            component = AtomicClaimResult(
                component_id=component_id,
                claim=claim,
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                explanation=ABSTENTION_EXPLANATION,
                evidence_used=[],
                gate_decision=decision,
                limitations=list(ABSTENTION_LIMITATIONS),
                retrieval_metadata=retrieval_metadata,
                retrieved_candidates=retrieved_candidates,
                model_input_evidence=[],
                latency_ms={
                    "retrieval": retrieval_ms,
                    "gate": gate_ms,
                    "verifier": 0.0,
                    "total": (time.perf_counter() - total_started) * 1000,
                },
            )
            self.explanation_validator.validate(
                component.to_dict(), [], gate_abstained=True
            )
            return component
        evidence = [
            {"passage_id": item["passage_id"], "text": item["text"]}
            for item in selected
        ]
        verifier_started = time.perf_counter()
        try:
            implementation = str(
                getattr(self.verifier, "implementation", "llm")
            ).casefold()
            output = validate_with_one_correction(
                self.verifier,
                claim,
                evidence,
                self.explanation_validator,
                implementation=implementation,
            )
            if not isinstance(output, dict) or output.get("verdict") not in COMPONENT_VERDICTS:
                raise ValueError("Verifier returned an invalid component verdict")
            confidence = output.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= confidence <= 1
            ):
                raise ValueError("Verifier confidence must be from zero to one")
            explanation = output.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("Verifier explanation must be non-empty")
            selected_ids = [item["passage_id"] for item in evidence]
            evidence_used = output.get("evidence_used", selected_ids)
            if not isinstance(evidence_used, list) or any(item not in selected_ids for item in evidence_used):
                raise ValueError("Verifier evidence_used must reference selected passages")
            limitations = output.get("limitations", [])
            if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
                raise ValueError("Verifier limitations must be a list of strings")
        except Exception as exc:
            raise ComponentVerificationError(f"COMPONENT_VERIFICATION_FAILED: Verifier failed for {component_id}: {exc}.") from exc
        verifier_ms = (time.perf_counter() - verifier_started) * 1000
        return AtomicClaimResult(
            component_id=component_id,
            claim=claim,
            verdict=output["verdict"],
            confidence=float(confidence),
            explanation=explanation.strip(),
            evidence_used=list(evidence_used),
            gate_decision=decision,
            limitations=list(limitations),
            retrieval_metadata=retrieval_metadata,
            retrieved_candidates=retrieved_candidates,
            model_input_evidence=[dict(item) for item in evidence],
            latency_ms={
                "retrieval": retrieval_ms,
                "gate": gate_ms,
                "verifier": verifier_ms,
                "total": (time.perf_counter() - total_started) * 1000,
            },
        )
