"""Request-correlated safety and observability boundary around verification."""

from __future__ import annotations

import time
import uuid
from typing import Any

from medclaim.observability import (
    MetricsRegistry,
    RequestTracer,
    configure_json_logging,
    safe_claim_hash,
)
from medclaim.safety import MANDATORY_SAFETY_DISCLAIMER, route_scope

from .configuration import RuntimeSettings


class VerificationServiceError(Exception):
    """Controlled application-service failure."""


class VerificationService:
    def __init__(
        self,
        settings: RuntimeSettings,
        pipeline: Any | None = None,
        *,
        metrics: MetricsRegistry | None = None,
        logger=None,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.metrics = metrics or MetricsRegistry()
        self.logger = logger or configure_json_logging(settings.log_level)

    def verify(self, claim: str, request_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        resolved_request_id = (
            request_id.strip()
            if isinstance(request_id, str) and request_id.strip()
            else str(uuid.uuid4())
        )
        base = {
            "request_id": resolved_request_id,
            "environment": self.settings.environment,
            "configuration_id": self.settings.configuration_id,
            "configuration_hash": self.settings.configuration_hash,
            "safe_claim_hash": safe_claim_hash(claim),
            "claim_length": len(claim),
        }
        tracer = RequestTracer(self.logger, base)
        self.metrics.observe("verification_requests_total")
        tracer.emit("request_received")
        decision = route_scope(claim)
        tracer.emit("scope_decided", scope_decision=decision.model_dump())
        if decision.action != "VERIFY":
            self.metrics.observe("scope_limited_requests_total", category=decision.category)
            elapsed = time.perf_counter() - started
            self.metrics.observe("request_duration_seconds", elapsed)
            response = {
                "request_id": resolved_request_id,
                "scope_decision": decision.model_dump(),
                "verification": None,
                "safety_disclaimer": MANDATORY_SAFETY_DISCLAIMER,
                "total_latency_ms": elapsed * 1000,
            }
            tracer.emit("request_completed", total_latency_ms=elapsed * 1000)
            return response
        if self.pipeline is None:
            elapsed = time.perf_counter() - started
            self.metrics.observe("provider_errors_total", code="PIPELINE_UNAVAILABLE")
            tracer.emit(
                "request_failed",
                error_code="PIPELINE_UNAVAILABLE",
                retryable=True,
                total_latency_ms=elapsed * 1000,
            )
            raise VerificationServiceError(
                "PIPELINE_UNAVAILABLE: Verification artifacts are not loaded."
            )
        try:
            tracer.emit(
                "retrieval_started",
                corpus_version=self._version("corpus_version"),
                bm25_index_version=self._version("bm25_index_version"),
                dense_index_version=self._version("dense_index_version"),
                retrieval_mode=self.settings.retrieval_mode,
                top_k=self.settings.top_k,
            )
            reranking_enabled = self.settings.retrieval_mode == "hybrid_reranked"
            if reranking_enabled:
                tracer.emit(
                    "reranking_started",
                    reranker_model=self.settings.reranker_model,
                    candidate_count=self.settings.top_k,
                )
            verification_started = time.perf_counter()
            tracer.emit(
                "verification_started",
                verifier_type="llm",
                verifier_model=self.settings.llm_model,
                prompt_version=self.settings.prompt_version,
            )
            result = self.pipeline.verify(claim, resolved_request_id)
            verification_seconds = time.perf_counter() - verification_started
            value = result.to_dict() if callable(getattr(result, "to_dict", None)) else result
            if not isinstance(value, dict):
                raise VerificationServiceError("RESULT_INVALID: Pipeline result must be an object.")
            passage_ids = list(value.get("evidence_used", []))
            self.metrics.observe(
                "retrieval_duration_seconds",
                verification_seconds,
                mode=self.settings.retrieval_mode,
            )
            self.metrics.observe("retrieved_candidates_count", float(len(passage_ids)))
            tracer.emit(
                "retrieval_completed",
                passage_ids=passage_ids,
                stage_latency_ms=verification_seconds * 1000,
            )
            if reranking_enabled:
                reranking_seconds = self._reranking_latency_seconds(value)
                self.metrics.observe("reranking_duration_seconds", reranking_seconds)
                self.metrics.observe("reranked_candidates_count", float(len(passage_ids)))
                tracer.emit(
                    "reranking_completed",
                    reranker_model=self.settings.reranker_model,
                    candidate_count=len(passage_ids),
                    passage_ids=passage_ids,
                    stage_latency_ms=reranking_seconds * 1000,
                )
            gate_status = self._gate_status(value)
            abstention_reason = self._abstention_reason(value)
            tracer.emit(
                "gate_decided",
                gate_status=gate_status,
                abstention_reason=abstention_reason,
            )
            if gate_status == "ABSTAIN":
                self.metrics.observe(
                    "evidence_abstentions_total",
                    reason=abstention_reason or "INSUFFICIENT_EVIDENCE",
                )
            tracer.emit(
                "verification_completed",
                verdict=value.get("verdict"),
                confidence=value.get("confidence"),
                stage_latency_ms=verification_seconds * 1000,
            )
            tracer.emit("result_validated", validation_retry_count=0)
            elapsed = time.perf_counter() - started
            verdict = str(value.get("verdict", "NOT_ENOUGH_INFO"))
            self.metrics.observe("verification_results_total", verdict=verdict)
            self.metrics.observe(
                "verification_duration_seconds", verification_seconds, verifier="llm"
            )
            self.metrics.observe("request_duration_seconds", elapsed)
            tracer.emit(
                "request_completed",
                verdict=verdict,
                confidence=value.get("confidence"),
                total_latency_ms=elapsed * 1000,
            )
            return {
                "request_id": resolved_request_id,
                "scope_decision": decision.model_dump(),
                "verification": value,
                "safety_disclaimer": MANDATORY_SAFETY_DISCLAIMER,
                "total_latency_ms": elapsed * 1000,
            }
        except VerificationServiceError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - started
            code = str(exc).split(":", 1)[0] or "VERIFICATION_FAILED"
            self.metrics.observe("provider_errors_total", code=code)
            if isinstance(exc, TimeoutError):
                self.metrics.observe("provider_timeouts_total")
            if "SCHEMA" in code or "VALIDATION" in code:
                self.metrics.observe("schema_validation_failures_total")
            tracer.emit(
                "request_failed",
                error_code=code,
                retryable=False,
                total_latency_ms=elapsed * 1000,
            )
            raise VerificationServiceError(f"VERIFICATION_FAILED: {exc}") from exc

    def _version(self, name: str) -> str:
        if self.pipeline is None:
            return "unavailable"
        retriever = getattr(self.pipeline, "retriever", None)
        for target in (
            retriever,
            getattr(retriever, "sparse", None),
            getattr(retriever, "dense", None),
        ):
            if target is not None:
                manifest = getattr(target, "index_manifest", None)
                corpus = getattr(target, "corpus_manifest", None)
                if name == "corpus_version" and isinstance(corpus, dict):
                    return str(corpus.get("corpus_version", "unknown"))
                if name.endswith("index_version") and isinstance(manifest, dict):
                    return str(manifest.get("index_version", "unknown"))
        return "unknown"

    @staticmethod
    def _gate_status(value: dict[str, Any]) -> str:
        components = value.get("component_results", [])
        statuses = [
            item.get("gate_decision", {}).get("status")
            for item in components
            if isinstance(item, dict)
        ]
        return "ABSTAIN" if "ABSTAIN" in statuses else "PROCEED"

    @staticmethod
    def _abstention_reason(value: dict[str, Any]) -> str | None:
        components = value.get("component_results", [])
        for item in components:
            if isinstance(item, dict) and item.get("gate_decision", {}).get("status") == "ABSTAIN":
                return str(item.get("gate_decision", {}).get("reason", "INSUFFICIENT_EVIDENCE"))
        return None

    @staticmethod
    def _reranking_latency_seconds(value: dict[str, Any]) -> float:
        total_ms = 0.0
        for item in value.get("component_results", []):
            if not isinstance(item, dict):
                continue
            latency = item.get("retrieval_metadata", {}).get("latency_ms", {})
            if isinstance(latency, dict) and isinstance(latency.get("reranking"), (int, float)):
                total_ms += float(latency["reranking"])
        return total_ms / 1000
