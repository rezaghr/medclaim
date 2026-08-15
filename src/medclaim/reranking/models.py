"""Runtime reranking protocol and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RerankerError(Exception):
    """Raised for controlled reranker failures."""


class RerankingConfigurationError(Exception):
    """Raised when reranking settings are invalid."""


class EvidenceReranker(Protocol):
    model_id: str
    model_revision: str | None
    device: str
    batch_size: int

    def rerank(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


@dataclass(frozen=True)
class RerankingConfiguration:
    enabled: bool = True
    model_id: str = "dolphin-llama3:8b"
    model_revision: str | None = None
    candidate_count: int = 20
    final_evidence_k: int = 5
    batch_size: int = 9
    device: str = "ollama"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            self._invalid("enabled must be a boolean")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            self._invalid("model_id must be a non-empty string")
        if self.model_revision is not None:
            self._invalid("model_revision is unsupported for Ollama")
        if self.device != "ollama":
            self._invalid("device must be ollama")
        for name, value in (
            ("candidate_count", self.candidate_count),
            ("final_evidence_k", self.final_evidence_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
                self._invalid(f"{name} must be an integer from 1 to 100")
        if self.final_evidence_k > self.candidate_count:
            self._invalid("final_evidence_k cannot exceed candidate_count")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            self._invalid("batch_size must be a positive integer")

    @staticmethod
    def _invalid(reason: str) -> None:
        raise RerankingConfigurationError(f"RERANKER_INVALID_CONFIGURATION: {reason}.")
