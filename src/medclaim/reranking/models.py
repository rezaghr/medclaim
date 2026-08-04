"""Validated reranking configuration and lightweight protocols."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingConfigurationError(Exception):
    """Raised when reranking settings cannot be used safely."""


class EvidenceReranker(Protocol):
    model_id: str
    model_revision: str | None
    device: str
    batch_size: int
    maximum_input_length: int

    def rerank(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]: ...


class EvidenceVerifier(Protocol):
    model_id: str
    prompt_version: str

    def verify(
        self, claim: str, evidence: list[dict[str, str]]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RerankingConfiguration:
    enabled: bool = True
    model_id: str = DEFAULT_RERANKER_MODEL
    model_revision: str | None = None
    candidate_count: int = 20
    final_evidence_k: int = 5
    batch_size: int = 16
    device: str = "cpu"
    maximum_input_length: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            self._invalid("enabled must be a boolean")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            self._invalid("model_id must be a non-empty string")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            self._invalid("model_revision must be null or a non-empty string")
        if self.device not in {"cpu", "cuda", "auto", "ollama"}:
            self._invalid("device must be cpu, cuda, auto, or ollama")
        for name, value in (
            ("candidate_count", self.candidate_count),
            ("final_evidence_k", self.final_evidence_k),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 100
            ):
                self._invalid(f"{name} must be an integer from 1 to 100")
        if self.final_evidence_k > self.candidate_count:
            self._invalid("final_evidence_k cannot exceed candidate_count")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            self._invalid("batch_size must be a positive integer")
        if (
            not isinstance(self.maximum_input_length, int)
            or isinstance(self.maximum_input_length, bool)
            or self.maximum_input_length < 1
        ):
            self._invalid("maximum_input_length must be a positive integer")

    @staticmethod
    def _invalid(reason: str) -> None:
        raise RerankingConfigurationError(
            f"RERANKER_INVALID_CONFIGURATION: {reason}."
        )

    @classmethod
    def from_dict(cls, value: Any) -> "RerankingConfiguration":
        if not isinstance(value, dict):
            cls._invalid("configuration must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            cls._invalid(f"unknown field {unknown[0]!r}")
        missing = sorted(allowed - set(value))
        if missing:
            cls._invalid(f"missing field {missing[0]!r}")
        return cls(**value)

    def with_overrides(self, **overrides: Any) -> "RerankingConfiguration":
        values = asdict(self)
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return RerankingConfiguration.from_dict(values)


def load_reranking_configuration(path: Path) -> RerankingConfiguration:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise RerankingConfigurationError(
            f"RERANKER_INVALID_CONFIGURATION: Configuration does not exist: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RerankingConfigurationError(
            f"RERANKER_INVALID_CONFIGURATION: Could not parse {path}: {exc.msg}."
        ) from exc
    except OSError as exc:
        raise RerankingConfigurationError(
            f"RERANKER_INVALID_CONFIGURATION: Could not read {path}: {exc}."
        ) from exc
    return RerankingConfiguration.from_dict(value)
