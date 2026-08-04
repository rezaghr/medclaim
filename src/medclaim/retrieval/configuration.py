"""Small validated retrieval-mode configuration without a framework dependency."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RetrievalConfigurationError(Exception):
    """Raised when retrieval settings are invalid or incomplete."""


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "hybrid"
    sparse_top_k: int = 50
    dense_top_k: int = 50
    fusion_top_k: int = 30
    rrf_k: int = 60
    final_evidence_k: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"bm25", "dense", "hybrid", "hybrid_reranked"}:
            raise RetrievalConfigurationError(
                "RETRIEVAL_INVALID_MODE: Mode must be bm25, dense, hybrid, or "
                "hybrid_reranked."
            )
        candidate_values = (
            self.sparse_top_k,
            self.dense_top_k,
            self.fusion_top_k,
            self.final_evidence_k,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 100
            for value in candidate_values
        ):
            raise RetrievalConfigurationError(
                "RETRIEVAL_INVALID_TOP_K: Candidate limits must be integers from 1 to 100."
            )
        if self.fusion_top_k > self.sparse_top_k + self.dense_top_k:
            raise RetrievalConfigurationError(
                "RETRIEVAL_INVALID_TOP_K: fusion_top_k cannot exceed the total "
                "component candidate limits."
            )
        if self.final_evidence_k > self.fusion_top_k:
            raise RetrievalConfigurationError(
                "RETRIEVAL_INVALID_TOP_K: final_evidence_k cannot exceed fusion_top_k."
            )
        if (
            not isinstance(self.rrf_k, int)
            or isinstance(self.rrf_k, bool)
            or self.rrf_k <= 0
        ):
            raise RetrievalConfigurationError(
                "RETRIEVAL_INVALID_RRF_K: rrf_k must be a positive integer."
            )

    @classmethod
    def from_dict(cls, value: Any) -> "RetrievalSettings":
        if not isinstance(value, dict):
            raise RetrievalConfigurationError(
                "RETRIEVAL_CONFIG_INVALID: Configuration must be a JSON object."
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RetrievalConfigurationError(
                f"RETRIEVAL_CONFIG_UNKNOWN_FIELD: Unknown field {unknown[0]!r}."
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise RetrievalConfigurationError(
                f"RETRIEVAL_CONFIG_MISSING_FIELD: Missing field {missing[0]!r}."
            )
        return cls(**value)

    def with_overrides(self, **overrides: Any) -> "RetrievalSettings":
        values = asdict(self)
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return RetrievalSettings.from_dict(values)


def load_retrieval_settings(path: Path) -> RetrievalSettings:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise RetrievalConfigurationError(
            f"RETRIEVAL_CONFIG_NOT_FOUND: Configuration does not exist: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RetrievalConfigurationError(
            f"RETRIEVAL_CONFIG_INVALID: Could not parse {path}: {exc.msg}."
        ) from exc
    except OSError as exc:
        raise RetrievalConfigurationError(
            f"RETRIEVAL_CONFIG_READ_FAILED: Could not read {path}: {exc}."
        ) from exc
    return RetrievalSettings.from_dict(value)


def create_retriever(
    mode: str,
    *,
    sparse_retriever: Any | None,
    dense_retriever: Any | None,
    reranked_retriever: Any | None = None,
    settings: RetrievalSettings | None = None,
) -> Any:
    """Select BM25, dense, or hybrid retrieval for a calling pipeline."""
    selected_settings = settings or RetrievalSettings(mode=mode)
    if mode != selected_settings.mode:
        selected_settings = selected_settings.with_overrides(mode=mode)
    if mode == "bm25":
        if sparse_retriever is None:
            raise RetrievalConfigurationError(
                "RETRIEVAL_NOT_READY: BM25 artifacts are not configured."
            )
        return sparse_retriever
    if mode == "dense":
        if dense_retriever is None:
            raise RetrievalConfigurationError(
                "RETRIEVAL_NOT_READY: Dense artifacts are not configured."
            )
        return dense_retriever
    if mode == "hybrid":
        if sparse_retriever is None or dense_retriever is None:
            raise RetrievalConfigurationError(
                "RETRIEVAL_NOT_READY: Hybrid mode requires both indexes."
            )
        from .hybrid import HybridRetriever

        return HybridRetriever(
            sparse_retriever,
            dense_retriever,
            sparse_top_k=selected_settings.sparse_top_k,
            dense_top_k=selected_settings.dense_top_k,
            fusion_top_k=selected_settings.fusion_top_k,
            rrf_k=selected_settings.rrf_k,
        )
    if mode == "hybrid_reranked":
        if reranked_retriever is None:
            raise RetrievalConfigurationError(
                "RETRIEVAL_NOT_READY: Reranker artifacts are not configured."
            )
        return reranked_retriever
    raise RetrievalConfigurationError(
        "RETRIEVAL_INVALID_MODE: Mode must be bm25, dense, hybrid, or "
        "hybrid_reranked."
    )
