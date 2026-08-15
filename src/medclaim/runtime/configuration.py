"""Strict versioned non-secret configuration with environment-only secrets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RuntimeConfigurationError(Exception):
    """Raised when deployment configuration is unknown, unsafe, or incomplete."""


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_id: str = "deployment-v1"
    environment: str = "development"
    llm_model: str = "dolphin-llama3:8b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    corpus_dir: Path | None = None
    bm25_index_dir: Path | None = None
    dense_index_dir: Path | None = None
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_query_prefix: str = ""
    reranker_provider: str = "ollama"
    reranker_model: str = "disabled"
    reranker_batch_size: int = Field(default=9, ge=1, le=30)
    retrieval_mode: str = "hybrid"
    retrieval_candidate_count: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=100)
    gate_minimum_score: float = Field(default=0.7, ge=0)
    gate_minimum_relevant_passages: int = Field(default=1, ge=1, le=100)
    gate_minimum_unique_documents: int = Field(default=1, ge=1, le=100)
    prompt_version: str = "medical-verifier-v3-secure"
    gate_version: str = "evidence-gate-v1"
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def configuration_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


_ENV_MAP = {
    "MEDCLAIM_ENV": "environment",
    "LLM_MODEL": "llm_model",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
    "MEDCLAIM_CORPUS_DIR": "corpus_dir",
    "MEDCLAIM_BM25_INDEX_DIR": "bm25_index_dir",
    "MEDCLAIM_DENSE_INDEX_DIR": "dense_index_dir",
    "MEDCLAIM_EMBEDDING_PROVIDER": "embedding_provider",
    "MEDCLAIM_EMBEDDING_MODEL": "embedding_model",
    "MEDCLAIM_EMBEDDING_QUERY_PREFIX": "embedding_query_prefix",
    "MEDCLAIM_RERANKER_PROVIDER": "reranker_provider",
    "MEDCLAIM_RERANKER_MODEL": "reranker_model",
    "MEDCLAIM_RERANKER_BATCH_SIZE": "reranker_batch_size",
    "MEDCLAIM_RETRIEVAL_MODE": "retrieval_mode",
    "MEDCLAIM_RETRIEVAL_CANDIDATE_COUNT": "retrieval_candidate_count",
    "MEDCLAIM_TOP_K": "top_k",
    "MEDCLAIM_GATE_MINIMUM_SCORE": "gate_minimum_score",
    "LOG_LEVEL": "log_level",
    "LOG_FORMAT": "log_format",
}


def load_runtime_settings(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    strict: bool = False,
) -> RuntimeSettings:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError) as exc:
        raise RuntimeConfigurationError(f"CONFIG_INVALID: Could not load {path}: {exc}.") from exc
    if not isinstance(raw, dict):
        raise RuntimeConfigurationError("CONFIG_INVALID: Deployment YAML must contain an object.")
    values: dict[str, Any] = dict(raw)
    source = os.environ if environ is None else environ
    for env_name, field in _ENV_MAP.items():
        if env_name in source and source[env_name] != "":
            if field in {"llm_timeout_seconds", "gate_minimum_score"}:
                try:
                    values[field] = float(source[env_name])
                except ValueError as exc:
                    raise RuntimeConfigurationError(
                        f"CONFIG_INVALID: {env_name} must be a number."
                    ) from exc
            else:
                values[field] = source[env_name]
    try:
        settings = RuntimeSettings.model_validate(values)
    except ValidationError as exc:
        raise RuntimeConfigurationError(f"CONFIG_INVALID: {exc.errors()[0]['msg']}.") from exc
    if settings.embedding_provider != "ollama":
        raise RuntimeConfigurationError("CONFIG_INVALID: embedding_provider must be ollama.")
    if settings.reranker_provider != "ollama":
        raise RuntimeConfigurationError("CONFIG_INVALID: reranker_provider must be ollama.")
    if settings.retrieval_mode not in {"bm25", "dense", "hybrid", "hybrid_reranked"}:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: retrieval_mode must be bm25, dense, hybrid, or hybrid_reranked."
        )
    if settings.top_k > settings.retrieval_candidate_count:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: top_k cannot exceed retrieval_candidate_count."
        )
    if settings.retrieval_mode == "hybrid_reranked" and settings.gate_minimum_score > 1:
        raise RuntimeConfigurationError("CONFIG_INVALID: reranker gate score cannot exceed one.")
    if strict:
        required = [("MEDCLAIM_CORPUS_DIR", settings.corpus_dir), ("LLM_MODEL", settings.llm_model)]
        if settings.retrieval_mode in {"bm25", "hybrid", "hybrid_reranked"}:
            required.append(("MEDCLAIM_BM25_INDEX_DIR", settings.bm25_index_dir))
        if settings.retrieval_mode in {"dense", "hybrid", "hybrid_reranked"}:
            required.extend(
                [
                    ("MEDCLAIM_DENSE_INDEX_DIR", settings.dense_index_dir),
                    ("MEDCLAIM_EMBEDDING_MODEL", settings.embedding_model),
                ]
            )
        if settings.retrieval_mode == "hybrid_reranked":
            required.append(
                (
                    "MEDCLAIM_RERANKER_MODEL",
                    None if settings.reranker_model == "disabled" else settings.reranker_model,
                )
            )
        missing = [name for name, value in required if not value]
        if missing:
            raise RuntimeConfigurationError(f"CONFIG_MISSING_REQUIRED: {missing[0]} is required.")
    return settings
