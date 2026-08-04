"""Strict versioned non-secret configuration with environment-only secrets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RuntimeConfigurationError(Exception):
    """Raised when deployment configuration is unknown, unsafe, or incomplete."""


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_id: str = "deployment-v1"
    environment: str = "development"
    llm_provider: str = "fake"
    llm_model: str = "fake-verifier-v1"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    corpus_dir: Path | None = None
    bm25_index_dir: Path | None = None
    dense_index_dir: Path | None = None
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_query_prefix: str = ""
    reranker_provider: str = "cross_encoder"
    reranker_model: str = "disabled"
    reranker_batch_size: int = Field(default=8, ge=1, le=30)
    retrieval_mode: str = "hybrid"
    retrieval_candidate_count: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=100)
    gate_minimum_score: float = Field(default=0.7, ge=0)
    gate_minimum_relevant_passages: int = Field(default=1, ge=1, le=100)
    gate_minimum_unique_documents: int = Field(default=1, ge=1, le=100)
    prompt_version: str = "medical-verifier-v3-secure"
    gate_version: str = "evidence-gate-v1"
    calibrator_version: str = "confidence-calibrator-v1"
    persistence_enabled: bool = False
    persist_claim_text: bool = False
    persist_explanation: bool = False
    database_url_configured: bool = False
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
    "LLM_PROVIDER": "llm_provider",
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
    "PERSISTENCE_ENABLED": "persistence_enabled",
    "PERSIST_CLAIM_TEXT": "persist_claim_text",
    "PERSIST_EXPLANATION": "persist_explanation",
    "LOG_LEVEL": "log_level",
    "LOG_FORMAT": "log_format",
}
_BOOLEAN_FIELDS = {"persistence_enabled", "persist_claim_text", "persist_explanation"}


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"true", "false"}:
        raise RuntimeConfigurationError(f"CONFIG_INVALID: {name} must be true or false.")
    return normalized == "true"


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
    if "database_url_configured" in raw:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: database_url_configured is derived from the environment and cannot appear in YAML."
        )
    values: dict[str, Any] = dict(raw)
    source = os.environ if environ is None else environ
    for env_name, field in _ENV_MAP.items():
        if env_name in source and source[env_name] != "":
            if field in _BOOLEAN_FIELDS:
                values[field] = _boolean(source[env_name], env_name)
            elif field in {"llm_timeout_seconds", "gate_minimum_score"}:
                try:
                    values[field] = float(source[env_name])
                except ValueError as exc:
                    raise RuntimeConfigurationError(
                        f"CONFIG_INVALID: {env_name} must be a number."
                    ) from exc
            else:
                values[field] = source[env_name]
    values["database_url_configured"] = bool(source.get("DATABASE_URL", ""))
    try:
        settings = RuntimeSettings.model_validate(values)
    except ValidationError as exc:
        raise RuntimeConfigurationError(f"CONFIG_INVALID: {exc.errors()[0]['msg']}.") from exc
    if settings.llm_provider not in {"fake", "ollama"}:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: llm_provider must be fake or ollama."
        )
    if settings.embedding_provider not in {"sentence_transformers", "ollama"}:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: embedding_provider must be sentence_transformers or ollama."
        )
    if settings.reranker_provider not in {"cross_encoder", "ollama"}:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: reranker_provider must be cross_encoder or ollama."
        )
    if settings.retrieval_mode not in {"bm25", "dense", "hybrid", "hybrid_reranked"}:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: retrieval_mode must be bm25, dense, hybrid, or hybrid_reranked."
        )
    if settings.top_k > settings.retrieval_candidate_count:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: top_k cannot exceed retrieval_candidate_count."
        )
    if settings.retrieval_mode == "hybrid_reranked" and settings.gate_minimum_score > 1:
        raise RuntimeConfigurationError(
            "CONFIG_INVALID: reranker gate score cannot exceed one."
        )
    if settings.persist_claim_text or settings.persist_explanation:
        if not settings.persistence_enabled:
            raise RuntimeConfigurationError(
                "CONFIG_PRIVACY_INVALID: Content persistence requires persistence_enabled."
            )
    if settings.persistence_enabled and not settings.database_url_configured:
        raise RuntimeConfigurationError(
            "CONFIG_MISSING_SECRET: DATABASE_URL is required when persistence is enabled."
        )
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
        if settings.llm_provider not in {"fake", "ollama"} and not source.get("LLM_API_KEY"):
            missing.append("LLM_API_KEY")
        if missing:
            raise RuntimeConfigurationError(f"CONFIG_MISSING_REQUIRED: {missing[0]} is required.")
    return settings
