"""Artifact and provider compatibility checks for deployment readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medclaim.corpus.scifact_corpus import corpus_content_hash

from .configuration import RuntimeSettings
from .ollama import OllamaProvider


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def readiness_snapshot(settings: RuntimeSettings) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    def record(name: str, operation) -> Any:
        try:
            detail = operation()
            checks[name] = {"ready": True, "detail": detail}
            return detail
        except Exception as exc:
            checks[name] = {"ready": False, "error": str(exc)}
            errors.append(name)
            return None

    def corpus_check() -> dict[str, Any]:
        if settings.corpus_dir is None:
            raise ValueError("MEDCLAIM_CORPUS_DIR is not configured")
        manifest = _json(settings.corpus_dir / "manifest.json")
        if manifest.get("artifact_type") != "medical_evidence_corpus":
            raise ValueError("corpus artifact_type is invalid")
        passages_path = settings.corpus_dir / "passages.jsonl"
        with passages_path.open(encoding="utf-8") as handle:
            passages = [json.loads(line) for line in handle if line.strip()]
        if len(passages) != manifest.get("passage_count"):
            raise ValueError("corpus passage count does not match its manifest")
        content_hash = manifest.get("content_hash")
        if not isinstance(content_hash, str) or corpus_content_hash(passages) != content_hash:
            raise ValueError("corpus checksum does not match passages")
        datasets = set(manifest.get("datasets", []))
        if datasets != {"scifact", "healthver", "pubhealth"}:
            raise ValueError("corpus datasets are not the supported medical set")
        return manifest

    corpus = record("corpus", corpus_check)

    def index_check(path: Path | None, artifact_type: str) -> dict[str, Any]:
        if path is None:
            raise ValueError(f"{artifact_type} directory is not configured")
        manifest = _json(path / "manifest.json")
        if manifest.get("artifact_type") != artifact_type:
            raise ValueError(f"expected {artifact_type} manifest")
        reference = manifest.get("corpus")
        if not isinstance(reference, dict) or corpus is None:
            raise ValueError("index has no corpus reference")
        if (
            reference.get("version") != corpus.get("corpus_version")
            or reference.get("content_hash") != corpus.get("content_hash")
            or reference.get("passage_count") != corpus.get("passage_count")
        ):
            raise ValueError("index references an incompatible corpus")
        return manifest

    if settings.retrieval_mode in {"bm25", "hybrid", "hybrid_reranked"}:
        record("bm25_index", lambda: index_check(settings.bm25_index_dir, "bm25_index"))
    else:
        checks["bm25_index"] = {"ready": True, "detail": "not_required"}
    if settings.retrieval_mode in {"dense", "hybrid", "hybrid_reranked"}:
        dense = record("dense_index", lambda: index_check(settings.dense_index_dir, "dense_index"))
    else:
        dense = None
        checks["dense_index"] = {"ready": True, "detail": "not_required"}

    def model_check() -> dict[str, str]:
        details: dict[str, str] = {}
        if dense is not None:
            embedding = dense.get("embedding")
            if (
                not isinstance(embedding, dict)
                or not isinstance(embedding.get("dimension"), int)
                or embedding["dimension"] < 1
            ):
                raise ValueError("dense embedding dimensions are invalid")
            if settings.embedding_provider == "ollama":
                details["embedding"] = OllamaProvider(
                    settings.embedding_model,
                    settings.ollama_base_url,
                    settings.llm_timeout_seconds,
                ).check()
        if settings.retrieval_mode == "hybrid_reranked" and (
            not settings.reranker_model or settings.reranker_model == "disabled"
        ):
            raise ValueError("reranker model is not configured")
        if settings.retrieval_mode == "hybrid_reranked":
            if settings.reranker_provider == "ollama":
                details["reranker"] = OllamaProvider(
                    settings.reranker_model,
                    settings.ollama_base_url,
                    settings.llm_timeout_seconds,
                ).check()
            else:
                details["reranker"] = settings.reranker_model
        return details or {"models": "not_required"}

    record("models", model_check)
    def verifier_check() -> str:
        if not settings.llm_provider or not settings.llm_model:
            raise ValueError("verifier provider/model is not configured")
        if settings.llm_provider == "ollama":
            return OllamaProvider(
                settings.llm_model,
                settings.ollama_base_url,
                settings.llm_timeout_seconds,
            ).check()
        return settings.llm_model

    record("verifier", verifier_check)
    record(
        "gate_calibrator",
        lambda: (
            {"gate": settings.gate_version, "calibrator": settings.calibrator_version}
            if settings.gate_version and settings.calibrator_version
            else (_ for _ in ()).throw(ValueError("gate/calibrator versions are incompatible"))
        ),
    )
    record(
        "persistence",
        lambda: (
            "disabled"
            if not settings.persistence_enabled
            else (
                "configured"
                if settings.database_url_configured
                else (_ for _ in ()).throw(ValueError("persistence schema unavailable"))
            )
        ),
    )
    return {
        "status": "ready" if not errors else "not_ready",
        "checks": checks,
        "failed_checks": errors,
    }
