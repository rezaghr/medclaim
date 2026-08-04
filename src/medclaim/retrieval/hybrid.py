"""Deterministic BM25 and dense fusion using reciprocal ranks only."""

from __future__ import annotations

import math
import time
from typing import Any


class HybridError(Exception):
    """Raised when hybrid retrieval cannot safely fuse its component results."""


def _validate_positive_limit(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 100
    ):
        raise HybridError(
            f"HYBRID_INVALID_PARAMETER: {name} must be an integer from 1 to 100."
        )


def _validate_rrf_k(rrf_k: int) -> None:
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise HybridError(
            "HYBRID_INVALID_PARAMETER: rrf_k must be a positive integer."
        )


def _component_record(
    result: dict[str, Any],
    *,
    component: str,
) -> tuple[str, int, float]:
    passage_id = result.get("passage_id")
    rank = result.get("rank")
    score_field = "bm25_score" if component == "bm25" else "dense_score"
    score = result.get(score_field)
    if not isinstance(passage_id, str) or not passage_id:
        raise HybridError(
            f"HYBRID_INVALID_RESULT: {component} result has no valid passage_id."
        )
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise HybridError(
            f"HYBRID_INVALID_RESULT: {component} result has an invalid rank."
        )
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise HybridError(
            f"HYBRID_INVALID_RESULT: {component} result has an invalid score."
        )
    return passage_id, rank, float(score)


def reciprocal_rank_fusion(
    sparse_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge component rankings without combining their incomparable raw scores."""
    _validate_rrf_k(rrf_k)
    if not isinstance(sparse_results, list) or not isinstance(dense_results, list):
        raise HybridError("HYBRID_INVALID_RESULT: Component results must be lists.")

    fused_by_id: dict[str, dict[str, Any]] = {}
    seen_by_component: dict[str, set[str]] = {"bm25": set(), "dense": set()}
    for component, results in (("bm25", sparse_results), ("dense", dense_results)):
        for result in results:
            if not isinstance(result, dict):
                raise HybridError(
                    f"HYBRID_INVALID_RESULT: {component} results must be objects."
                )
            passage_id, source_rank, source_score = _component_record(
                result, component=component
            )
            if passage_id in seen_by_component[component]:
                raise HybridError(
                    "HYBRID_DUPLICATE_COMPONENT_RESULT: "
                    f"{component} returned {passage_id} more than once."
                )
            seen_by_component[component].add(passage_id)
            if passage_id not in fused_by_id:
                required = ("document_id", "dataset", "text")
                if any(not isinstance(result.get(field), str) for field in required):
                    raise HybridError(
                        "HYBRID_INVALID_RESULT: Component result is missing provenance."
                    )
                fused_by_id[passage_id] = {
                    "passage_id": passage_id,
                    "document_id": result["document_id"],
                    "dataset": result["dataset"],
                    "text": result["text"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                    "rrf_score": 0.0,
                }
            fused = fused_by_id[passage_id]
            if any(
                fused[field] != result[field]
                for field in ("document_id", "dataset", "text")
            ):
                raise HybridError(
                    "HYBRID_PASSAGE_MISMATCH: Component results disagree on passage data."
                )
            if component == "bm25":
                fused["bm25_rank"] = source_rank
                fused["bm25_score"] = source_score
            else:
                fused["dense_rank"] = source_rank
                fused["dense_score"] = source_score
            fused["rrf_score"] += 1.0 / (rrf_k + source_rank)

    ranked = sorted(
        fused_by_id.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(
                rank
                for rank in (item["bm25_rank"], item["dense_rank"])
                if rank is not None
            ),
            item["passage_id"],
        ),
    )
    return [dict(item, rank=rank) for rank, item in enumerate(ranked, start=1)]


class HybridRetriever:
    """Run BM25 and dense search independently and fuse their rankings with RRF."""

    def __init__(
        self,
        sparse_retriever: Any,
        dense_retriever: Any,
        sparse_top_k: int = 50,
        dense_top_k: int = 50,
        fusion_top_k: int = 30,
        rrf_k: int = 60,
    ) -> None:
        _validate_positive_limit("sparse_top_k", sparse_top_k)
        _validate_positive_limit("dense_top_k", dense_top_k)
        _validate_positive_limit("fusion_top_k", fusion_top_k)
        _validate_rrf_k(rrf_k)
        if fusion_top_k > sparse_top_k + dense_top_k:
            raise HybridError(
                "HYBRID_INVALID_PARAMETER: fusion_top_k exceeds total candidates."
            )
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.fusion_top_k = fusion_top_k
        self.rrf_k = rrf_k
        self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        try:
            sparse_corpus = self.sparse_retriever.index_manifest["corpus"]
            dense_corpus = self.dense_retriever.index_manifest["corpus"]
            sparse_identity = (
                sparse_corpus["version"],
                sparse_corpus["content_hash"],
                sparse_corpus["passage_count"],
            )
            dense_identity = (
                dense_corpus["version"],
                dense_corpus["content_hash"],
                dense_corpus["passage_count"],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise HybridError(
                "HYBRID_INDEX_MANIFEST_INVALID: Component corpus metadata is missing."
            ) from exc
        if sparse_identity != dense_identity:
            raise HybridError(
                "HYBRID_CORPUS_MISMATCH: The BM25 and dense indexes were built "
                "from different corpus artifacts."
            )
        self.corpus_version = sparse_identity[0]
        self.corpus_content_hash = sparse_identity[1]
        self.passage_count = sparse_identity[2]

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        effective_top_k = self.fusion_top_k if top_k is None else top_k
        _validate_positive_limit("top_k", effective_top_k)
        if effective_top_k > self.sparse_top_k + self.dense_top_k:
            raise HybridError(
                "HYBRID_INVALID_PARAMETER: top_k exceeds total component candidates."
            )

        total_started = time.perf_counter()
        sparse_started = time.perf_counter()
        sparse = self.sparse_retriever.search(query, self.sparse_top_k)
        sparse_latency = (time.perf_counter() - sparse_started) * 1000
        dense_started = time.perf_counter()
        dense = self.dense_retriever.search(query, self.dense_top_k)
        dense_latency = (time.perf_counter() - dense_started) * 1000
        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            sparse["results"], dense["results"], self.rrf_k
        )[:effective_top_k]
        for rank, result in enumerate(fused, start=1):
            result["rank"] = rank
        fusion_latency = (time.perf_counter() - fusion_started) * 1000
        total_latency = (time.perf_counter() - total_started) * 1000
        return {
            "query": sparse["query"],
            "retrieval_mode": "hybrid",
            "configuration": {
                "sparse_top_k": self.sparse_top_k,
                "dense_top_k": self.dense_top_k,
                "fusion_top_k": effective_top_k,
                "rrf_k": self.rrf_k,
            },
            "returned_count": len(fused),
            "latency_ms": {
                "bm25": sparse_latency,
                "dense": dense_latency,
                "fusion": fusion_latency,
                "total": total_latency,
            },
            "corpus_version": self.corpus_version,
            "bm25_index_version": self.sparse_retriever.index_manifest[
                "index_version"
            ],
            "dense_index_version": self.dense_retriever.index_manifest[
                "index_version"
            ],
            "results": fused,
        }
