"""Optional Ollama reranking stage over a hybrid candidate pool."""

from __future__ import annotations

import time
from typing import Any

from medclaim.reranking.models import EvidenceReranker, RerankerError, RerankingConfiguration


class RerankedRetrievalError(Exception):
    """Raised when the reranked retrieval stage cannot complete."""


class RerankedRetriever:
    """Retrieve hybrid candidates and optionally rerank the final evidence."""

    def __init__(
        self,
        hybrid_retriever: Any,
        reranker: EvidenceReranker | None,
        configuration: RerankingConfiguration | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.configuration = configuration or RerankingConfiguration()
        if self.configuration.enabled and reranker is None:
            raise RerankedRetrievalError(
                "RERANKER_INVALID_CONFIGURATION: Enabled reranking requires a reranker."
            )
        if reranker is not None and (
            reranker.model_id != self.configuration.model_id
            or reranker.model_revision != self.configuration.model_revision
            or reranker.batch_size != self.configuration.batch_size
        ):
            raise RerankedRetrievalError(
                "RERANKER_INVALID_CONFIGURATION: Reranker metadata does not match settings."
            )

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        final_k = self.configuration.final_evidence_k if top_k is None else top_k
        if (
            not isinstance(final_k, int)
            or isinstance(final_k, bool)
            or not 1 <= final_k <= self.configuration.candidate_count
        ):
            raise RerankedRetrievalError(
                "RERANKER_INVALID_CONFIGURATION: top_k must be between 1 and candidate_count."
            )
        total_started = time.perf_counter()
        hybrid_started = time.perf_counter()
        hybrid = self.hybrid_retriever.search(
            query, top_k=self.configuration.candidate_count
        )
        hybrid_latency = (time.perf_counter() - hybrid_started) * 1000
        candidates = hybrid["results"]
        reranking_started = time.perf_counter()
        if self.configuration.enabled and candidates:
            try:
                assert self.reranker is not None
                results, traced_candidates = self.reranker.rerank(
                    hybrid["query"], candidates, min(final_k, len(candidates))
                )
            except RerankerError as exc:
                raise RerankedRetrievalError(str(exc)) from exc
            retrieval_mode = "hybrid_reranked"
        else:
            results = [dict(candidate) for candidate in candidates[:final_k]]
            traced_candidates = candidates
            retrieval_mode = "hybrid"
        reranking_latency = (time.perf_counter() - reranking_started) * 1000
        total_latency = (time.perf_counter() - total_started) * 1000
        return {
            "query": hybrid["query"],
            "retrieval_mode": retrieval_mode,
            "configuration": {
                "reranking_enabled": self.configuration.enabled,
                "hybrid_candidate_count": self.configuration.candidate_count,
                "final_evidence_k": final_k,
                "reranker_model": self.configuration.model_id,
                "reranker_model_revision": self.configuration.model_revision,
                "reranker_device": (
                    self.reranker.device
                    if self.reranker is not None
                    else self.configuration.device
                ),
                "reranker_batch_size": self.configuration.batch_size,
            },
            "returned_count": len(results),
            "latency_ms": {
                "hybrid_retrieval": hybrid_latency,
                "reranking": reranking_latency,
                "total": total_latency,
            },
            "corpus_version": hybrid["corpus_version"],
            "bm25_index_version": hybrid["bm25_index_version"],
            "dense_index_version": hybrid["dense_index_version"],
            "candidate_results": [dict(candidate) for candidate in traced_candidates],
            "results": results,
        }
