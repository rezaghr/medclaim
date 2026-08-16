"""Construct the production verification pipeline from versioned artifacts."""

from __future__ import annotations

from typing import Any

from medclaim.evidence_gate import EvidenceGate, EvidenceGateConfiguration
from medclaim.reranking import (
    OllamaEvidenceReranker,
    RerankingConfiguration,
)
from medclaim.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
)
from medclaim.retrieval.embedding import OllamaEmbedder
from medclaim.security import SecureVerifier
from medclaim.verification import VerificationPipeline

from .configuration import RuntimeSettings
from .ollama import OllamaProvider


class RuntimePipelineError(Exception):
    """Raised when a configured runtime pipeline cannot be assembled safely."""


def build_runtime_pipeline(settings: RuntimeSettings):
    if settings.corpus_dir is None:
        raise RuntimePipelineError("PIPELINE_NOT_READY: Corpus directory is not configured.")

    sparse = None
    dense = None
    if settings.retrieval_mode in {"bm25", "hybrid", "hybrid_reranked"}:
        if settings.bm25_index_dir is None:
            raise RuntimePipelineError("PIPELINE_NOT_READY: BM25 index is not configured.")
        sparse = BM25Retriever(settings.bm25_index_dir, settings.corpus_dir)
    if settings.retrieval_mode in {"dense", "hybrid", "hybrid_reranked"}:
        if settings.dense_index_dir is None:
            raise RuntimePipelineError("PIPELINE_NOT_READY: Dense index is not configured.")
        embedder = None
        if settings.embedding_provider == "ollama":
            embedder = OllamaEmbedder(
                settings.embedding_model,
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
                input_prefix=settings.embedding_query_prefix,
            )
        dense = DenseRetriever(
            settings.dense_index_dir,
            settings.corpus_dir,
            embedder=embedder,
            corpus_data=(sparse.corpus_manifest, sparse.passages) if sparse else None,
        )

    retriever: Any
    if settings.retrieval_mode == "bm25":
        retriever = sparse
        score_field = "bm25_score"
    elif settings.retrieval_mode == "dense":
        retriever = dense
        score_field = "dense_score"
    elif settings.retrieval_mode in {"hybrid", "hybrid_reranked"}:
        retriever = HybridRetriever(sparse, dense)
        score_field = "rrf_score"
    else:
        raise RuntimePipelineError(f"PIPELINE_RETRIEVAL_UNSUPPORTED: {settings.retrieval_mode!r}.")

    provider = OllamaProvider(
        settings.llm_model,
        settings.ollama_base_url,
        settings.llm_timeout_seconds,
    )
    if settings.retrieval_mode == "hybrid_reranked":
        reranker_provider = (
            provider
            if settings.reranker_model == settings.llm_model
            else OllamaProvider(
                settings.reranker_model,
                settings.ollama_base_url,
                settings.llm_timeout_seconds,
            )
        )
        reranking = RerankingConfiguration(
            enabled=True,
            model_id=settings.reranker_model,
            candidate_count=settings.retrieval_candidate_count,
            final_evidence_k=settings.top_k,
            batch_size=settings.reranker_batch_size,
            device="ollama",
        )
        reranker = OllamaEvidenceReranker(
            reranker_provider,
            model_id=settings.reranker_model,
            batch_size=reranking.batch_size,
        )
        retriever = RerankedRetriever(retriever, reranker, reranking)
        score_field = "reranker_score"
    verifier = SecureVerifier(provider)
    gate = EvidenceGate(
        EvidenceGateConfiguration(
            version=settings.gate_version,
            enabled=True,
            minimum_score=settings.gate_minimum_score,
            minimum_relevant_passages=settings.gate_minimum_relevant_passages,
            minimum_unique_documents=settings.gate_minimum_unique_documents,
            score_field=score_field,
        )
    )
    return VerificationPipeline(
        retriever,
        verifier,
        gate,
        final_evidence_k=settings.top_k,
    )
