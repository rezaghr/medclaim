"""Offline evaluation helpers for MedClaim retrieval experiments."""

from .bm25_evaluation import EvaluationError, evaluate_bm25
from .dense_evaluation import evaluate_dense
from .classification_metrics import classification_metrics
from .gate_decomposition import evaluate_gate_and_decomposition
from .reranking_comparison import compare_reranking
from .retrieval_comparison import compare_retrieval
from .retrieval_metrics import (
    any_gold_passage_recall_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvaluationError",
    "any_gold_passage_recall_at_k",
    "complete_evidence_recall_at_k",
    "compare_retrieval",
    "compare_reranking",
    "classification_metrics",
    "evaluate_gate_and_decomposition",
    "evaluate_bm25",
    "evaluate_dense",
    "reciprocal_rank",
]
