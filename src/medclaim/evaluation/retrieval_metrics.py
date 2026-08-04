"""Small, deterministic retrieval metrics for gold evidence passages."""

from __future__ import annotations


def complete_evidence_recall_at_k(
    retrieved_ids: list[str],
    gold_evidence_sets: list[set[str]],
    k: int,
) -> bool:
    """Return whether top-k contains every passage in any non-empty gold set."""
    top_k = set(retrieved_ids[:k])
    return any(
        evidence_set and evidence_set.issubset(top_k)
        for evidence_set in gold_evidence_sets
    )


def any_gold_passage_recall_at_k(
    retrieved_ids: list[str],
    gold_passage_ids: set[str],
    k: int,
) -> bool:
    """Return whether top-k contains at least one gold passage."""
    return bool(set(retrieved_ids[:k]) & gold_passage_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    gold_passage_ids: set[str],
) -> tuple[int | None, float]:
    """Return the one-based first-gold rank and its reciprocal."""
    for rank, passage_id in enumerate(retrieved_ids, start=1):
        if passage_id in gold_passage_ids:
            return rank, 1.0 / rank
    return None, 0.0
