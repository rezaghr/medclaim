from medclaim.evaluation.retrieval_metrics import (
    any_gold_passage_recall_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
)


def test_complete_single_and_multi_passage_evidence():
    retrieved = ["p1", "p2", "p3"]
    assert complete_evidence_recall_at_k(retrieved, [{"p1"}], 1)
    assert not complete_evidence_recall_at_k(retrieved, [{"p1", "p2"}], 1)
    assert complete_evidence_recall_at_k(retrieved, [{"p1", "p2"}], 2)


def test_alternative_complete_evidence_sets():
    assert complete_evidence_recall_at_k(
        ["p3", "p4"], [{"p1", "p2"}, {"p3"}], 1
    )
    assert not complete_evidence_recall_at_k(["p1"], [set()], 1)


def test_recall_at_requested_depths():
    retrieved = [f"p{index}" for index in range(1, 21)]
    gold_sets = [{"p6"}]
    assert not complete_evidence_recall_at_k(retrieved, gold_sets, 5)
    assert complete_evidence_recall_at_k(retrieved, gold_sets, 10)
    assert complete_evidence_recall_at_k(retrieved, gold_sets, 20)


def test_any_gold_passage_recall():
    retrieved = ["p1", "p2", "p3"]
    assert any_gold_passage_recall_at_k(retrieved, {"p2", "p4"}, 2)
    assert not any_gold_passage_recall_at_k(retrieved, {"p4"}, 3)


def test_first_gold_rank_and_reciprocal_rank():
    assert reciprocal_rank(["p1", "p2", "p3", "gold"], {"gold"}) == (4, 0.25)


def test_no_retrieved_gold_evidence():
    assert reciprocal_rank(["p1", "p2"], {"gold"}) == (None, 0.0)
