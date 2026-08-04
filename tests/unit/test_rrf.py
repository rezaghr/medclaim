import json

import pytest

from medclaim.retrieval.hybrid import HybridError, reciprocal_rank_fusion


def result(passage_id, rank, score, component):
    return {
        "rank": rank,
        "passage_id": passage_id,
        "document_id": f"document:{passage_id}",
        "dataset": "scifact",
        "text": f"Text for {passage_id}",
        f"{component}_score": score,
    }


def test_rrf_preserves_single_source_and_overlapping_passages():
    sparse = [result("p1", 1, 20.0, "bm25"), result("p2", 2, 10.0, "bm25")]
    dense = [result("p1", 2, 0.8, "dense"), result("p3", 1, 0.9, "dense")]
    fused = reciprocal_rank_fusion(sparse, dense, 60)
    by_id = {item["passage_id"]: item for item in fused}
    assert set(by_id) == {"p1", "p2", "p3"}
    assert by_id["p1"]["bm25_rank"] == 1
    assert by_id["p1"]["dense_rank"] == 2
    assert by_id["p1"]["bm25_score"] == 20.0
    assert by_id["p1"]["dense_score"] == 0.8
    assert by_id["p2"]["dense_rank"] is None
    assert by_id["p2"]["dense_score"] is None
    assert by_id["p3"]["bm25_rank"] is None
    assert by_id["p3"]["bm25_score"] is None
    json.dumps(fused, allow_nan=False)


def test_rrf_formula_and_raw_scores_have_no_influence():
    sparse = [result("p", 2, 1.0, "bm25")]
    dense = [result("p", 5, 0.2, "dense")]
    first = reciprocal_rank_fusion(sparse, dense, 60)
    sparse[0]["bm25_score"] = 999999.0
    dense[0]["dense_score"] = -0.8
    second = reciprocal_rank_fusion(sparse, dense, 60)
    expected = 1 / 62 + 1 / 65
    assert first[0]["rrf_score"] == pytest.approx(expected)
    assert second[0]["rrf_score"] == pytest.approx(expected)
    assert first[0]["rank"] == second[0]["rank"]


def test_deterministic_tie_breaking_uses_best_rank_then_passage_id():
    sparse = [result("p2", 1, 4.0, "bm25")]
    dense = [result("p1", 1, 0.4, "dense")]
    assert [
        item["passage_id"] for item in reciprocal_rank_fusion(sparse, dense)
    ] == ["p1", "p2"]


@pytest.mark.parametrize(
    ("sparse", "dense", "expected"),
    [([], [], []), ([result("p1", 1, 1.0, "bm25")], [], ["p1"]), ([], [result("p2", 1, 0.5, "dense")], ["p2"])],
)
def test_empty_component_lists(sparse, dense, expected):
    assert [
        item["passage_id"] for item in reciprocal_rank_fusion(sparse, dense)
    ] == expected


@pytest.mark.parametrize("rrf_k", [0, -1, True, 1.5])
def test_invalid_rrf_k(rrf_k):
    with pytest.raises(HybridError, match="HYBRID_INVALID_PARAMETER"):
        reciprocal_rank_fusion([], [], rrf_k)


def test_duplicate_component_passage_is_rejected():
    duplicate = result("p1", 1, 1.0, "bm25")
    with pytest.raises(HybridError, match="HYBRID_DUPLICATE_COMPONENT_RESULT"):
        reciprocal_rank_fusion([duplicate, dict(duplicate, rank=2)], [])
