import builtins
import json

import numpy as np
import pytest

from medclaim.reranking.cross_encoder import CrossEncoderReranker, RerankerError


def candidate(passage_id, rank, text=None):
    return {
        "rank": rank,
        "passage_id": passage_id,
        "document_id": f"document:{passage_id}",
        "dataset": "scifact",
        "text": text or f"Evidence {passage_id}",
        "bm25_rank": rank,
        "bm25_score": float(10 - rank),
        "dense_rank": rank,
        "dense_score": float(1 / rank),
        "rrf_score": float(0.1 / rank),
    }


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append(
            {
                "pairs": list(pairs),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


def reranker(model):
    return CrossEncoderReranker(
        "fake/cross-encoder",
        batch_size=2,
        maximum_input_length=128,
        model_revision="test-rev",
        model=model,
    )


def test_batch_alignment_ranking_and_field_preservation():
    model = FakeCrossEncoder([0.2, 0.9, -1.0])
    candidates = [candidate("p1", 1), candidate("p2", 2), candidate("p3", 3)]
    results = reranker(model).rerank("claim text", candidates, 2)
    assert len(model.calls) == 1
    assert model.calls[0] == {
        "pairs": [
            ("claim text", "Evidence p1"),
            ("claim text", "Evidence p2"),
            ("claim text", "Evidence p3"),
        ],
        "batch_size": 2,
        "show_progress_bar": False,
    }
    assert [item["passage_id"] for item in results] == ["p2", "p1"]
    assert [item["rank"] for item in results] == [1, 2]
    assert [item["pre_rerank_rank"] for item in results] == [2, 1]
    assert results[0]["bm25_score"] == candidates[1]["bm25_score"]
    assert results[0]["dense_score"] == candidates[1]["dense_score"]
    assert results[0]["rrf_score"] == candidates[1]["rrf_score"]
    assert type(results[0]["reranker_score"]) is float
    json.dumps(results, allow_nan=False)


def test_deterministic_tie_breaking_uses_pre_rank_then_id():
    candidates = [candidate("p2", 1), candidate("p1", 1), candidate("p3", 2)]
    results = reranker(FakeCrossEncoder([1.0, 1.0, 1.0])).rerank(
        "claim", candidates, 3
    )
    assert [item["passage_id"] for item in results] == ["p1", "p2", "p3"]


def test_empty_candidates_do_not_call_model():
    model = FakeCrossEncoder([])
    assert reranker(model).rerank("", [], 5) == []
    assert model.calls == []


@pytest.mark.parametrize(
    ("scores", "error"),
    [
        ([1.0], "RERANKER_SCORE_COUNT_MISMATCH"),
        ([1.0, np.nan], "RERANKER_INVALID_SCORE"),
        ([1.0, np.inf], "RERANKER_INVALID_SCORE"),
    ],
)
def test_invalid_model_scores(scores, error):
    with pytest.raises(RerankerError, match=error):
        reranker(FakeCrossEncoder(scores)).rerank(
            "claim", [candidate("p1", 1), candidate("p2", 2)], 2
        )


def test_invalid_claim_candidate_and_top_k():
    selected = reranker(FakeCrossEncoder([1.0]))
    with pytest.raises(RerankerError, match="RERANKER_EMPTY_CLAIM"):
        selected.rerank(" ", [candidate("p1", 1)], 1)
    with pytest.raises(RerankerError, match="RERANKER_INVALID_CANDIDATE"):
        selected.rerank("claim", [{"passage_id": "p1"}], 1)
    with pytest.raises(RerankerError, match="RERANKER_INVALID_CONFIGURATION"):
        selected.rerank("claim", [candidate("p1", 1)], 2)
    invalid_score = candidate("p1", 1)
    invalid_score["rrf_score"] = float("nan")
    with pytest.raises(RerankerError, match="RERANKER_INVALID_CANDIDATE"):
        selected.rerank("claim", [invalid_score], 1)


def test_model_prediction_failure_is_controlled():
    with pytest.raises(RerankerError, match="RERANKER_FAILED"):
        reranker(FakeCrossEncoder(RuntimeError("failed"))).rerank(
            "claim", [candidate("p1", 1)], 1
        )


def test_model_load_failure_is_controlled(monkeypatch):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(RerankerError, match="RERANKER_MODEL_LOAD_FAILED"):
        CrossEncoderReranker("fake/model")
