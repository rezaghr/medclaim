import json

import pytest

from medclaim.reranking.models import RerankingConfiguration
from medclaim.retrieval.reranked import RerankedRetrievalError, RerankedRetriever

from tests.unit.test_cross_encoder_reranker import candidate


class FakeHybrid:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return {
            "query": query.strip(),
            "corpus_version": "corpus-v1",
            "bm25_index_version": "bm25-v1",
            "dense_index_version": "dense-v1",
            "results": self.candidates[:top_k],
        }


class FakeReranker:
    model_id = "fake/reranker"
    model_revision = "rev"
    device = "cpu"
    batch_size = 4
    maximum_input_length = 128

    def __init__(self):
        self.calls = []

    def rerank(self, claim, candidates, top_k):
        self.calls.append((claim, candidates, top_k))
        outputs = []
        for rank, item in enumerate(reversed(candidates), start=1):
            output = dict(item)
            output["pre_rerank_rank"] = item["rank"]
            output["reranker_score"] = float(rank)
            output["rank"] = rank
            outputs.append(output)
        return outputs[:top_k]


def configuration(enabled=True):
    return RerankingConfiguration(
        enabled=enabled,
        model_id="fake/reranker",
        model_revision="rev",
        candidate_count=3,
        final_evidence_k=2,
        batch_size=4,
        device="cpu",
        maximum_input_length=128,
    )


def test_enabled_reranking_result_and_latency_contract():
    hybrid = FakeHybrid([candidate("p1", 1), candidate("p2", 2), candidate("p3", 3)])
    selected = FakeReranker()
    result = RerankedRetriever(hybrid, selected, configuration()).search(" claim ")
    assert hybrid.calls == [(" claim ", 3)]
    assert len(selected.calls) == 1
    assert result["retrieval_mode"] == "hybrid_reranked"
    assert result["returned_count"] == 2
    assert [row["passage_id"] for row in result["candidate_results"]] == ["p1", "p2", "p3"]
    assert set(result["latency_ms"]) == {"hybrid_retrieval", "reranking", "total"}
    assert result["configuration"]["reranker_model"] == "fake/reranker"
    json.dumps(result, allow_nan=False)


def test_disabled_reranking_uses_first_hybrid_passages_without_model():
    hybrid = FakeHybrid([candidate("p1", 1), candidate("p2", 2), candidate("p3", 3)])
    result = RerankedRetriever(hybrid, None, configuration(False)).search("claim")
    assert result["retrieval_mode"] == "hybrid"
    assert [item["passage_id"] for item in result["results"]] == ["p1", "p2"]


def test_empty_hybrid_results_skip_reranker():
    selected = FakeReranker()
    result = RerankedRetriever(FakeHybrid([]), selected, configuration()).search("claim")
    assert result["results"] == []
    assert selected.calls == []


def test_enabled_mode_requires_matching_reranker():
    with pytest.raises(RerankedRetrievalError, match="RERANKER_INVALID_CONFIGURATION"):
        RerankedRetriever(FakeHybrid([]), None, configuration())
