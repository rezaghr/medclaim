import json

import pytest

from medclaim.reranking.cross_encoder import RerankerError
from medclaim.reranking.ollama import OllamaEvidenceReranker


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, *, prompt, response_schema):
        self.calls.append((prompt, response_schema))
        return json.dumps(self.response)


def candidates():
    return [
        {"rank": 1, "passage_id": "p1", "text": "Generic tamoxifen treatment."},
        {"rank": 2, "passage_id": "p2", "text": "CYP2D6 changes tamoxifen metabolism."},
    ]


def test_ollama_reranker_scores_and_reorders_candidates():
    provider = FakeProvider(
        {
            "scores": [
                {"passage_id": "p1", "relevance_score": 0.2},
                {"passage_id": "p2", "relevance_score": 0.9},
            ]
        }
    )
    reranker = OllamaEvidenceReranker(provider, model_id="dolphin", batch_size=2)
    result = reranker.rerank("CYP2D6 affects tamoxifen metabolism", candidates(), 2)
    assert [row["passage_id"] for row in result] == ["p2", "p1"]
    assert result[0]["reranker_score"] == 0.9
    assert result[0]["pre_rerank_rank"] == 2
    assert "Generic discussion" in provider.calls[0][0]


def test_ollama_reranker_rejects_missing_candidate_score():
    provider = FakeProvider(
        {"scores": [{"passage_id": "p1", "relevance_score": 0.2}]}
    )
    reranker = OllamaEvidenceReranker(provider, model_id="dolphin")
    with pytest.raises(RerankerError, match="RERANKER_INVALID_SCORE"):
        reranker.rerank("claim", candidates(), 1)


def test_ollama_reranker_subdivides_incomplete_batches():
    class SubdividingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, *, prompt, response_schema):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"scores": []})
            passage_id = "p1" if 'passage_id="p1"' in prompt else "p2"
            return json.dumps(
                {"scores": [{"passage_id": passage_id, "relevance_score": 0.8}]}
            )

    provider = SubdividingProvider()
    reranker = OllamaEvidenceReranker(provider, model_id="dolphin", batch_size=2)
    result = reranker.rerank("claim", candidates(), 2)
    assert provider.calls == 3
    assert {row["passage_id"] for row in result} == {"p1", "p2"}
