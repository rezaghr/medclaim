import json

import pytest

from medclaim.retrieval.hybrid import HybridError, HybridRetriever


def component_result(passage_id, rank, component):
    score_name = "bm25_score" if component == "bm25" else "dense_score"
    return {
        "rank": rank,
        "passage_id": passage_id,
        "document_id": f"document:{passage_id}",
        "dataset": "scifact",
        "text": f"Evidence {passage_id}",
        score_name: float(10 - rank),
    }


class FakeRetriever:
    def __init__(self, component, passage_ids, corpus=None):
        self.component = component
        self.passage_ids = passage_ids
        self.calls = []
        self.index_manifest = {
            "index_version": f"{component}-v1",
            "corpus": corpus
            or {
                "version": "corpus-v1",
                "content_hash": "sha256:" + "a" * 64,
                "passage_count": 4,
            },
        }

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        results = [
            component_result(passage_id, rank, self.component)
            for rank, passage_id in enumerate(self.passage_ids[:top_k], start=1)
        ]
        return {"query": query.strip(), "results": results, "latency_ms": 1.0}


def test_hybrid_runs_independently_truncates_and_preserves_structure():
    sparse = FakeRetriever("bm25", ["p1", "p2", "p3"])
    dense = FakeRetriever("dense", ["p2", "p4", "p3"])
    retriever = HybridRetriever(sparse, dense, 3, 3, 2, 60)
    result = retriever.search("  claim  ")
    assert sparse.calls == [("  claim  ", 3)]
    assert dense.calls == [("  claim  ", 3)]
    assert result["returned_count"] == 2
    assert [item["rank"] for item in result["results"]] == [1, 2]
    assert result["results"][0]["passage_id"] == "p2"
    assert set(result["latency_ms"]) == {"bm25", "dense", "fusion", "total"}
    assert all(value >= 0 for value in result["latency_ms"].values())
    assert result["configuration"] == {
        "sparse_top_k": 3,
        "dense_top_k": 3,
        "fusion_top_k": 2,
        "rrf_k": 60,
    }
    assert result["corpus_version"] == "corpus-v1"
    json.dumps(result, allow_nan=False)


def test_search_top_k_override_and_streamlit_ready_evidence():
    sparse = FakeRetriever("bm25", ["p1", "p2"])
    dense = FakeRetriever("dense", ["p2", "p3"])
    result = HybridRetriever(sparse, dense, 2, 2, 3).search("claim", top_k=1)
    assert result["returned_count"] == 1
    assert result["configuration"]["fusion_top_k"] == 1
    evidence = result["results"][0]
    assert all(
        field in evidence
        for field in (
            "rank",
            "passage_id",
            "document_id",
            "dataset",
            "text",
            "bm25_rank",
            "bm25_score",
            "dense_rank",
            "dense_score",
            "rrf_score",
        )
    )


def test_hybrid_evidence_can_feed_a_fake_verifier():
    sparse = FakeRetriever("bm25", ["p1", "p2"])
    dense = FakeRetriever("dense", ["p2", "p3"])
    results = HybridRetriever(sparse, dense, 2, 2, 3).search("claim")["results"]

    def fake_verifier(claim, evidence):
        return {
            "claim": claim,
            "evidence_used": [item["passage_id"] for item in evidence],
            "texts": [item["text"] for item in evidence],
        }

    verified = fake_verifier("claim", results[:2])
    assert verified["evidence_used"] == ["p2", "p1"]
    assert all(verified["texts"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"sparse_top_k": 0},
        {"dense_top_k": 101},
        {"fusion_top_k": 0},
        {"rrf_k": 0},
        {"sparse_top_k": 1, "dense_top_k": 1, "fusion_top_k": 3},
    ],
)
def test_invalid_hybrid_parameters(overrides):
    parameters = {
        "sparse_top_k": 2,
        "dense_top_k": 2,
        "fusion_top_k": 2,
        "rrf_k": 60,
    }
    parameters.update(overrides)
    with pytest.raises(HybridError, match="HYBRID_INVALID_PARAMETER"):
        HybridRetriever(
            FakeRetriever("bm25", []), FakeRetriever("dense", []), **parameters
        )


@pytest.mark.parametrize("field", ["version", "content_hash", "passage_count"])
def test_component_corpus_mismatch(field):
    sparse = FakeRetriever("bm25", [])
    changed = dict(sparse.index_manifest["corpus"])
    changed[field] = "different" if field != "passage_count" else 5
    dense = FakeRetriever("dense", [], corpus=changed)
    with pytest.raises(HybridError, match="HYBRID_CORPUS_MISMATCH"):
        HybridRetriever(sparse, dense)
