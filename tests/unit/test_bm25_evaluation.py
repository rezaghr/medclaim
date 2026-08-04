import json

import pytest

import medclaim.evaluation.bm25_evaluation as evaluation_module
from medclaim.evaluation.bm25_evaluation import EvaluationError, evaluate_bm25
from medclaim.retrieval.bm25 import BM25Error


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def claim(claim_id, text, split="dev", label="SUPPORTS"):
    return {
        "claim_id": claim_id,
        "dataset": "scifact",
        "claim_text": text,
        "original_split": split,
        "unified_label": label,
    }


def gold(claim_id, sets, split="dev"):
    return {
        "claim_id": claim_id,
        "dataset": "scifact",
        "original_split": split,
        "unified_label": "SUPPORTS",
        "evidence_sets": [
            {"evidence_set_id": f"{claim_id}:e:{index}", "passage_ids": ids}
            for index, ids in enumerate(sets)
        ],
    }


class FakeRetriever:
    def __init__(self, index_dir, corpus_dir):
        self.corpus_manifest = {
            "corpus_version": "corpus-v1",
            "content_hash": "sha256:" + "a" * 64,
        }
        self.index_manifest = {
            "index_version": "index-v1",
            "configuration": {
                "tokenizer": "simple-alphanumeric-v1",
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
            },
        }
        self.passages_by_id = {f"p{index}": {} for index in range(1, 31)}

    def search(self, query, top_k):
        if query == "partial":
            ids = ["p1"] + [f"p{index}" for index in range(3, 30)]
        elif query == "miss":
            ids = [f"p{index}" for index in range(3, 30)]
        elif query == "rank four":
            ids = ["p3", "p4", "p5", "p1"] + [
                f"p{index}" for index in range(6, 30)
            ]
        else:
            ids = ["p1", "p2"] + [f"p{index}" for index in range(3, 30)]
        results = [
            {
                "rank": rank,
                "passage_id": passage_id,
                "document_id": f"document:{passage_id}",
                "bm25_score": float(top_k - rank),
            }
            for rank, passage_id in enumerate(ids[:top_k], start=1)
        ]
        return {"results": results, "latency_ms": 2.0}


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch):
    monkeypatch.setattr(evaluation_module, "BM25Retriever", FakeRetriever)


def run_evaluation(tmp_path, claims, gold_records, ks=None, output_name="experiment"):
    claims_path = tmp_path / f"{output_name}-claims.jsonl"
    gold_path = tmp_path / f"{output_name}-gold.jsonl"
    write_jsonl(claims_path, claims)
    write_jsonl(gold_path, gold_records)
    output_dir = tmp_path / output_name
    metrics = evaluate_bm25(
        claims_path,
        gold_path,
        tmp_path / "corpus",
        tmp_path / "index",
        "dev",
        output_dir,
        [5, 10, 20] if ks is None else ks,
    )
    return metrics, output_dir


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_split_filtering_and_transparent_exclusions(tmp_path):
    claims = [
        claim("claim:3", "train", "train"),
        claim("claim:1", "complete"),
        claim("claim:2", "no evidence"),
        claim("claim:4", "unlabeled", label=None),
    ]
    metrics, output_dir = run_evaluation(
        tmp_path, claims, [gold("claim:1", [["p1"]])]
    )
    assert metrics["total_claims_in_split"] == 3
    assert metrics["evaluated_claims"] == 1
    assert metrics["excluded_claims"] == 2
    assert metrics["exclusion_reasons"] == {
        "no_gold_evidence": 1,
        "unlabeled": 1,
    }
    assert [record["claim_id"] for record in read_jsonl(output_dir / "predictions.jsonl")] == [
        "claim:1"
    ]


def test_metrics_serialization_and_first_gold_rank(tmp_path):
    metrics, output_dir = run_evaluation(
        tmp_path,
        [claim("claim:1", "rank four")],
        [gold("claim:1", [["p1"]])],
    )
    prediction = read_jsonl(output_dir / "predictions.jsonl")[0]
    assert prediction["first_gold_rank"] == 4
    assert prediction["reciprocal_rank"] == 0.25
    assert metrics["mrr"] == 0.25
    json.dumps(metrics, allow_nan=False)


def test_retrieval_error_classification(tmp_path):
    claims = [claim("claim:2", "miss"), claim("claim:1", "partial")]
    gold_records = [
        gold("claim:1", [["p1", "p2"]]),
        gold("claim:2", [["p1"]]),
    ]
    _, output_dir = run_evaluation(tmp_path, claims, gold_records)
    errors = read_jsonl(output_dir / "retrieval_errors.jsonl")
    assert [error["claim_id"] for error in errors] == ["claim:1", "claim:2"]
    assert [error["error_type"] for error in errors] == [
        "PARTIAL_EVIDENCE_SET_RETRIEVED",
        "NO_GOLD_PASSAGE_IN_TOP_K",
    ]


def test_alternative_evidence_set_is_complete(tmp_path):
    _, output_dir = run_evaluation(
        tmp_path,
        [claim("claim:1", "complete")],
        [gold("claim:1", [["p8", "p9"], ["p1"]])],
    )
    prediction = read_jsonl(output_dir / "predictions.jsonl")[0]
    assert prediction["complete_evidence_recall"] == {
        "5": True,
        "10": True,
        "20": True,
    }


@pytest.mark.parametrize("ks", [[], [0], [101], [5, 5], [True]])
def test_invalid_k_values(tmp_path, ks):
    with pytest.raises(EvaluationError, match="EVALUATION_INVALID_K"):
        run_evaluation(
            tmp_path,
            [claim("claim:1", "complete")],
            [gold("claim:1", [["p1"]])],
            ks=ks,
        )


def test_duplicate_claim_ids(tmp_path):
    duplicate = claim("claim:1", "complete")
    with pytest.raises(EvaluationError, match="EVALUATION_DUPLICATE_CLAIM_ID"):
        run_evaluation(
            tmp_path,
            [duplicate, duplicate],
            [gold("claim:1", [["p1"]])],
        )


def test_unknown_claim_in_gold_mapping(tmp_path):
    with pytest.raises(EvaluationError, match="EVALUATION_UNKNOWN_CLAIM"):
        run_evaluation(
            tmp_path,
            [claim("claim:1", "complete")],
            [gold("claim:1", [["p1"]]), gold("claim:unknown", [["p2"]])],
        )


def test_prediction_order_is_deterministic(tmp_path):
    claims = [claim("claim:2", "complete"), claim("claim:1", "complete")]
    gold_records = [gold("claim:2", [["p2"]]), gold("claim:1", [["p1"]])]
    _, output_dir = run_evaluation(tmp_path, claims, gold_records)
    assert [record["claim_id"] for record in read_jsonl(output_dir / "predictions.jsonl")] == [
        "claim:1",
        "claim:2",
    ]


def test_existing_output_is_immutable(tmp_path):
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    marker = output_dir / "marker"
    marker.write_text("preserve")
    with pytest.raises(EvaluationError, match="EVALUATION_OUTPUT_EXISTS"):
        evaluate_bm25(
            tmp_path / "claims",
            tmp_path / "gold",
            tmp_path / "corpus",
            tmp_path / "index",
            "dev",
            output_dir,
            [5],
        )
    assert marker.read_text() == "preserve"


def test_corpus_index_compatibility_failure_is_controlled(tmp_path, monkeypatch):
    class FailingRetriever:
        def __init__(self, index_dir, corpus_dir):
            raise BM25Error("BM25_INDEX_CORPUS_MISMATCH: incompatible")

    monkeypatch.setattr(evaluation_module, "BM25Retriever", FailingRetriever)
    with pytest.raises(EvaluationError, match="BM25_INDEX_CORPUS_MISMATCH"):
        evaluate_bm25(
            tmp_path / "claims",
            tmp_path / "gold",
            tmp_path / "corpus",
            tmp_path / "index",
            "dev",
            tmp_path / "output",
            [5],
        )
