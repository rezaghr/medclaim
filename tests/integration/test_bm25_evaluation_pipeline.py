import json
import shutil
from pathlib import Path

import pytest

from medclaim.evaluation.bm25_evaluation import EvaluationError, evaluate_bm25
from medclaim.retrieval.bm25 import build_bm25_index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bm25_corpus"


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_bm25_evaluation_pipeline_end_to_end(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shutil.copy2(FIXTURES / "passages.jsonl", corpus_dir / "passages.jsonl")
    shutil.copy2(FIXTURES / "manifest.json", corpus_dir / "manifest.json")
    index_dir = build_bm25_index(corpus_dir, tmp_path / "indexes", "index-v1")

    claims_path = tmp_path / "claims.jsonl"
    gold_path = tmp_path / "gold_evidence.jsonl"
    claims = [
        {
            "claim_id": "scifact:claim:2",
            "dataset": "scifact",
            "claim_text": "This claim has no evidence.",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
        },
        {
            "claim_id": "scifact:claim:1",
            "dataset": "scifact",
            "claim_text": "Hydroxychloroquine viral infection",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
        },
        {
            "claim_id": "scifact:claim:3",
            "dataset": "scifact",
            "claim_text": "Vitamin D",
            "original_split": "train",
            "unified_label": "SUPPORTS",
        },
    ]
    gold_records = [
        {
            "claim_id": "scifact:claim:1",
            "dataset": "scifact",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
            "evidence_sets": [
                {
                    "evidence_set_id": "scifact:claim:1:evidence:0",
                    "passage_ids": ["scifact:document:20:p:0"],
                }
            ],
        }
    ]
    write_jsonl(claims_path, claims)
    write_jsonl(gold_path, gold_records)

    first_dir = tmp_path / "experiment-a"
    second_dir = tmp_path / "experiment-b"
    first_metrics = evaluate_bm25(
        claims_path,
        gold_path,
        corpus_dir,
        index_dir,
        "dev",
        first_dir,
        [20, 5, 10],
    )
    second_metrics = evaluate_bm25(
        claims_path,
        gold_path,
        corpus_dir,
        index_dir,
        "dev",
        second_dir,
        [5, 10, 20],
    )

    assert {path.name for path in first_dir.iterdir()} == {
        "predictions.jsonl",
        "metrics.json",
        "retrieval_errors.jsonl",
        "manifest.json",
    }
    prediction = read_jsonl(first_dir / "predictions.jsonl")[0]
    assert prediction["claim_id"] == "scifact:claim:1"
    assert prediction["retrieved"][0]["passage_id"] == "scifact:document:20:p:0"
    assert prediction["complete_evidence_recall"] == {
        "5": True,
        "10": True,
        "20": True,
    }
    assert first_metrics["complete_evidence_recall_at_k"] == {
        "5": 1.0,
        "10": 1.0,
        "20": 1.0,
    }
    assert first_metrics["excluded_claims"] == 1
    assert read_jsonl(first_dir / "retrieval_errors.jsonl") == []
    json.dumps(first_metrics, allow_nan=False)

    manifest = json.loads((first_dir / "manifest.json").read_text())
    assert manifest["artifact_type"] == "retrieval_evaluation"
    assert manifest["ks"] == [5, 10, 20]
    assert manifest["corpus_version"] == "bm25-fixture-v1"
    assert manifest["index_version"] == "index-v1"
    assert not any(str(tmp_path) in path.read_text() for path in first_dir.iterdir())

    first_predictions = read_jsonl(first_dir / "predictions.jsonl")
    second_predictions = read_jsonl(second_dir / "predictions.jsonl")
    for record in first_predictions + second_predictions:
        record.pop("latency_ms")
    assert first_predictions == second_predictions
    for metrics in (first_metrics, second_metrics):
        metrics.pop("mean_latency_ms")
        metrics.pop("median_latency_ms")
        metrics.pop("experiment")
    assert first_metrics == second_metrics

    with pytest.raises(EvaluationError, match="EVALUATION_OUTPUT_EXISTS"):
        evaluate_bm25(
            claims_path,
            gold_path,
            corpus_dir,
            index_dir,
            "dev",
            first_dir,
            [5, 10, 20],
        )
