import csv
import json

import pytest

from medclaim.evaluation.bm25_evaluation import EvaluationError
from medclaim.evaluation.retrieval_comparison import compare_retrieval
from medclaim.retrieval.bm25 import BM25Retriever, build_bm25_index
from medclaim.retrieval.configuration import RetrievalSettings
from medclaim.retrieval.dense import DenseRetriever
from medclaim.retrieval.hybrid import HybridRetriever

from tests.dense_helpers import FakeEmbedder, build_fake_dense_index


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_hybrid_search_and_comparison_end_to_end(tmp_path):
    corpus_dir, dense_index_dir, _ = build_fake_dense_index(tmp_path)
    bm25_index_dir = build_bm25_index(
        corpus_dir, tmp_path / "bm25-indexes", "bm25-v1"
    )
    sparse = BM25Retriever(bm25_index_dir, corpus_dir)
    dense = DenseRetriever(
        dense_index_dir, corpus_dir, embedder=FakeEmbedder()
    )
    hybrid_result = HybridRetriever(sparse, dense, 5, 5, 5, 60).search(
        "Hydroxychloroquine viral infection"
    )
    assert hybrid_result["results"][0]["passage_id"] == "scifact:document:20:p:0"
    assert hybrid_result["results"][0]["bm25_rank"] == 1
    assert hybrid_result["results"][0]["dense_rank"] == 1

    claims_path = tmp_path / "claims.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    claims = [
        {
            "claim_id": "scifact:claim:2",
            "claim_text": "Aspirin reduces pain",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
        },
        {
            "claim_id": "scifact:claim:1",
            "claim_text": "Hydroxychloroquine viral infection",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
        },
        {
            "claim_id": "scifact:claim:3",
            "claim_text": "No annotated evidence",
            "original_split": "dev",
            "unified_label": "SUPPORTS",
        },
    ]
    gold = [
        {
            "claim_id": "scifact:claim:1",
            "original_split": "dev",
            "evidence_sets": [
                {"passage_ids": ["scifact:document:20:p:0"]}
            ],
        },
        {
            "claim_id": "scifact:claim:2",
            "original_split": "dev",
            "evidence_sets": [
                {"passage_ids": ["scifact:document:30:p:0"]}
            ],
        },
    ]
    write_jsonl(claims_path, claims)
    write_jsonl(gold_path, gold)
    settings = RetrievalSettings(
        sparse_top_k=5,
        dense_top_k=5,
        fusion_top_k=5,
        rrf_k=60,
        final_evidence_k=2,
    )
    output_dir = tmp_path / "comparison"
    metrics = compare_retrieval(
        claims_path,
        gold_path,
        corpus_dir,
        bm25_index_dir,
        dense_index_dir,
        "dev",
        output_dir,
        [1, 3, 5],
        settings=settings,
        dense_embedder=FakeEmbedder(),
    )
    assert set(metrics["methods"]) == {"bm25", "dense", "hybrid"}
    assert metrics["evaluated_claims"] == 2
    assert metrics["excluded_claims"] == 1
    assert {path.name for path in output_dir.iterdir()} == {
        "bm25_predictions.jsonl",
        "dense_predictions.jsonl",
        "hybrid_predictions.jsonl",
        "metrics.json",
        "comparison.csv",
        "retrieval_errors.jsonl",
        "manifest.json",
    }
    prediction_ids = {
        method: [
            record["claim_id"]
            for record in read_jsonl(output_dir / f"{method}_predictions.jsonl")
        ]
        for method in ("bm25", "dense", "hybrid")
    }
    assert prediction_ids["bm25"] == prediction_ids["dense"] == prediction_ids["hybrid"]
    with (output_dir / "comparison.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["method"] for row in rows] == ["bm25", "dense", "hybrid"]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["hybrid_configuration"]["rrf_k"] == 60
    json.dumps(metrics, allow_nan=False)

    with pytest.raises(EvaluationError, match="EVALUATION_OUTPUT_EXISTS"):
        compare_retrieval(
            claims_path,
            gold_path,
            corpus_dir,
            bm25_index_dir,
            dense_index_dir,
            "dev",
            output_dir,
            [1, 3, 5],
            settings=settings,
            dense_embedder=FakeEmbedder(),
        )
