import json

from medclaim.evaluation.dense_evaluation import evaluate_dense
from medclaim.retrieval.dense import DenseRetriever

from tests.dense_helpers import FakeEmbedder, build_fake_dense_index


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dense_build_search_and_evaluation_end_to_end(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    search = DenseRetriever(
        index_dir, corpus_dir, embedder=FakeEmbedder()
    ).search("medicine assessed against a viral illness", 5)
    assert search["results"][0]["passage_id"] == "scifact:document:20:p:0"
    assert all(
        field in search["results"][0]
        for field in ("rank", "passage_id", "document_id", "dataset", "text")
    )

    claims_path = tmp_path / "claims.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        claims_path,
        [
            {
                "claim_id": "scifact:claim:1",
                "dataset": "scifact",
                "claim_text": "medicine assessed against a viral illness",
                "original_split": "dev",
                "unified_label": "SUPPORTS",
            },
            {
                "claim_id": "scifact:claim:2",
                "dataset": "scifact",
                "claim_text": "excluded claim",
                "original_split": "dev",
                "unified_label": "SUPPORTS",
            },
        ],
    )
    write_jsonl(
        gold_path,
        [
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
        ],
    )
    output_dir = tmp_path / "dense-evaluation"
    metrics = evaluate_dense(
        claims_path,
        gold_path,
        corpus_dir,
        index_dir,
        "dev",
        output_dir,
        [5, 10, 20],
        embedder=FakeEmbedder(),
    )
    assert metrics["retrieval_mode"] == "dense"
    assert metrics["complete_evidence_recall_at_k"] == {
        "5": 1.0,
        "10": 1.0,
        "20": 1.0,
    }
    assert metrics["mrr"] == 1.0
    assert metrics["excluded_claims"] == 1
    assert {path.name for path in output_dir.iterdir()} == {
        "predictions.jsonl",
        "metrics.json",
        "retrieval_errors.jsonl",
        "manifest.json",
    }
    prediction = read_jsonl(output_dir / "predictions.jsonl")[0]
    assert "dense_score" in prediction["retrieved"][0]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["embedding"] == {
        "model_id": "fake/semantic-v1",
        "model_revision": "test-rev",
        "dimension": 4,
        "normalize_embeddings": True,
    }
    json.dumps(metrics, allow_nan=False)
