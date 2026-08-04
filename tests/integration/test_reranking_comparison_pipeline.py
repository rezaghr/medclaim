import json

import pytest

from medclaim.evaluation.bm25_evaluation import EvaluationError
from medclaim.evaluation.reranking_comparison import compare_reranking
from medclaim.reranking.cross_encoder import CrossEncoderReranker
from medclaim.reranking.models import RerankingConfiguration
from medclaim.retrieval.bm25 import build_bm25_index
from medclaim.retrieval.configuration import RetrievalSettings

from tests.dense_helpers import FakeEmbedder, build_fake_dense_index


class FakeCrossEncoderModel:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append((list(pairs), batch_size, show_progress_bar))
        return [
            10.0
            if "shared scientific" in passage.lower()
            else 8.0
            if "hydroxychloroquine" in passage.lower()
            else 0.0
            for _, passage in pairs
        ]


class FakeVerifier:
    model_id = "fake/verifier-v1"
    prompt_version = "evidence-only-v1"

    def __init__(self):
        self.calls = []

    def verify(self, claim, evidence):
        self.calls.append((claim, evidence))
        return {
            "verdict": "REFUTES" if "hydroxychloroquine" in claim.lower() else "SUPPORTS",
            "latency_ms": 2.0,
        }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_reranking_retrieval_and_verifier_ablation_end_to_end(tmp_path):
    corpus_dir, dense_index_dir, _ = build_fake_dense_index(tmp_path)
    bm25_index_dir = build_bm25_index(
        corpus_dir, tmp_path / "bm25-indexes", "bm25-v1"
    )
    claims_path = tmp_path / "claims.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        claims_path,
        [
            {
                "claim_id": "scifact:claim:1",
                "claim_text": "unrelated wording",
                "original_split": "dev",
                "unified_label": "SUPPORTS",
            },
            {
                "claim_id": "scifact:claim:2",
                "claim_text": "Hydroxychloroquine viral infection",
                "original_split": "dev",
                "unified_label": "REFUTES",
            },
            {
                "claim_id": "scifact:claim:3",
                "claim_text": "No gold evidence",
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
                "original_split": "dev",
                "evidence_sets": [
                    {"passage_ids": ["scifact:document:50:p:0"]}
                ],
            },
            {
                "claim_id": "scifact:claim:2",
                "original_split": "dev",
                "evidence_sets": [
                    {"passage_ids": ["scifact:document:20:p:0"]}
                ],
            },
        ],
    )
    model = FakeCrossEncoderModel()
    reranker = CrossEncoderReranker(
        "fake/reranker-v1",
        model_revision="test-rev",
        batch_size=2,
        maximum_input_length=128,
        model=model,
    )
    reranking = RerankingConfiguration(
        model_id="fake/reranker-v1",
        model_revision="test-rev",
        candidate_count=5,
        final_evidence_k=2,
        batch_size=2,
        maximum_input_length=128,
    )
    retrieval = RetrievalSettings(
        sparse_top_k=5,
        dense_top_k=5,
        fusion_top_k=5,
        rrf_k=60,
        final_evidence_k=2,
    )
    verifier = FakeVerifier()
    output_dir = tmp_path / "reranking-comparison"
    result = compare_reranking(
        claims_path,
        gold_path,
        corpus_dir,
        bm25_index_dir,
        dense_index_dir,
        "dev",
        output_dir,
        reranker,
        reranking,
        retrieval,
        verifier,
        dense_embedder=FakeEmbedder(),
    )

    retrieval_metrics = result["retrieval_metrics"]
    assert retrieval_metrics["methods"]["hybrid"][
        "candidate_pool_complete_evidence_recall_at_k"
    ]["5"] == 1.0
    assert retrieval_metrics["methods"]["hybrid_reranked"][
        "candidate_pool_complete_evidence_recall_at_k"
    ]["5"] == 1.0
    assert retrieval_metrics["methods"]["hybrid_reranked"][
        "complete_evidence_recall_at_k"
    ]["2"] >= retrieval_metrics["methods"]["hybrid"][
        "complete_evidence_recall_at_k"
    ]["2"]
    classification = result["classification_metrics"]
    assert classification["status"] == "available"
    assert classification["methods"]["hybrid"]["accuracy"] == 1.0
    assert classification["methods"]["hybrid_reranked"]["macro_f1"] == 1.0
    assert all(
        set(evidence_item) == {"passage_id", "text"}
        for _, evidence in verifier.calls
        for evidence_item in evidence
    )
    assert all(
        isinstance(pair, tuple) and len(pair) == 2
        for pairs, _, _ in model.calls
        for pair in pairs
    )
    assert all(batch_size == 2 for _, batch_size, _ in model.calls)
    assert {path.name for path in output_dir.iterdir()} == {
        "hybrid_predictions.jsonl",
        "reranked_predictions.jsonl",
        "retrieval_metrics.json",
        "classification_metrics.json",
        "comparison.csv",
        "reranking_changes.jsonl",
        "errors.jsonl",
        "manifest.json",
    }
    changes = read_jsonl(output_dir / "reranking_changes.jsonl")
    assert len(changes) == 2
    assert json.loads((output_dir / "manifest.json").read_text())[
        "sample_experiment"
    ] is False
    json.dumps(result, allow_nan=False)

    with pytest.raises(EvaluationError, match="RERANKING_EXPERIMENT_EXISTS"):
        compare_reranking(
            claims_path,
            gold_path,
            corpus_dir,
            bm25_index_dir,
            dense_index_dir,
            "dev",
            output_dir,
            reranker,
            reranking,
            retrieval,
            verifier,
            dense_embedder=FakeEmbedder(),
        )


def test_sample_run_without_verifier_is_marked_and_transparent(tmp_path):
    corpus_dir, dense_index_dir, _ = build_fake_dense_index(tmp_path)
    bm25_index_dir = build_bm25_index(
        corpus_dir, tmp_path / "bm25-indexes", "bm25-v1"
    )
    claims_path = tmp_path / "claims.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        claims_path,
        [
            {
                "claim_id": "claim:1",
                "claim_text": "Aspirin pain",
                "original_split": "dev",
                "unified_label": "SUPPORTS",
            }
        ],
    )
    write_jsonl(
        gold_path,
        [
            {
                "claim_id": "claim:1",
                "evidence_sets": [
                    {"passage_ids": ["scifact:document:30:p:0"]}
                ],
            }
        ],
    )
    model = FakeCrossEncoderModel()
    reranker = CrossEncoderReranker(
        "fake/reranker-v1", batch_size=2, maximum_input_length=128, model=model
    )
    config = RerankingConfiguration(
        model_id="fake/reranker-v1",
        candidate_count=5,
        final_evidence_k=2,
        batch_size=2,
        maximum_input_length=128,
    )
    output = tmp_path / "sample"
    result = compare_reranking(
        claims_path,
        gold_path,
        corpus_dir,
        bm25_index_dir,
        dense_index_dir,
        "dev",
        output,
        reranker,
        config,
        RetrievalSettings(
            sparse_top_k=5,
            dense_top_k=5,
            fusion_top_k=5,
            final_evidence_k=2,
        ),
        max_claims=1,
        dense_embedder=FakeEmbedder(),
    )
    assert result["classification_metrics"]["status"] == "not_available"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sample_experiment"] is True
    assert manifest["max_claims"] == 1
