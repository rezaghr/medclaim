import json

from medclaim.corpus.combined import build_combined_corpus
from medclaim.datasets.unified import build_unified_dataset
from medclaim.retrieval.bm25 import BM25Retriever, build_bm25_index
from tests.multi_dataset_helpers import create_all_normalized_fixtures, read_jsonl


def test_three_dataset_pipeline_and_bm25_compatibility(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    dataset = build_unified_dataset(*sources.values(), tmp_path / "datasets", "medical-dataset-v1")
    corpus = build_combined_corpus(dataset, tmp_path / "corpora", "medical-corpus-v1")

    assert {path.name for path in dataset.iterdir()} == {
        "claims.jsonl", "documents.jsonl", "evidence_relations.jsonl",
        "label_schema.json", "quality_report.json", "manifest.json",
    }
    assert {path.name for path in corpus.iterdir()} == {
        "documents.jsonl", "passages.jsonl", "gold_evidence.jsonl",
        "quality_report.json", "manifest.json",
    }
    claims = read_jsonl(dataset / "claims.jsonl")
    documents = read_jsonl(corpus / "documents.jsonl")
    passages = read_jsonl(corpus / "passages.jsonl")
    gold = read_jsonl(corpus / "gold_evidence.jsonl")
    document_ids = {row["document_id"] for row in documents}
    passage_ids = {row["passage_id"] for row in passages}
    assert all(item["document_id"] in document_ids for claim in claims for item in claim["evidence_sets"])
    assert all(passage_id in passage_ids for claim in gold for item in claim["evidence_sets"] for passage_id in item["passage_ids"])
    assert {row["dataset"] for row in passages} == {"scifact", "healthver", "pubhealth"}
    report = json.loads((dataset / "quality_report.json").read_text())
    assert report["label_distribution"] == {
        "SUPPORTS": 1, "REFUTES": 1, "NOT_ENOUGH_INFO": 1, "MIXED": 1, "UNLABELED": 0
    }
    assert all(row["gold_explanation"] not in {passage["text"] for passage in passages} for row in claims if row["gold_explanation"])

    index = build_bm25_index(corpus, tmp_path / "indexes", "medical-bm25-v1")
    retriever = BM25Retriever(index, corpus)
    assert retriever.search("Vitamin immune", top_k=1)["results"][0]["dataset"] == "scifact"
    assert retriever.search("Smoking respiratory", top_k=1)["results"][0]["dataset"] == "healthver"
