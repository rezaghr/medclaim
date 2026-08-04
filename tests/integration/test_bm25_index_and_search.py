import hashlib
import json
import shutil
from pathlib import Path

import pytest

from medclaim.retrieval.bm25 import BM25Error, BM25Retriever, build_bm25_index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bm25_corpus"


def checksum(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_bm25_index_and_search_end_to_end(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shutil.copy2(FIXTURES / "passages.jsonl", corpus_dir / "passages.jsonl")
    shutil.copy2(FIXTURES / "manifest.json", corpus_dir / "manifest.json")
    output_root = tmp_path / "indexes"

    index_dir = build_bm25_index(
        corpus_dir, output_root, "bm25-fixture-index-v1"
    )
    assert {path.name for path in index_dir.iterdir()} == {
        "index.pkl",
        "passage_ids.json",
        "manifest.json",
    }
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["artifact_type"] == "bm25_index"
    assert manifest["corpus"]["passage_count"] == 5
    assert manifest["files"]["index"]["sha256"] == checksum(index_dir / "index.pkl")
    assert manifest["files"]["passage_ids"]["sha256"] == checksum(
        index_dir / "passage_ids.json"
    )

    first = BM25Retriever(index_dir, corpus_dir).search("hydroxychloroquine", 3)
    second = BM25Retriever(index_dir, corpus_dir).search("hydroxychloroquine", 3)
    assert first["results"][0] == second["results"][0]
    assert first["results"][0]["passage_id"] == "scifact:document:20:p:0"
    assert first["results"][0]["text"] == (
        "Hydroxychloroquine was evaluated for viral infection."
    )
    assert set(first["results"][0]) == {
        "rank",
        "passage_id",
        "document_id",
        "dataset",
        "text",
        "bm25_score",
    }
    assert first["latency_ms"] >= 0
    json.dumps(first, allow_nan=False)
    assert not (corpus_dir / "gold_evidence.jsonl").exists()

    with pytest.raises(BM25Error, match="BM25_INDEX_VERSION_EXISTS"):
        build_bm25_index(corpus_dir, output_root, "bm25-fixture-index-v1")

    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text())
    corpus_manifest["content_hash"] = "sha256:" + "0" * 64
    (corpus_dir / "manifest.json").write_text(json.dumps(corpus_manifest))
    with pytest.raises(BM25Error, match="BM25_CORPUS_HASH_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)
