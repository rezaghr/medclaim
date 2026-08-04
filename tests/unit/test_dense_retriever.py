import hashlib
import json

import faiss
import numpy as np
import pytest

from medclaim.retrieval.dense import DenseError, DenseRetriever

from tests.dense_helpers import FakeEmbedder, build_fake_dense_index, read_passages, write_passages


def checksum(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_manifest_checksum(index_dir, key):
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = index_dir / manifest["files"][key]["path"]
    manifest["files"][key]["sha256"] = checksum(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_semantic_search_top_k_and_json_contract(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    retriever = DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())
    result = retriever.search("medicine assessed against a viral illness", 2)
    assert result["results"][0]["passage_id"] == "scifact:document:20:p:0"
    assert result["returned_count"] == 2
    assert result["retrieval_mode"] == "dense"
    assert set(result["results"][0]) == {
        "rank",
        "passage_id",
        "document_id",
        "dataset",
        "text",
        "dense_score",
    }
    assert type(result["results"][0]["dense_score"]) is float
    json.dumps(result, allow_nan=False)


def test_top_k_larger_than_corpus_and_deterministic_ties(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    retriever = DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())
    result = retriever.search("shared scientific finding", 10)
    assert result["returned_count"] == 5
    tied = [
        item["passage_id"]
        for item in result["results"]
        if item["passage_id"] in {
            "scifact:document:40:p:0",
            "scifact:document:50:p:0",
        }
    ]
    assert tied == sorted(tied)
    assert retriever.search("shared scientific finding", 1)["results"][0][
        "passage_id"
    ] == "scifact:document:40:p:0"
    repeated = retriever.search("shared scientific finding", 10)
    assert [
        (item["passage_id"], item["dense_score"]) for item in result["results"]
    ] == [
        (item["passage_id"], item["dense_score"]) for item in repeated["results"]
    ]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_invalid_dense_queries(tmp_path, query):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    retriever = DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())
    with pytest.raises(DenseError, match="DENSE_EMPTY_QUERY"):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1, 101, True])
def test_invalid_dense_top_k(tmp_path, top_k):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    retriever = DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())
    with pytest.raises(DenseError, match="DENSE_INVALID_TOP_K"):
        retriever.search("vitamin", top_k)


def test_query_vector_dimension_validation(tmp_path):
    class WrongQueryEmbedder(FakeEmbedder):
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype=np.float32)

    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    retriever = DenseRetriever(index_dir, corpus_dir, embedder=WrongQueryEmbedder())
    with pytest.raises(DenseError, match="DENSE_DIMENSION_MISMATCH"):
        retriever.search("vitamin")


@pytest.mark.parametrize("filename", ["index.faiss", "embeddings.npy", "passage_ids.json"])
def test_dense_artifact_checksum_validation(tmp_path, filename):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    with (index_dir / filename).open("ab") as output_file:
        output_file.write(b"x")
    with pytest.raises(DenseError, match="DENSE_INDEX_CHECKSUM_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())


def test_corpus_version_and_content_hash_mismatch(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_version"] = "other-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DenseError, match="DENSE_INDEX_CORPUS_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())

    manifest["corpus_version"] = "bm25-fixture-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    passages = read_passages(corpus_dir / "passages.jsonl")
    passages[0]["text"] += " altered"
    write_passages(corpus_dir / "passages.jsonl", passages)
    with pytest.raises(DenseError, match="DENSE_CORPUS_HASH_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())


def test_model_metadata_validation(tmp_path):
    class WrongModelEmbedder(FakeEmbedder):
        model_id = "wrong/model"

    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    with pytest.raises(DenseError, match="DENSE_MODEL_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=WrongModelEmbedder())

    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding"]["normalize_embeddings"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DenseError, match="DENSE_INDEX_MANIFEST_INVALID"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())


def test_faiss_ntotal_validation(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    short_index = faiss.IndexFlatIP(4)
    short_index.add(np.ones((4, 4), dtype=np.float32))
    faiss.write_index(short_index, str(index_dir / "index.faiss"))
    rewrite_manifest_checksum(index_dir, "index")
    with pytest.raises(DenseError, match="DENSE_INDEX_COUNT_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())


def test_passage_mapping_alignment_validation(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    mapping_path = index_dir / "passage_ids.json"
    mapping = json.loads(mapping_path.read_text())
    mapping.reverse()
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    rewrite_manifest_checksum(index_dir, "passage_ids")
    with pytest.raises(DenseError, match="DENSE_PASSAGE_MAPPING_MISMATCH"):
        DenseRetriever(index_dir, corpus_dir, embedder=FakeEmbedder())
