import json

import faiss
import numpy as np
import pytest

from medclaim.retrieval.dense import DenseError, build_dense_index

from tests.dense_helpers import (
    MODEL_ID,
    MODEL_REVISION,
    FakeEmbedder,
    build_fake_dense_index,
    copy_corpus,
    read_passages,
    update_corpus_manifest,
    write_passages,
)


def test_dense_index_artifacts_alignment_batching_and_normalization(tmp_path):
    corpus_dir, index_dir, embedder = build_fake_dense_index(tmp_path)
    assert {path.name for path in index_dir.iterdir()} == {
        "index.faiss",
        "embeddings.npy",
        "passage_ids.json",
        "manifest.json",
    }
    passages = read_passages(corpus_dir / "passages.jsonl")
    passage_ids = json.loads((index_dir / "passage_ids.json").read_text())
    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert passage_ids == [passage["passage_id"] for passage in passages]
    assert embeddings.shape == (5, 4)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), np.ones(5))
    assert index.ntotal == 5
    assert index.d == 4
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]["texts"]) == 5
    assert embedder.calls[0]["batch_size"] == 2
    assert embedder.calls[0]["normalize_embeddings"] is True
    assert manifest["embedding"] == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": 4,
        "normalize_embeddings": True,
            "dtype": "float32",
            "batch_size": 2,
            "provider": "sentence_transformers",
            "document_prefix": None,
            "query_prefix": None,
        }
    assert not (corpus_dir / "gold_evidence.jsonl").exists()


def test_duplicate_passage_id_and_empty_passage_are_rejected_before_encoding(tmp_path):
    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    passages = read_passages(corpus_dir / "passages.jsonl")
    passages[1]["passage_id"] = passages[0]["passage_id"]
    write_passages(corpus_dir / "passages.jsonl", passages)
    update_corpus_manifest(corpus_dir, passages)
    embedder = FakeEmbedder()
    with pytest.raises(DenseError, match="DENSE_DUPLICATE_PASSAGE_ID"):
        build_dense_index(
            corpus_dir,
            tmp_path / "indexes",
            "dense-v1",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            embedder=embedder,
        )
    assert embedder.calls == []

    passages[1]["passage_id"] = "scifact:document:10:p:0"
    passages[0]["text"] = "  "
    write_passages(corpus_dir / "passages.jsonl", passages)
    update_corpus_manifest(corpus_dir, passages)
    with pytest.raises(DenseError, match="DENSE_EMPTY_PASSAGE"):
        build_dense_index(
            corpus_dir,
            tmp_path / "indexes",
            "dense-v1",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            embedder=embedder,
        )
    assert embedder.calls == []


@pytest.mark.parametrize("failure", ["nan", "dimension"])
def test_invalid_embedding_outputs_are_rejected(tmp_path, failure):
    class InvalidEmbedder(FakeEmbedder):
        def encode(self, texts, **kwargs):
            if failure == "nan":
                result = np.ones((len(texts), 4), dtype=np.float32)
                result[0, 0] = np.nan
                return result
            return np.ones((len(texts), 3), dtype=np.float32)

    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    expected = (
        "DENSE_NONFINITE_EMBEDDING"
        if failure == "nan"
        else "DENSE_DIMENSION_MISMATCH"
    )
    with pytest.raises(DenseError, match=expected):
        build_dense_index(
            corpus_dir,
            tmp_path / "indexes",
            "dense-v1",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            embedder=InvalidEmbedder(),
        )


def test_existing_dense_index_version_is_immutable(tmp_path):
    corpus_dir, index_dir, _ = build_fake_dense_index(tmp_path)
    before = (index_dir / "manifest.json").read_bytes()
    with pytest.raises(DenseError, match="DENSE_INDEX_VERSION_EXISTS"):
        build_dense_index(
            corpus_dir,
            tmp_path / "indexes",
            "dense-v1",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            embedder=FakeEmbedder(),
        )
    assert (index_dir / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("version", ["", ".", "..", "../dense", "a/b", "bad name"])
def test_invalid_dense_index_versions(tmp_path, version):
    with pytest.raises(DenseError, match="DENSE_INVALID_VERSION"):
        build_dense_index(
            tmp_path,
            tmp_path / "indexes",
            version,
            embedder=FakeEmbedder(),
        )
