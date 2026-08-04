import json
import hashlib
import shutil
from pathlib import Path

import pytest

from medclaim.corpus.scifact_corpus import corpus_content_hash
from medclaim.retrieval.bm25 import BM25Error, BM25Retriever, build_bm25_index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bm25_corpus"


def read_passages(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_passages(path, passages):
    path.write_text(
        "".join(json.dumps(passage, separators=(",", ":")) + "\n" for passage in passages),
        encoding="utf-8",
    )


def copy_corpus(destination):
    destination.mkdir(parents=True)
    shutil.copy2(FIXTURES / "passages.jsonl", destination / "passages.jsonl")
    shutil.copy2(FIXTURES / "manifest.json", destination / "manifest.json")


def build_fixture_index(tmp_path, version="index-v1"):
    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    index_dir = build_bm25_index(corpus_dir, tmp_path / "indexes", version)
    return corpus_dir, index_dir


def test_index_count_and_passage_order(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    passage_ids = json.loads((index_dir / "passage_ids.json").read_text())
    original_ids = [item["passage_id"] for item in read_passages(corpus_dir / "passages.jsonl")]
    retriever = BM25Retriever(index_dir, corpus_dir)
    assert manifest["corpus"]["passage_count"] == 5
    assert passage_ids == original_ids
    assert retriever.bm25.corpus_size == 5


def test_duplicate_passage_id_rejection(tmp_path):
    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    passages = read_passages(corpus_dir / "passages.jsonl")
    passages[1]["passage_id"] = passages[0]["passage_id"]
    write_passages(corpus_dir / "passages.jsonl", passages)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    manifest["content_hash"] = corpus_content_hash(passages)
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BM25Error, match="BM25_DUPLICATE_PASSAGE_ID"):
        build_bm25_index(corpus_dir, tmp_path / "indexes", "index-v1")


def test_duplicate_text_is_accepted(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    passage_ids = json.loads((index_dir / "passage_ids.json").read_text())
    assert "scifact:document:40:p:0" in passage_ids
    assert "scifact:document:50:p:0" in passage_ids


def test_empty_tokenized_passage_rejection(tmp_path):
    corpus_dir = tmp_path / "corpus"
    copy_corpus(corpus_dir)
    passages = read_passages(corpus_dir / "passages.jsonl")
    passages[0]["text"] = "..."
    write_passages(corpus_dir / "passages.jsonl", passages)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    manifest["content_hash"] = corpus_content_hash(passages)
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BM25Error, match="BM25_EMPTY_TOKENIZED_PASSAGE"):
        build_bm25_index(corpus_dir, tmp_path / "indexes", "index-v1")


def test_exact_and_multi_term_ranking(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    retriever = BM25Retriever(index_dir, corpus_dir)
    assert retriever.search("hydroxychloroquine", 3)["results"][0]["passage_id"] == (
        "scifact:document:20:p:0"
    )
    assert retriever.search("vitamin immune respiratory", 3)["results"][0][
        "passage_id"
    ] == "scifact:document:10:p:0"


def test_rank_numbering_top_k_and_score_serialization(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    retriever = BM25Retriever(index_dir, corpus_dir)
    result = retriever.search("scientific", 2)
    assert [item["rank"] for item in result["results"]] == [1, 2]
    assert result["returned_count"] == 2
    assert all(type(item["bm25_score"]) is float for item in result["results"])
    json.dumps(result, allow_nan=False)


def test_top_k_larger_than_corpus(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    result = BM25Retriever(index_dir, corpus_dir).search("vitamin", 10)
    assert result["returned_count"] == 5


@pytest.mark.parametrize("query", ["", "   ", "..."])
def test_empty_query_rejection(tmp_path, query):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    with pytest.raises(BM25Error, match="BM25_EMPTY_QUERY"):
        BM25Retriever(index_dir, corpus_dir).search(query)


@pytest.mark.parametrize("top_k", [0, -1, 101, True])
def test_invalid_top_k_rejection(tmp_path, top_k):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    with pytest.raises(BM25Error, match="BM25_INVALID_TOP_K"):
        BM25Retriever(index_dir, corpus_dir).search("vitamin", top_k)


def test_deterministic_tie_breaking(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    results = BM25Retriever(index_dir, corpus_dir).search("absentterm", 5)["results"]
    assert [item["passage_id"] for item in results] == sorted(
        item["passage_id"] for item in results
    )


def test_serialization_round_trip_is_deterministic(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    first = BM25Retriever(index_dir, corpus_dir).search("vitamin immune", 5)
    second = BM25Retriever(index_dir, corpus_dir).search("vitamin immune", 5)
    assert [(r["passage_id"], r["bm25_score"]) for r in first["results"]] == [
        (r["passage_id"], r["bm25_score"]) for r in second["results"]
    ]


def test_index_checksum_validation(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    with (index_dir / "index.pkl").open("ab") as output_file:
        output_file.write(b"x")
    with pytest.raises(BM25Error, match="BM25_INDEX_CHECKSUM_MISMATCH.*index.pkl"):
        BM25Retriever(index_dir, corpus_dir)


def test_mapping_checksum_validation(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    with (index_dir / "passage_ids.json").open("ab") as output_file:
        output_file.write(b" ")
    with pytest.raises(
        BM25Error, match="BM25_INDEX_CHECKSUM_MISMATCH.*passage_ids.json"
    ):
        BM25Retriever(index_dir, corpus_dir)


def test_corpus_version_mismatch(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    manifest["corpus_version"] = "other-v1"
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BM25Error, match="BM25_INDEX_CORPUS_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)


def test_corpus_content_hash_mismatch(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    passages = read_passages(corpus_dir / "passages.jsonl")
    passages[0]["text"] += " changed"
    write_passages(corpus_dir / "passages.jsonl", passages)
    with pytest.raises(BM25Error, match="BM25_CORPUS_HASH_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)


def test_index_manifest_corpus_hash_mismatch(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    manifest["corpus"]["content_hash"] = "sha256:" + "0" * 64
    (index_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BM25Error, match="BM25_INDEX_CORPUS_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)


def test_missing_passage_mapping(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    passages = read_passages(corpus_dir / "passages.jsonl")[:-1]
    write_passages(corpus_dir / "passages.jsonl", passages)
    with pytest.raises(BM25Error, match="BM25_CORPUS_COUNT_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)


def test_shortened_passage_id_mapping_is_rejected(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    mapping_path = index_dir / "passage_ids.json"
    mapping = json.loads(mapping_path.read_text())[:-1]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["passage_ids"]["sha256"] = (
        "sha256:" + hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BM25Error, match="BM25_PASSAGE_MAPPING_MISMATCH"):
        BM25Retriever(index_dir, corpus_dir)


@pytest.mark.parametrize(
    ("k1", "b", "epsilon"),
    [(0, 0.75, 0.25), (-1, 0.75, 0.25), (1.5, -0.1, 0.25), (1.5, 1.1, 0.25), (1.5, 0.75, -1)],
)
def test_build_parameter_validation(tmp_path, k1, b, epsilon):
    with pytest.raises(BM25Error, match="BM25_INVALID_PARAMETER"):
        build_bm25_index(tmp_path, tmp_path / "indexes", "index-v1", k1, b, epsilon)


def test_existing_index_directory_is_immutable(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    marker = index_dir / "index.pkl"
    before = marker.read_bytes()
    with pytest.raises(BM25Error, match="BM25_INDEX_VERSION_EXISTS"):
        build_bm25_index(corpus_dir, tmp_path / "indexes", "index-v1")
    assert marker.read_bytes() == before


@pytest.mark.parametrize("version", ["", ".", "..", "../index", "a/b", "bad name"])
def test_invalid_index_version(tmp_path, version):
    with pytest.raises(BM25Error, match="BM25_INVALID_VERSION"):
        build_bm25_index(tmp_path, tmp_path / "indexes", version)


def test_no_gold_evidence_file_is_required(tmp_path):
    corpus_dir, index_dir = build_fixture_index(tmp_path)
    assert not (corpus_dir / "gold_evidence.jsonl").exists()
    assert BM25Retriever(index_dir, corpus_dir).search("vitamin")["results"]
