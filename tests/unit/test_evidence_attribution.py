import json

import pytest

from medclaim.corpus.scifact_corpus import corpus_content_hash, sha256_text
from medclaim.explanation.attribution import AttributionError, CorpusResolver


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    documents = [{
        "document_id": "scifact:document:1", "dataset": "scifact",
        "source_document_id": "1", "title": "Authoritative title",
        "source_type": "scientific_abstract", "source_url": "https://example.invalid/source",
        "publication_year": 2024, "text": "Evidence text.",
        "content_hash": sha256_text("Evidence text."), "corpus_version": "corpus-v1", "metadata": {},
    }]
    passages = [{
        "passage_id": "scifact:document:1:p:0", "document_id": "scifact:document:1",
        "dataset": "scifact", "passage_index": 0, "text": "Evidence text.",
        "start_char": 0, "end_char": 14, "token_count": 2,
        "content_hash": sha256_text("Evidence text."), "corpus_version": "corpus-v1",
        "metadata": {"source_type": "scientific_abstract"},
    }]
    (directory / "documents.jsonl").write_text(json.dumps(documents[0]) + "\n")
    (directory / "passages.jsonl").write_text(json.dumps(passages[0]) + "\n")
    write_json(directory / "manifest.json", {
        "artifact_type": "medical_evidence_corpus", "corpus_version": "corpus-v1",
        "document_count": 1, "passage_count": 1, "content_hash": corpus_content_hash(passages),
    })
    return directory


def supplied():
    return [{
        "passage_id": "scifact:document:1:p:0", "document_id": "scifact:document:1",
        "text": "model supplied text ignored", "bm25_rank": 4, "bm25_score": 8.92,
        "dense_rank": 2, "dense_score": 0.81, "pre_rerank_rank": 2,
        "rank": 1, "rrf_score": 0.03, "reranker_score": 7.42,
    }]


def test_authoritative_source_and_retrieval_metadata_are_resolved(tmp_path):
    resolver = CorpusResolver(corpus(tmp_path))
    result = {
        "evidence_used": ["scifact:document:1:p:0"],
        "title": "Invented title",
        "component_results": [{"component_id": "req:component:1", "evidence_used": ["scifact:document:1:p:0"]}],
    }
    attribution = resolver.resolve(result, supplied())[0]
    assert attribution["title"] == "Authoritative title"
    assert attribution["text"] == "Evidence text."
    assert attribution["component_ids"] == ["req:component:1"]
    assert attribution["retrieval"]["rerank_rank"] == 1


def test_unknown_or_unsupplied_citation_fails(tmp_path):
    resolver = CorpusResolver(corpus(tmp_path))
    with pytest.raises(AttributionError, match="UNSUPPLIED_CITATION"):
        resolver.resolve({"evidence_used": ["unknown"]}, supplied())
    with pytest.raises(AttributionError, match="UNKNOWN_PASSAGE"):
        resolver.resolve({"evidence_used": ["unknown"]}, [{"passage_id": "unknown"}])


def test_cross_version_citation_fails(tmp_path):
    resolver = CorpusResolver(corpus(tmp_path))
    rows = supplied()
    rows[0]["corpus_version"] = "other-v1"
    with pytest.raises(AttributionError, match="VERSION_MISMATCH"):
        resolver.resolve({"evidence_used": ["scifact:document:1:p:0"]}, rows)
