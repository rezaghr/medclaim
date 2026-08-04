import json

import pytest

from medclaim.corpus.combined import build_combined_corpus
from medclaim.corpus.scifact_corpus import CorpusBuildError, corpus_content_hash, sha256_text
from medclaim.datasets.unified import build_unified_dataset
from tests.multi_dataset_helpers import create_all_normalized_fixtures, read_jsonl


def build_fixture(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    dataset = build_unified_dataset(*sources.values(), tmp_path / "datasets", "dataset-v1")
    corpus = build_combined_corpus(dataset, tmp_path / "corpora", "corpus-v1", max_passage_words=8)
    return dataset, corpus


def test_dataset_specific_passages_offsets_hashes_and_metadata(tmp_path):
    _, corpus = build_fixture(tmp_path)
    documents = read_jsonl(corpus / "documents.jsonl")
    passages = read_jsonl(corpus / "passages.jsonl")
    assert [row["dataset"] for row in passages] == ["scifact", "healthver", "pubhealth", "pubhealth"]
    source_text = {row["document_id"]: row["text"] for row in documents}
    for passage in passages:
        assert source_text[passage["document_id"]][passage["start_char"]:passage["end_char"]] == passage["text"]
        assert passage["content_hash"] == sha256_text(passage["text"])
        assert passage["metadata"]["source_type"]
        assert passage["passage_id"] == f"{passage['document_id']}:p:{passage['passage_index']}"
    assert passages[0]["metadata"]["source_sentence_indices"] == [0]
    assert passages[1]["metadata"]["source_sentence_indices"] == []


def test_gold_evidence_resolution_and_explanation_exclusion(tmp_path):
    dataset, corpus = build_fixture(tmp_path)
    gold = read_jsonl(corpus / "gold_evidence.jsonl")
    claims = read_jsonl(dataset / "claims.jsonl")
    passages = read_jsonl(corpus / "passages.jsonl")
    assert len(gold) == len(claims) == 4
    assert gold[0]["evidence_sets"][0]["passage_ids"] == ["scifact:document:10:p:0"]
    assert gold[1]["evidence_sets"][0]["passage_ids"] == ["healthver:document:20:p:0"]
    by_id = {row["claim_id"]: row for row in gold}
    assert by_id["pubhealth:claim:4"]["evidence_sets"] == []
    explanations = {row["gold_explanation"] for row in claims if row["gold_explanation"]}
    assert all(explanation not in passage["text"] for explanation in explanations for passage in passages)


def test_combined_report_manifest_hash_and_immutability(tmp_path):
    _, corpus = build_fixture(tmp_path)
    passages = read_jsonl(corpus / "passages.jsonl")
    manifest = json.loads((corpus / "manifest.json").read_text())
    report = json.loads((corpus / "quality_report.json").read_text())
    assert manifest["dataset"] == "multi_dataset"
    assert manifest["content_hash"] == corpus_content_hash(passages)
    assert report["documents_per_dataset"] == {"scifact": 1, "healthver": 1, "pubhealth": 1}
    assert report["claims_with_resolved_evidence"] == 3
    assert report["claims_without_evidence"] == 1
    with pytest.raises(CorpusBuildError, match="VERSION_EXISTS"):
        build_combined_corpus(corpus.parent.parent / "datasets" / "dataset-v1", corpus.parent, "corpus-v1")


def test_pubhealth_rule_split_respects_word_maximum_and_reports_duplicates(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    pub_docs_path = sources["pubhealth"] / "documents.jsonl"
    pub_docs = read_jsonl(pub_docs_path)
    pub_docs[0]["sentences"] = []
    pub_docs[0]["text"] = "Repeated evidence phrase. Repeated evidence phrase."
    from tests.multi_dataset_helpers import write_json, write_jsonl
    write_jsonl(pub_docs_path, pub_docs)
    source_manifest = json.loads((sources["pubhealth"] / "manifest.json").read_text())
    source_manifest.pop("content_hash")
    write_json(sources["pubhealth"] / "manifest.json", source_manifest)
    dataset = build_unified_dataset(*sources.values(), tmp_path / "datasets", "dataset-v1")
    corpus = build_combined_corpus(dataset, tmp_path / "corpora", "corpus-v1", max_passage_words=3)
    passages = [row for row in read_jsonl(corpus / "passages.jsonl") if row["dataset"] == "pubhealth"]
    report = json.loads((corpus / "quality_report.json").read_text())
    assert all(row["token_count"] <= 3 for row in passages)
    assert report["exact_duplicate_passage_groups"] == 1


def test_oversized_source_sentence_is_split_and_all_chunks_retain_gold_provenance(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    pub_docs_path = sources["pubhealth"] / "documents.jsonl"
    pub_docs = read_jsonl(pub_docs_path)
    long_sentence = "Aspirin lowers risk for some people but does not prevent every heart attack."
    pub_docs[0]["text"] = long_sentence
    pub_docs[0]["sentences"] = [long_sentence]
    from tests.multi_dataset_helpers import write_json, write_jsonl

    write_jsonl(pub_docs_path, pub_docs)
    source_manifest = json.loads((sources["pubhealth"] / "manifest.json").read_text())
    source_manifest.pop("content_hash")
    write_json(sources["pubhealth"] / "manifest.json", source_manifest)
    dataset = build_unified_dataset(*sources.values(), tmp_path / "datasets", "dataset-v1")
    corpus = build_combined_corpus(
        dataset, tmp_path / "corpora", "corpus-v1", max_passage_words=4
    )

    passages = [
        row
        for row in read_jsonl(corpus / "passages.jsonl")
        if row["dataset"] == "pubhealth"
    ]
    assert len(passages) == 4
    assert all(row["token_count"] <= 4 for row in passages)
    assert all(row["metadata"]["source_sentence_indices"] == [0] for row in passages)
    gold = {
        row["claim_id"]: row for row in read_jsonl(corpus / "gold_evidence.jsonl")
    }
    assert gold["pubhealth:claim:3"]["evidence_sets"][0]["passage_ids"] == [
        row["passage_id"] for row in passages
    ]


def test_punctuation_only_tail_does_not_push_previous_chunk_over_limit(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    pub_docs_path = sources["pubhealth"] / "documents.jsonl"
    pub_docs = read_jsonl(pub_docs_path)
    sentence = "one two three four ."
    pub_docs[0]["text"] = sentence
    pub_docs[0]["sentences"] = [sentence]
    from tests.multi_dataset_helpers import write_json, write_jsonl

    write_jsonl(pub_docs_path, pub_docs)
    source_manifest = json.loads((sources["pubhealth"] / "manifest.json").read_text())
    source_manifest.pop("content_hash")
    write_json(sources["pubhealth"] / "manifest.json", source_manifest)
    dataset = build_unified_dataset(*sources.values(), tmp_path / "datasets", "dataset-v1")
    corpus = build_combined_corpus(
        dataset, tmp_path / "corpora", "corpus-v1", max_passage_words=4
    )
    passages = [
        row
        for row in read_jsonl(corpus / "passages.jsonl")
        if row["dataset"] == "pubhealth"
    ]
    assert [row["token_count"] for row in passages] == [3, 2]
    assert all(any(character.isalnum() for character in row["text"]) for row in passages)
