import copy

import pytest

from medclaim.corpus.scifact_corpus import (
    CorpusBuildError,
    build_document_record,
    build_merged_passages,
    build_quality_report,
    build_scifact_corpus,
    build_sentence_passages,
    passage_id_for,
    resolve_gold_evidence,
    sha256_text,
    validate_version,
    whitespace_token_count,
)


def source_document(sentences=None, document_id="scifact:document:123"):
    sentences = sentences or ["First sentence.", "Second sentence.", "Third sentence."]
    return {
        "document_id": document_id,
        "dataset": "scifact",
        "source_document_id": document_id.rsplit(":", 1)[-1],
        "title": "Fixture title",
        "source_type": "scientific_abstract",
        "abstract_sentences": sentences,
        "text": " ".join(sentences),
        "metadata": {"structured": False},
    }


def source_claim(sentence_indices, document_id="scifact:document:123", claim_id="1"):
    return {
        "claim_id": f"scifact:claim:{claim_id}",
        "dataset": "scifact",
        "original_split": "train",
        "unified_label": "SUPPORTS",
        "evidence_sets": [
            {
                "evidence_set_id": f"scifact:claim:{claim_id}:evidence:0",
                "relationship": "SUPPORTS",
                "document_id": document_id,
                "sentence_indices": sentence_indices,
            }
        ],
    }


def sentence_passages(document=None, version="test-v1"):
    return build_sentence_passages(document or source_document(), version, 120)


def test_stable_passage_id():
    assert passage_id_for("scifact:document:123", 4) == (
        "scifact:document:123:p:4"
    )


def test_document_content_hash_is_stable():
    assert sha256_text("same text") == sha256_text("same text")
    assert sha256_text("same text").startswith("sha256:")


def test_passage_content_hash_changes_with_text():
    assert sha256_text("Text.") != sha256_text("text.")


def test_sentence_passage_count_and_order():
    passages = sentence_passages()
    assert len(passages) == 3
    assert [passage["passage_index"] for passage in passages] == [0, 1, 2]


def test_offset_correctness():
    document = source_document()
    for passage in sentence_passages(document):
        assert document["text"][passage["start_char"] : passage["end_char"]] == (
            passage["text"]
        )


def test_repeated_sentence_offsets_are_incremental():
    document = source_document(["Repeated.", "Repeated."])
    passages = sentence_passages(document)
    assert [(item["start_char"], item["end_char"]) for item in passages] == [
        (0, 9),
        (10, 19),
    ]


def test_whitespace_token_count():
    assert whitespace_token_count("Vitamin D affects immune function.") == 5


def test_gold_evidence_mapping():
    passages = sentence_passages()
    gold = resolve_gold_evidence([source_claim([2])], passages)
    assert gold[0]["evidence_sets"][0]["passage_ids"] == [
        "scifact:document:123:p:2"
    ]
    assert passages[2]["metadata"]["is_gold_for_any_claim"] is True
    assert passages[0]["metadata"]["is_gold_for_any_claim"] is False


def test_multiple_sentence_evidence_set_maps_to_two_passages():
    passages = sentence_passages()
    gold = resolve_gold_evidence([source_claim([1, 2])], passages)
    assert gold[0]["evidence_sets"][0]["passage_ids"] == [
        "scifact:document:123:p:1",
        "scifact:document:123:p:2",
    ]


def test_multiple_evidence_sets_remain_separate():
    claim = source_claim([0])
    second = copy.deepcopy(claim["evidence_sets"][0])
    second["evidence_set_id"] = "scifact:claim:1:evidence:1"
    second["sentence_indices"] = [2]
    claim["evidence_sets"].append(second)
    gold = resolve_gold_evidence([claim], sentence_passages())
    assert len(gold[0]["evidence_sets"]) == 2


def test_missing_document_failure():
    with pytest.raises(
        CorpusBuildError, match="CORPUS_MISSING_EVIDENCE_DOCUMENT"
    ):
        resolve_gold_evidence(
            [source_claim([0], document_id="scifact:document:999")],
            sentence_passages(),
        )


def test_invalid_sentence_index_failure():
    with pytest.raises(
        CorpusBuildError, match="CORPUS_UNRESOLVED_SENTENCE_REFERENCE.*index 7"
    ):
        resolve_gold_evidence([source_claim([7])], sentence_passages())


def test_empty_evidence_sentence_list_failure():
    with pytest.raises(CorpusBuildError, match="CORPUS_INVALID_EVIDENCE"):
        resolve_gold_evidence([source_claim([])], sentence_passages())


def test_empty_sentence_failure():
    document = source_document(["Valid.", "   "])
    with pytest.raises(CorpusBuildError, match="CORPUS_EMPTY_SENTENCE.*index 1"):
        sentence_passages(document)


def test_text_mismatch_failure():
    document = source_document()
    document["text"] = "Different text."
    with pytest.raises(CorpusBuildError, match="CORPUS_DOCUMENT_TEXT_MISMATCH"):
        sentence_passages(document)


def test_short_sentence_merge():
    document = source_document(
        ["Background.", "This study examined vitamin D supplementation."]
    )
    passages = build_merged_passages(document, "test-v1", 5, 120)
    assert len(passages) == 1
    assert passages[0]["metadata"]["source_sentence_indices"] == [0, 1]


def test_no_merge_at_threshold():
    document = source_document(["One two three four five.", "Next sentence."])
    passages = build_merged_passages(document, "test-v1", 5, 120)
    assert passages[0]["metadata"]["source_sentence_indices"] == [0]


def test_no_merge_above_maximum():
    document = source_document(["Short.", "one two three four five"])
    passages = build_merged_passages(document, "test-v1", 5, 5)
    assert len(passages) == 2


def test_oversized_sentence_failure():
    document = source_document(["one two three four"])
    with pytest.raises(CorpusBuildError, match="CORPUS_PASSAGE_TOO_LONG"):
        build_sentence_passages(document, "test-v1", 3)


def test_merged_gold_mapping_deduplicates_passage_id():
    document = source_document(["Background.", "A longer next sentence."])
    passages = build_merged_passages(document, "test-v1", 3, 120)
    gold = resolve_gold_evidence([source_claim([0, 1])], passages)
    assert gold[0]["evidence_sets"][0]["passage_ids"] == [
        "scifact:document:123:p:0"
    ]


def test_duplicate_passage_report():
    document = source_document(["Repeated.", "Repeated."])
    passages = sentence_passages(document)
    report = build_quality_report(
        [build_document_record(document, "test-v1")],
        passages,
        resolve_gold_evidence([], passages),
        {
            "merge_short_sentences": False,
            "short_sentence_word_threshold": 5,
            "max_passage_words": 120,
        },
        "test-v1",
    )
    assert report["exact_duplicate_passage_groups"] == 1
    assert report["exact_duplicate_passage_count"] == 2
    assert report["warnings"]


@pytest.mark.parametrize("version", ["", ".", "..", "../corpus", "a/b", "bad name"])
def test_version_validation_rejects_unsafe_values(version):
    with pytest.raises(CorpusBuildError, match="CORPUS_INVALID_VERSION"):
        validate_version(version)


@pytest.mark.parametrize("version", ["scifact-v1", "2026-08-v1", "sentence_1.0"])
def test_version_validation_accepts_safe_values(version):
    assert validate_version(version) == version


def test_existing_version_failure(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for filename in ("documents.jsonl", "claims.jsonl", "manifest.json"):
        (input_dir / filename).write_text("{}\n", encoding="utf-8")
    version_dir = tmp_path / "corpora" / "test-v1"
    version_dir.mkdir(parents=True)
    marker = version_dir / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(CorpusBuildError, match="CORPUS_VERSION_EXISTS"):
        build_scifact_corpus(input_dir, tmp_path / "corpora", "test-v1")
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_deterministic_in_memory_records():
    document = source_document()
    first_document = build_document_record(document, "test-v1")
    second_document = build_document_record(document, "test-v1")
    first_passages = sentence_passages(document)
    second_passages = sentence_passages(document)
    first_gold = resolve_gold_evidence([source_claim([1])], first_passages)
    second_gold = resolve_gold_evidence([source_claim([1])], second_passages)
    assert (first_document, first_passages, first_gold) == (
        second_document,
        second_passages,
        second_gold,
    )
