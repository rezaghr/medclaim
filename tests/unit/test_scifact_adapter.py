import json
import shutil
from pathlib import Path

import pytest

from medclaim.datasets.scifact import (
    SciFactPreparationError,
    _normalize_claims,
    _normalize_documents,
    claim_id_for,
    clean_claim_text,
    document_id_for,
    map_scifact_label,
    normalize_scifact_claim,
    normalize_scifact_document,
    prepare_scifact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scifact"


def source_document(doc_id=10, sentence_count=3):
    return {
        "doc_id": doc_id,
        "title": "A title",
        "abstract": [f"Sentence {index}." for index in range(sentence_count)],
        "structured": False,
    }


def documents_by_id(*records):
    documents = [normalize_scifact_document(record) for record in records]
    return {document["document_id"]: document for document in documents}


def source_claim(claim_id=42, label="SUPPORT", sentences=None, doc_id=10):
    sentences = [0, 2] if sentences is None else sentences
    return {
        "id": claim_id,
        "claim": "A scientific claim.",
        "evidence": {
            str(doc_id): [{"label": label, "sentences": sentences}]
        },
        "cited_doc_ids": [doc_id],
    }


def copy_fixtures(destination: Path):
    destination.mkdir(parents=True)
    for fixture in FIXTURES.glob("*.jsonl"):
        shutil.copy2(fixture, destination / fixture.name)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_support_mapping():
    assert map_scifact_label("SUPPORT") == "SUPPORTS"


def test_contradict_mapping():
    assert map_scifact_label("CONTRADICT") == "REFUTES"


def test_unknown_label_rejection():
    with pytest.raises(SciFactPreparationError, match="SCIFACT_UNKNOWN_LABEL"):
        map_scifact_label("PARTIAL_SUPPORT", "scifact:claim:42")


def test_stable_claim_id():
    assert claim_id_for(42) == "scifact:claim:42"


def test_stable_document_id():
    assert document_id_for(123456) == "scifact:document:123456"


def test_minimal_claim_cleaning():
    assert clean_claim_text("  Vitamin D   affects immunity.  ") == (
        "Vitamin D affects immunity."
    )


def test_abstract_sentence_preservation():
    source = source_document()
    normalized = normalize_scifact_document(source)

    assert normalized["abstract_sentences"] == source["abstract"]
    assert normalized["text"] == " ".join(source["abstract"])


def test_evidence_reference_preservation():
    documents = documents_by_id(source_document())
    normalized = normalize_scifact_claim(source_claim(), "train", documents)

    assert normalized["evidence_sets"] == [
        {
            "evidence_set_id": "scifact:claim:42:evidence:0",
            "relationship": "SUPPORTS",
            "document_id": "scifact:document:10",
            "sentence_indices": [0, 2],
        }
    ]


def test_missing_document_validation():
    with pytest.raises(SciFactPreparationError, match="SCIFACT_MISSING_DOCUMENT"):
        normalize_scifact_claim(source_claim(doc_id=999), "train", {})


def test_missing_document_writes_failure_quality_report(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)
    train_path = raw / "claims_train.jsonl"
    records = read_jsonl(train_path)
    records[0] = source_claim(claim_id=1, doc_id=999, sentences=[0])
    train_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(SciFactPreparationError, match="SCIFACT_MISSING_DOCUMENT"):
        prepare_scifact(raw, output)

    report = json.loads(
        (output / "quality_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["missing_document_references"] == 1
    assert report["invalid_sentence_references"] == 0
    assert "SCIFACT_MISSING_DOCUMENT" in report["errors"][0]
    assert not (output / "claims.jsonl").exists()


def test_out_of_range_sentence_validation():
    documents = documents_by_id(source_document(sentence_count=2))

    with pytest.raises(
        SciFactPreparationError, match="SCIFACT_INVALID_SENTENCE_INDEX.*index 2"
    ):
        normalize_scifact_claim(
            source_claim(sentences=[2]), "train", documents
        )


def test_duplicate_claim_detection():
    documents = documents_by_id(source_document())
    duplicate = source_claim()

    with pytest.raises(SciFactPreparationError, match="SCIFACT_DUPLICATE_CLAIM_ID"):
        _normalize_claims({"train": [duplicate], "dev": [duplicate]}, documents)


def test_duplicate_document_detection():
    duplicate = source_document()

    with pytest.raises(
        SciFactPreparationError, match="SCIFACT_DUPLICATE_DOCUMENT_ID"
    ):
        _normalize_documents([duplicate, duplicate])


def test_split_preservation():
    documents = documents_by_id(source_document())
    normalized = normalize_scifact_claim(source_claim(), "train", documents)

    assert normalized["original_split"] == "train"


def test_unlabeled_record_handling():
    normalized = normalize_scifact_claim(
        {"id": 7, "claim": "An unlabeled claim."}, "test", {}
    )

    assert normalized["original_label"] is None
    assert normalized["unified_label"] is None
    assert normalized["evidence_sets"] == []


def test_deterministic_normalized_output(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)

    prepare_scifact(raw, output)
    first_claims = (output / "claims.jsonl").read_bytes()
    first_documents = (output / "documents.jsonl").read_bytes()
    first_report = (output / "quality_report.json").read_bytes()
    prepare_scifact(raw, output, overwrite=True)

    assert (output / "claims.jsonl").read_bytes() == first_claims
    assert (output / "documents.jsonl").read_bytes() == first_documents
    assert (output / "quality_report.json").read_bytes() == first_report


def test_quality_report_counts(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)

    report = prepare_scifact(raw, output)

    assert report["document_count"] == 3
    assert report["claim_count"] == 6
    assert report["split_counts"] == {"train": 3, "dev": 2, "test": 1}
    assert report["original_label_counts"] == {
        "CONTRADICT": 2,
        "SUPPORT": 2,
        "UNLABELED": 2,
    }
    assert report["unified_label_counts"] == {
        "REFUTES": 2,
        "SUPPORTS": 2,
        "UNLABELED": 2,
    }
    assert report["evidence_set_count"] == 5


def test_prepare_scifact_integration(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)
    raw_before = {path.name: path.read_bytes() for path in raw.iterdir()}

    prepare_scifact(raw, output)

    expected_files = {
        "claims.jsonl",
        "documents.jsonl",
        "label_mapping.json",
        "quality_report.json",
        "manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    claims = read_jsonl(output / "claims.jsonl")
    documents = read_jsonl(output / "documents.jsonl")
    quality_report = json.loads(
        (output / "quality_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    document_ids = {document["document_id"] for document in documents}
    assert all(
        evidence["document_id"] in document_ids
        for claim in claims
        for evidence in claim["evidence_sets"]
    )
    assert quality_report["status"] == "success"
    assert manifest["artifact_type"] == "normalized_dataset"

    first_normalized = {
        filename: (output / filename).read_bytes()
        for filename in ("claims.jsonl", "documents.jsonl", "quality_report.json")
    }
    unrelated = output / "developer_notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    prepare_scifact(raw, output, overwrite=True)
    assert all(
        (output / filename).read_bytes() == contents
        for filename, contents in first_normalized.items()
    )
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert {path.name: path.read_bytes() for path in raw.iterdir()} == raw_before


def test_existing_outputs_require_overwrite(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)
    prepare_scifact(raw, output)

    with pytest.raises(SciFactPreparationError, match="Use --overwrite"):
        prepare_scifact(raw, output)


def test_missing_test_split_adds_warning(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    copy_fixtures(raw)
    (raw / "claims_test.jsonl").unlink()

    report = prepare_scifact(raw, output)

    assert report["split_counts"]["test"] == 0
    assert report["warnings"]


def test_invalid_json_reports_file_and_line(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"valid": true}\n{broken}\n', encoding="utf-8")

    from medclaim.datasets.scifact import load_jsonl

    with pytest.raises(
        SciFactPreparationError, match=r"broken\.jsonl at line 2"
    ):
        load_jsonl(source)


def test_non_standard_nan_is_rejected(tmp_path):
    source = tmp_path / "nan.jsonl"
    source.write_text('{"doc_id": NaN}\n', encoding="utf-8")

    from medclaim.datasets.scifact import load_jsonl

    with pytest.raises(SciFactPreparationError, match="SCIFACT_INVALID_JSON"):
        load_jsonl(source)
