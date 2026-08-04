import json

import pytest

from medclaim.datasets.unified import UnifiedDatasetError, build_unified_dataset
from tests.multi_dataset_helpers import create_all_normalized_fixtures, read_jsonl, write_json, write_jsonl


def build_fixture(tmp_path, version="medical-dataset-v1"):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    output = build_unified_dataset(
        sources["scifact"], sources["healthver"], sources["pubhealth"], tmp_path / "datasets", version
    )
    return sources, output


def test_all_dataset_claims_are_preserved_in_common_schema(tmp_path):
    _, output = build_fixture(tmp_path)
    claims = read_jsonl(output / "claims.jsonl")
    assert [row["dataset"] for row in claims] == ["scifact", "healthver", "pubhealth", "pubhealth"]
    assert claims[0]["claim_id"] == "scifact:claim:1"
    assert claims[0]["original_label"] == "SUPPORT"
    by_id = {row["claim_id"]: row for row in claims}
    assert by_id["pubhealth:claim:3"]["unified_label"] == "MIXED"
    assert by_id["pubhealth:claim:3"]["gold_explanation"] == "Benefits and limitations coexist."
    assert by_id["pubhealth:claim:4"]["evidence_sets"] == []
    assert set(claims[0]) == {
        "claim_id", "dataset", "source_claim_id", "claim_text", "original_split",
        "original_label", "unified_label", "language", "evidence_sets",
        "gold_explanation", "metadata",
    }


def test_documents_and_evidence_relations_are_normalized(tmp_path):
    _, output = build_fixture(tmp_path)
    documents = read_jsonl(output / "documents.jsonl")
    relations = read_jsonl(output / "evidence_relations.jsonl")
    assert documents[0]["sentences"] == ["Vitamin D supports immune function."]
    assert documents[1]["source_url"] == "https://example.invalid/health"
    assert relations[0]["source_sentence_indices"] == [0]
    assert all(row["passage_ids"] == [] for row in relations)


def test_manifest_counts_hash_and_deterministic_outputs(tmp_path):
    sources, first = build_fixture(tmp_path / "first")
    second = build_unified_dataset(
        sources["scifact"], sources["healthver"], sources["pubhealth"], tmp_path / "second" / "datasets", "medical-dataset-v1"
    )
    for filename in ("claims.jsonl", "documents.jsonl", "evidence_relations.jsonl", "label_schema.json", "quality_report.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["claim_count"] == 4
    assert manifest["document_count"] == 3
    assert manifest["evidence_set_count"] == 3
    assert manifest["content_hash"].startswith("sha256:")


def test_source_label_mismatch_is_rejected(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    path = sources["pubhealth"] / "claims.jsonl"
    claims = read_jsonl(path)
    claims[0]["unified_label"] = "REFUTES"
    write_jsonl(path, claims)
    manifest = json.loads((sources["pubhealth"] / "manifest.json").read_text())
    manifest.pop("content_hash")
    write_json(sources["pubhealth"] / "manifest.json", manifest)
    with pytest.raises(UnifiedDatasetError, match="LABEL_MISMATCH"):
        build_unified_dataset(*sources.values(), tmp_path / "datasets", "v1")


def test_missing_document_and_namespace_collision_are_rejected(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "missing")
    path = sources["healthver"] / "claims.jsonl"
    claims = read_jsonl(path)
    claims[0]["evidence_sets"][0]["document_id"] = "healthver:document:999"
    write_jsonl(path, claims)
    manifest = json.loads((sources["healthver"] / "manifest.json").read_text())
    manifest.pop("content_hash")
    write_json(sources["healthver"] / "manifest.json", manifest)
    with pytest.raises(UnifiedDatasetError, match="MISSING_DOCUMENT"):
        build_unified_dataset(*sources.values(), tmp_path / "datasets", "v1")

    sources = create_all_normalized_fixtures(tmp_path / "duplicate")
    path = sources["scifact"] / "claims.jsonl"
    claims = read_jsonl(path)
    claims.append(dict(claims[0]))
    write_jsonl(path, claims)
    manifest = json.loads((sources["scifact"] / "manifest.json").read_text())
    manifest["claim_count"] = 2
    manifest.pop("content_hash")
    write_json(sources["scifact"] / "manifest.json", manifest)
    with pytest.raises(UnifiedDatasetError, match="NAMESPACE_COLLISION"):
        build_unified_dataset(*sources.values(), tmp_path / "datasets2", "v1")


def test_source_count_and_content_hash_validation_and_immutability(tmp_path):
    sources = create_all_normalized_fixtures(tmp_path / "processed")
    manifest_path = sources["scifact"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["content_hash"] = "sha256:" + "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(UnifiedDatasetError, match="CONTENT_HASH_MISMATCH"):
        build_unified_dataset(*sources.values(), tmp_path / "datasets", "v1")

    manifest["content_hash"] = None
    manifest["claim_count"] = 99
    write_json(manifest_path, manifest)
    with pytest.raises(UnifiedDatasetError, match="COUNT_MISMATCH"):
        build_unified_dataset(*sources.values(), tmp_path / "datasets", "v1")

    sources = create_all_normalized_fixtures(tmp_path / "fresh")
    build_unified_dataset(*sources.values(), tmp_path / "immutable", "v1")
    with pytest.raises(UnifiedDatasetError, match="VERSION_EXISTS"):
        build_unified_dataset(*sources.values(), tmp_path / "immutable", "v1")
