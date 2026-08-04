import json
import shutil
from pathlib import Path

import pytest

from medclaim.corpus.scifact_corpus import CorpusBuildError, build_scifact_corpus

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed_scifact"
DETERMINISTIC_FILES = (
    "documents.jsonl",
    "passages.jsonl",
    "gold_evidence.jsonl",
    "quality_report.json",
)


def copy_fixtures(destination: Path) -> None:
    destination.mkdir(parents=True)
    for fixture in FIXTURES.iterdir():
        shutil.copy2(fixture, destination / fixture.name)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_scifact_corpus_end_to_end(tmp_path):
    input_dir = tmp_path / "processed_scifact"
    first_root = tmp_path / "first" / "corpora"
    second_root = tmp_path / "second" / "corpora"
    copy_fixtures(input_dir)
    source_before = {path.name: path.read_bytes() for path in input_dir.iterdir()}

    first_dir = build_scifact_corpus(input_dir, first_root, "scifact-v1")
    second_dir = build_scifact_corpus(input_dir, second_root, "scifact-v1")

    assert {path.name for path in first_dir.iterdir()} == {
        "documents.jsonl",
        "passages.jsonl",
        "gold_evidence.jsonl",
        "quality_report.json",
        "manifest.json",
    }
    documents = read_jsonl(first_dir / "documents.jsonl")
    passages = read_jsonl(first_dir / "passages.jsonl")
    gold = read_jsonl(first_dir / "gold_evidence.jsonl")
    report = json.loads((first_dir / "quality_report.json").read_text())
    manifest = json.loads((first_dir / "manifest.json").read_text())
    document_by_id = {document["document_id"]: document for document in documents}
    passage_ids = {passage["passage_id"] for passage in passages}

    assert all(passage["document_id"] in document_by_id for passage in passages)
    assert all(
        document_by_id[passage["document_id"]]["text"]
        [passage["start_char"] : passage["end_char"]]
        == passage["text"]
        for passage in passages
    )
    assert all(
        passage_id in passage_ids
        for claim in gold
        for evidence_set in claim["evidence_sets"]
        for passage_id in evidence_set["passage_ids"]
    )
    assert report["document_count"] == 2
    assert report["passage_count"] == 5
    assert report["gold_evidence_set_count"] == 3
    assert manifest["document_count"] == 2
    assert manifest["passage_count"] == 5
    assert manifest["claim_count"] == 3
    assert manifest["content_hash"].startswith("sha256:")
    second_manifest = json.loads((second_dir / "manifest.json").read_text())
    assert second_manifest["content_hash"] == manifest["content_hash"]
    assert all(
        (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
        for filename in DETERMINISTIC_FILES
    )
    serialized_outputs = "".join(
        path.read_text(encoding="utf-8") for path in first_dir.iterdir()
    )
    assert str(tmp_path) not in serialized_outputs
    assert {path.name: path.read_bytes() for path in input_dir.iterdir()} == source_before

    with pytest.raises(CorpusBuildError, match="CORPUS_VERSION_EXISTS"):
        build_scifact_corpus(input_dir, first_root, "scifact-v1")
