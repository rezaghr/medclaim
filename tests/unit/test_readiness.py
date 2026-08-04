import json

from medclaim.corpus.scifact_corpus import corpus_content_hash
from medclaim.runtime.configuration import RuntimeSettings
from medclaim.runtime.readiness import readiness_snapshot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def settings(tmp_path):
    corpus = tmp_path / "corpus"
    sparse = tmp_path / "bm25"
    dense = tmp_path / "dense"
    corpus.mkdir()
    passages = [{"passage_id": "p1"}]
    (corpus / "passages.jsonl").write_text(json.dumps(passages[0]) + "\n", encoding="utf-8")
    corpus_manifest = {
        "artifact_type": "medical_evidence_corpus",
        "corpus_version": "medical-v1",
        "content_hash": corpus_content_hash(passages),
        "passage_count": 1,
        "datasets": ["scifact", "healthver", "pubhealth"],
    }
    write_json(corpus / "manifest.json", corpus_manifest)
    reference = {
        "version": "medical-v1",
        "content_hash": corpus_content_hash(passages),
        "passage_count": 1,
    }
    write_json(sparse / "manifest.json", {"artifact_type": "bm25_index", "corpus": reference})
    write_json(
        dense / "manifest.json",
        {
            "artifact_type": "dense_index",
            "corpus": reference,
            "embedding": {"dimension": 384},
        },
    )
    return RuntimeSettings(
        corpus_dir=corpus,
        bm25_index_dir=sparse,
        dense_index_dir=dense,
        reranker_model="reranker-v1",
    )


def test_readiness_passes_compatible_artifacts(tmp_path):
    assert readiness_snapshot(settings(tmp_path))["status"] == "ready"


def test_readiness_fails_controlled_for_incompatible_index(tmp_path):
    configured = settings(tmp_path)
    manifest_path = configured.bm25_index_dir / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["corpus"]["version"] = "wrong"
    write_json(manifest_path, value)
    result = readiness_snapshot(configured)
    assert result["status"] == "not_ready"
    assert "bm25_index" in result["failed_checks"]
