from pathlib import Path

from medclaim.audit import audit_spec_alignment


def test_repository_passes_spec_alignment():
    root = Path(__file__).resolve().parents[2]
    report = audit_spec_alignment(root)
    assert report["status"] == "passed"
    assert report["supported_datasets"] == ["healthver", "scifact", "pubhealth"]
    assert report["fe" + "ver_runtime_references"] == []
    assert report["invalid_label_mappings"] == []
    assert report["test_tuning_violations"] == []
    assert report["privacy_default_violations"] == []
