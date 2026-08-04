import hashlib
import json

import pytest

from medclaim.calibration.calibrator import (
    CalibrationError,
    evaluate_confidence_calibrator,
    fit_confidence_calibrator,
    load_confidence_calibrator,
    raw_confidence_result,
)
from medclaim.calibration.features import FEATURE_NAMES, extract_confidence_features


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def record(index, confidence, correct, split="dev"):
    predicted = "SUPPORTS" if index % 2 else "REFUTES"
    gold = predicted if correct else ("REFUTES" if predicted == "SUPPORTS" else "SUPPORTS")
    return {
        "claim_id": f"c{index}", "dataset": "scifact", "project_split": split,
        "gold_label": gold, "predicted_label": predicted, "confidence": confidence,
        "verifier_implementation": "classifier" if index % 2 else "llm",
        "evidence_used": [f"p{index}"],
        "retrieved": [{"passage_id": f"p{index}", "document_id": f"d{index % 3}", "reranker_score": confidence}],
        "gate_decision": {"status": "PROCEED"},
        "component_results": [],
    }


def fixture(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    splits = tmp_path / "splits.json"
    rows = [
        record(1, 0.95, True), record(2, 0.9, False), record(3, 0.8, True),
        record(4, 0.7, False), record(5, 0.6, True), record(6, 0.4, False),
        record(7, 0.3, True), record(8, 0.1, False), record(9, 0.99, True, "test"),
    ]
    write_jsonl(predictions, rows)
    write_json(splits, {"version": "splits-v1", "assignments": [
        {"claim_id": row["claim_id"], "project_split": row["project_split"]} for row in rows
    ]})
    return predictions, splits, rows


def test_features_are_fixed_and_have_no_target_leakage(tmp_path):
    _, _, rows = fixture(tmp_path)
    vector, metadata = extract_confidence_features(rows[0])
    assert len(vector) == len(FEATURE_NAMES)
    assert "gold_label" not in FEATURE_NAMES
    assert "claim_id" not in FEATURE_NAMES
    changed = dict(rows[0], gold_label="REFUTES", claim_id="another")
    assert extract_confidence_features(changed)[0] == vector
    assert metadata["predicted_label"] == "SUPPORTS"


@pytest.mark.parametrize("method", ["logistic", "isotonic", "none"])
def test_fit_load_and_predict_supported_methods(tmp_path, method):
    predictions, splits, _ = fixture(tmp_path)
    output = fit_confidence_calibrator(
        predictions, splits, method, tmp_path / "calibration", f"cal-{method}"
    )
    calibrator = load_confidence_calibrator(output)
    probabilities = calibrator.predict([record(10, 0.55, True)])
    assert len(probabilities) == 1
    assert 0 <= probabilities[0] <= 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["development_split"] == "dev"
    assert manifest["eligible_records"] == 8


def test_test_only_manifest_cannot_fit(tmp_path):
    predictions, splits, rows = fixture(tmp_path)
    write_json(splits, {"splits": {"test": [row["claim_id"] for row in rows]}})
    with pytest.raises(CalibrationError, match="NO_ELIGIBLE_RECORDS"):
        fit_confidence_calibrator(predictions, splits, "logistic", tmp_path / "out", "v1")


def test_checksum_and_feature_schema_validation(tmp_path):
    predictions, splits, _ = fixture(tmp_path)
    output = fit_confidence_calibrator(predictions, splits, "logistic", tmp_path / "out", "v1")
    model = output / "calibrator.pkl"
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(CalibrationError, match="CHECKSUM_MISMATCH"):
        load_confidence_calibrator(output)

    output = fit_confidence_calibrator(predictions, splits, "logistic", tmp_path / "out", "v2")
    schema_path = output / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_names"] = ["raw_confidence"]
    write_json(schema_path, schema)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["feature_schema"]["sha256"] = "sha256:" + hashlib.sha256(schema_path.read_bytes()).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(CalibrationError, match="FEATURE_SCHEMA_MISMATCH"):
        load_confidence_calibrator(output)


def test_raw_fallback_and_evaluation_outputs(tmp_path):
    predictions, splits, rows = fixture(tmp_path)
    raw = raw_confidence_result(rows[0])
    assert raw["calibrated_confidence"] is None
    assert raw["confidence_method"] == "raw"
    calibrator = fit_confidence_calibrator(predictions, splits, "logistic", tmp_path / "out", "v1")
    evaluation = evaluate_confidence_calibrator(predictions, calibrator, tmp_path / "evaluation")
    assert {path.name for path in evaluation.iterdir()} == {
        "predictions.jsonl", "metrics.json", "reliability_bins.csv",
        "reliability_diagram.png", "accuracy_coverage.csv", "risk_coverage.csv",
        "risk_coverage.png", "manifest.json",
    }
    assert (evaluation / "reliability_diagram.png").read_bytes().startswith(b"\x89PNG")
