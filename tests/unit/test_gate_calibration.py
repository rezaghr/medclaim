import csv
import json

import pytest

from medclaim.evidence_gate.calibration import calibrate_evidence_gate
from medclaim.evidence_gate.gate import EvidenceGateError


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def inputs(tmp_path):
    claims = tmp_path / "claims.jsonl"
    gold = tmp_path / "gold.jsonl"
    splits = tmp_path / "splits.json"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(claims, [
        {"claim_id": "c1", "unified_label": "SUPPORTS"},
        {"claim_id": "c2", "unified_label": "NOT_ENOUGH_INFO"},
        {"claim_id": "c3", "unified_label": "MIXED"},
    ])
    write_jsonl(gold, [
        {"claim_id": "c1", "evidence_sets": [{"passage_ids": ["p1"]}]},
        {"claim_id": "c2", "evidence_sets": []},
        {"claim_id": "c3", "evidence_sets": [{"passage_ids": ["p3"]}]},
    ])
    write_json(splits, {"version": "splits-v1", "assignments": [
        {"claim_id": "c1", "project_split": "dev"},
        {"claim_id": "c2", "project_split": "dev"},
        {"claim_id": "c3", "project_split": "dev"},
    ]})
    write_jsonl(predictions, [
        {"claim_id": "c1", "retrieved": [{"passage_id": "p1", "document_id": "d1", "reranker_score": 0.8}], "verification": {"verdict": "SUPPORTS"}},
        {"claim_id": "c2", "retrieved": [{"passage_id": "p2", "document_id": "d2", "reranker_score": 0.2}], "verification": {"verdict": "NOT_ENOUGH_INFO"}},
        {"claim_id": "c3", "retrieved": [{"passage_id": "p3", "document_id": "d3", "reranker_score": 0.9}]},
    ])
    return claims, gold, splits, predictions


def test_calibration_selects_development_threshold_and_writes_artifacts(tmp_path):
    paths = inputs(tmp_path)
    output = calibrate_evidence_gate(*paths, "dev", tmp_path / "gates", "gate-v1")
    assert {path.name for path in output.iterdir()} == {
        "config.json", "threshold_results.csv", "metrics.json", "manifest.json"
    }
    config = json.loads((output / "config.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert config["minimum_score"] == 0.8
    assert metrics["eligible_claims"] == 2
    assert metrics["exclusion_reasons"] == {"AMBIGUOUS_LABEL": 1}
    assert manifest["development_split"] == "dev"
    rows = list(csv.DictReader((output / "threshold_results.csv").open()))
    assert [float(row["threshold"]) for row in rows] == [0.2, 0.8]


def test_test_split_calibration_is_forbidden(tmp_path):
    with pytest.raises(EvidenceGateError, match="development-only"):
        calibrate_evidence_gate(*inputs(tmp_path), "test", tmp_path / "gates", "gate-v1")


def test_calibration_is_immutable(tmp_path):
    paths = inputs(tmp_path)
    calibrate_evidence_gate(*paths, "dev", tmp_path / "gates", "gate-v1")
    with pytest.raises(EvidenceGateError, match="GATE_CALIBRATION_OUTPUT_EXISTS"):
        calibrate_evidence_gate(*paths, "dev", tmp_path / "gates", "gate-v1")


def test_no_eligible_claims_fails(tmp_path):
    paths = inputs(tmp_path)
    write_json(paths[2], {"splits": {"test": ["c1", "c2", "c3"]}})
    with pytest.raises(EvidenceGateError, match="GATE_CALIBRATION_NO_ELIGIBLE_CLAIMS"):
        calibrate_evidence_gate(*paths, "dev", tmp_path / "gates", "gate-v1")
