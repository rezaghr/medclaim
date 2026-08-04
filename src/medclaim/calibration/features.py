"""Dataset-agnostic runtime feature extraction for confidence calibration."""

from __future__ import annotations

import math
from typing import Any

LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")
FEATURE_NAMES = (
    "raw_confidence",
    "top_score",
    "mean_selected_reranker_score",
    "gate_proceeded",
    "selected_passage_count",
    "unique_document_count",
    "verifier_is_classifier",
    "predicted_supports",
    "predicted_refutes",
    "predicted_not_enough_info",
    "predicted_mixed",
    "decomposed",
)


class CalibrationFeatureError(ValueError):
    pass


def _result(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("result")
    return value if isinstance(value, dict) else record


def extract_confidence_features(record: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    if not isinstance(record, dict):
        raise CalibrationFeatureError("CALIBRATION_FEATURE_INVALID: Prediction must be an object.")
    result = _result(record)
    raw = result.get("raw_confidence", result.get("confidence", record.get("raw_confidence")))
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or not 0 <= float(raw) <= 1
    ):
        raise CalibrationFeatureError("CALIBRATION_FEATURE_INVALID: raw confidence must be from zero to one.")
    label = record.get("predicted_label", result.get("verdict"))
    if label not in LABELS:
        raise CalibrationFeatureError("CALIBRATION_FEATURE_INVALID: Predicted label is invalid.")
    candidates = record.get("selected_evidence", record.get("retrieved", record.get("results", [])))
    if not isinstance(candidates, list):
        candidates = []
    used_ids = result.get("evidence_used", [])
    if not isinstance(used_ids, list):
        used_ids = []
    components = result.get("component_results", [])
    gate_decision = record.get("gate_decision")
    if not isinstance(gate_decision, dict) and isinstance(components, list) and len(components) == 1 and isinstance(components[0], dict):
        gate_decision = components[0].get("gate_decision")
    component_decisions = [
        item.get("gate_decision") for item in components
        if isinstance(item, dict) and isinstance(item.get("gate_decision"), dict)
    ] if isinstance(components, list) else []
    gate_proceeded = (
        all(item.get("status") == "PROCEED" for item in component_decisions)
        if component_decisions
        else not isinstance(gate_decision, dict) or gate_decision.get("status") == "PROCEED"
    )
    selected = [] if not gate_proceeded else [
        item for item in candidates
        if isinstance(item, dict) and (not used_ids or item.get("passage_id") in used_ids)
    ]
    all_scores = [
        float(item["reranker_score"])
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("reranker_score"), (int, float))
        and not isinstance(item.get("reranker_score"), bool)
        and math.isfinite(float(item["reranker_score"]))
    ]
    selected_scores = [
        float(item["reranker_score"])
        for item in selected
        if isinstance(item.get("reranker_score"), (int, float))
        and not isinstance(item.get("reranker_score"), bool)
        and math.isfinite(float(item["reranker_score"]))
    ]
    documents = {item.get("document_id") for item in selected if isinstance(item.get("document_id"), str)}
    verifier = str(record.get("verifier_implementation", result.get("technical_metadata", {}).get("verifier_implementation", "llm"))).casefold()
    decomposed = isinstance(components, list) and len(components) > 1
    features = [
        float(raw),
        max(all_scores, default=0.0),
        sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
        1.0 if gate_proceeded else 0.0,
        float(len(selected) if selected else len(used_ids)),
        float(len(documents)),
        1.0 if verifier == "classifier" else 0.0,
        *(1.0 if label == candidate else 0.0 for candidate in LABELS),
        1.0 if decomposed else 0.0,
    ]
    metadata = {
        "raw_confidence": float(raw),
        "predicted_label": label,
        "selected_passage_count": int(features[4]),
        "unique_document_count": int(features[5]),
    }
    return features, metadata
