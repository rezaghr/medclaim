"""Offline metrics and immutable artifacts for US-013 prediction records."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .classification_metrics import classification_metrics


class GateDecompositionEvaluationError(Exception):
    """Raised when US-013 evaluation inputs or outputs are invalid."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                rows.append(row)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise GateDecompositionEvaluationError(f"GATE_EVALUATION_INVALID: Could not read {path}: {exc}.") from exc
    return rows


def _binary_metrics(gold: list[bool], predicted: list[bool]) -> dict[str, float]:
    def metrics(label: bool) -> tuple[float, float, float]:
        tp = sum(a == label and p == label for a, p in zip(gold, predicted, strict=True))
        fp = sum(a != label and p == label for a, p in zip(gold, predicted, strict=True))
        fn = sum(a == label and p != label for a, p in zip(gold, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    sufficient = metrics(True)
    insufficient = metrics(False)
    return {
        "sufficiency_precision": sufficient[0],
        "sufficiency_recall": sufficient[1],
        "sufficiency_f1": sufficient[2],
        "sufficiency_macro_f1": (sufficient[2] + insufficient[2]) / 2,
        "insufficient_evidence_recall": insufficient[1],
    }


def _decision(record: dict[str, Any]) -> dict[str, Any] | None:
    decision = record.get("gate_decision")
    if isinstance(decision, dict):
        return decision
    result = record.get("result")
    components = result.get("component_results") if isinstance(result, dict) else record.get("component_results")
    if isinstance(components, list) and len(components) == 1 and isinstance(components[0], dict):
        decision = components[0].get("gate_decision")
        return decision if isinstance(decision, dict) else None
    return None


def _verdict(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str):
        return value
    result = record.get("result")
    value = result.get(key.replace("predicted_", "")) if isinstance(result, dict) else None
    return value if isinstance(value, str) else None


def evaluate_gate_and_decomposition(predictions_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise GateDecompositionEvaluationError(f"GATE_EVALUATION_OUTPUT_EXISTS: {output_dir} already exists.")
    rows = _load_jsonl(predictions_path)
    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in rows:
        gold_label = _verdict(row, "gold_label")
        predicted_label = _verdict(row, "predicted_label")
        decision = _decision(row)
        if gold_label is None or predicted_label is None:
            errors.append({"claim_id": row.get("claim_id"), "error_type": "INVALID_PREDICTION", "reason": "Missing gold or predicted label."})
            continue
        valid.append({**row, "_gold": gold_label, "_predicted": predicted_label, "_decision": decision})
    classification = classification_metrics(
        [row["_gold"] for row in valid], [row["_predicted"] for row in valid]
    )
    gate_rows = [row for row in valid if isinstance(row.get("gold_sufficient"), bool) and row["_decision"] is not None]
    gold_sufficiency = [row["gold_sufficient"] for row in gate_rows]
    predicted_sufficiency = [row["_decision"].get("status") == "PROCEED" for row in gate_rows]
    gate_binary = _binary_metrics(gold_sufficiency, predicted_sufficiency) if gate_rows else {}
    abstained = sum(not value for value in predicted_sufficiency)
    proceeded = [row for row, proceed in zip(gate_rows, predicted_sufficiency, strict=True) if proceed]
    proceeded_accuracy = (
        sum(row["_gold"] == row["_predicted"] for row in proceeded) / len(proceeded)
        if proceeded else 0.0
    )
    component_rows = [
        component
        for row in valid
        for component in (
            row.get("result", {}).get("component_results", [])
            if isinstance(row.get("result"), dict)
            else row.get("component_results", [])
        )
        if isinstance(component, dict)
    ]
    avoided = sum(
        component.get("gate_decision", {}).get("status") == "ABSTAIN"
        for component in component_rows
    )
    gate_metrics = {
        "evaluated_sufficiency_claims": len(gate_rows),
        **gate_binary,
        "abstention_rate": abstained / len(gate_rows) if gate_rows else 0.0,
        "coverage": len(proceeded) / len(gate_rows) if gate_rows else 0.0,
        "risk": 1 - proceeded_accuracy if proceeded else 0.0,
        "accuracy_among_non_abstained_claims": proceeded_accuracy,
        "overall_accuracy": classification.get("accuracy"),
        "overall_macro_f1": classification.get("macro_f1"),
        "nei_metrics": classification.get("per_label", {}).get("NOT_ENOUGH_INFO"),
        "verifier_calls_avoided": avoided,
        "mean_verifier_calls_avoided": avoided / len(valid) if valid else 0.0,
        "mean_latency_saved_ms": mean(
            [float(row.get("estimated_latency_saved_ms", 0.0)) for row in valid]
        ) if valid else 0.0,
    }
    mixed_label = classification.get("per_label", {}).get("MIXED")
    decomposed = [row for row in valid if len(
        row.get("result", {}).get("component_results", [])
        if isinstance(row.get("result"), dict) else row.get("component_results", [])
    ) > 1]
    failures = sum(bool(row.get("decomposition_failed")) for row in valid)
    false_attempts = sum(bool(row.get("false_decomposition_attempt")) for row in valid)
    mixed_metrics = {
        "mixed_precision": mixed_label.get("precision") if mixed_label else None,
        "mixed_recall": mixed_label.get("recall") if mixed_label else None,
        "mixed_f1": mixed_label.get("f1") if mixed_label else None,
        "component_level_accuracy": None,
        "component_ground_truth_available": False,
        "limitation": "Component-level ground truth was not supplied.",
        "claims_decomposed": len(decomposed),
        "false_decomposition_attempts": false_attempts,
        "decomposition_failures": failures,
        "average_components_per_decomposed_claim": mean([
            len(row.get("result", {}).get("component_results", []) if isinstance(row.get("result"), dict) else row.get("component_results", []))
            for row in decomposed
        ]) if decomposed else 0.0,
    }
    curve: list[dict[str, Any]] = []
    scored = [row for row in gate_rows if isinstance(row["_decision"].get("top_score"), (int, float))]
    for threshold in sorted({float(row["_decision"]["threshold"]) for row in scored} | {float(row["_decision"]["top_score"]) for row in scored}):
        covered = [float(row["_decision"]["top_score"]) >= threshold for row in scored]
        selected = [row for row, keep in zip(scored, covered, strict=True) if keep]
        accuracy = sum(row["_gold"] == row["_predicted"] for row in selected) / len(selected) if selected else 0.0
        curve.append({"threshold": threshold, "coverage": sum(covered) / len(covered), "risk": 1 - accuracy if selected else 0.0})
    manifest = {
        "artifact_type": "gate_decomposition_evaluation",
        "version": output_dir.name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "evaluated_claims": len(valid),
        "error_count": len(errors),
        "thresholds_are_analysis_only": True,
        "outputs": {
            "predictions": "predictions.jsonl",
            "gate_metrics": "gate_metrics.json",
            "classification_metrics": "classification_metrics.json",
            "mixed_metrics": "mixed_metrics.json",
            "abstention_curve": "abstention_curve.csv",
            "errors": "errors.jsonl",
        },
    }
    temporary: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
        (temporary / "predictions.jsonl").write_bytes(predictions_path.read_bytes())
        for filename, value in (("gate_metrics.json", gate_metrics), ("classification_metrics.json", classification), ("mixed_metrics.json", mixed_metrics), ("manifest.json", manifest)):
            (temporary / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (temporary / "abstention_curve.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("threshold", "coverage", "risk"))
            writer.writeheader()
            writer.writerows(curve)
        (temporary / "errors.jsonl").write_text("".join(json.dumps(row) + "\n" for row in errors), encoding="utf-8")
        os.rename(temporary, output_dir)
    except Exception:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir
