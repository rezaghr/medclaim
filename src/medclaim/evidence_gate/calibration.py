"""Deterministic development-only calibration for the evidence gate."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gate import EvidenceGate, EvidenceGateConfiguration, EvidenceGateError


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGateError(f"GATE_CONFIG_NOT_FOUND: Required input does not exist: {path}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceGateError(f"GATE_CONFIG_INVALID: Could not read {path}: {exc}.") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                rows.append(value)
    except FileNotFoundError as exc:
        raise EvidenceGateError(f"GATE_CONFIG_NOT_FOUND: Required input does not exist: {path}.") from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceGateError(f"GATE_CONFIG_INVALID: Could not parse {path} line {locals().get('number', 0)}: {exc}.") from exc
    return rows


def _index(rows: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in result:
            raise EvidenceGateError(f"GATE_CONFIG_INVALID: Invalid or duplicate {kind} claim_id {claim_id!r}.")
        result[claim_id] = row
    return result


def _split_assignments(value: Any) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise EvidenceGateError("GATE_CONFIG_INVALID: Split manifest must be an object.")
    version = str(value.get("version", value.get("split_version", "unknown")))
    assignments: dict[str, str] = {}
    raw_assignments = value.get("assignments")
    if isinstance(raw_assignments, list):
        for row in raw_assignments:
            if isinstance(row, dict) and isinstance(row.get("claim_id"), str):
                split = row.get("project_split", row.get("split"))
                if isinstance(split, str):
                    assignments[row["claim_id"]] = split
    raw_map = value.get("claim_splits")
    if isinstance(raw_map, dict):
        assignments.update({str(key): str(split) for key, split in raw_map.items()})
    raw_splits = value.get("splits")
    if isinstance(raw_splits, dict):
        for split, claim_ids in raw_splits.items():
            if isinstance(claim_ids, list):
                for claim_id in claim_ids:
                    assignments[str(claim_id)] = str(split)
    if not assignments:
        raise EvidenceGateError("GATE_CONFIG_INVALID: Split manifest has no claim assignments.")
    return assignments, version


def _binary_metrics(gold: list[bool], predicted: list[bool]) -> dict[str, float]:
    def label_metrics(label: bool) -> tuple[float, float, float]:
        tp = sum(actual == label and guess == label for actual, guess in zip(gold, predicted, strict=True))
        fp = sum(actual != label and guess == label for actual, guess in zip(gold, predicted, strict=True))
        fn = sum(actual == label and guess != label for actual, guess in zip(gold, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    sufficient = label_metrics(True)
    insufficient = label_metrics(False)
    return {
        "sufficiency_precision": sufficient[0],
        "sufficiency_recall": sufficient[1],
        "sufficiency_f1": sufficient[2],
        "insufficient_evidence_recall": insufficient[1],
        "sufficiency_macro_f1": (sufficient[2] + insufficient[2]) / 2,
    }


def _gold_sets(record: dict[str, Any]) -> list[set[str]]:
    evidence_sets = record.get("evidence_sets", [])
    if not isinstance(evidence_sets, list):
        return []
    output = []
    for item in evidence_sets:
        passage_ids = item.get("passage_ids") if isinstance(item, dict) else None
        if isinstance(passage_ids, list) and passage_ids and all(isinstance(value, str) for value in passage_ids):
            output.append(set(passage_ids))
    return output


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def calibrate_evidence_gate(
    claims_path: Path,
    gold_evidence_path: Path,
    split_manifest_path: Path,
    predictions_path: Path,
    split: str,
    output_root: Path,
    version: str,
    minimum_relevant_passages: int = 1,
    minimum_unique_documents: int = 1,
    score_field: str = "reranker_score",
) -> Path:
    """Calibrate a gate from development predictions and write immutable artifacts."""
    if split not in {"dev", "validation"}:
        raise EvidenceGateError("GATE_CONFIG_INVALID: Calibration is development-only; test split is forbidden.")
    output_dir = output_root / version
    if output_dir.exists():
        raise EvidenceGateError(f"GATE_CALIBRATION_OUTPUT_EXISTS: Version {version!r} already exists.")
    claims = _index(_load_jsonl(claims_path), "claim")
    gold = _index(_load_jsonl(gold_evidence_path), "gold evidence")
    predictions = _index(_load_jsonl(predictions_path), "prediction")
    assignments, split_version = _split_assignments(_load_json(split_manifest_path))
    exclusions: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    normalized_split = "dev" if split == "validation" else split
    for claim_id in sorted(claims):
        assigned = assignments.get(claim_id)
        assigned = "dev" if assigned == "validation" else assigned
        if assigned != normalized_split:
            continue
        claim = claims[claim_id]
        gold_record = gold.get(claim_id)
        prediction = predictions.get(claim_id)
        if gold_record is None or prediction is None:
            exclusions["MISSING_GOLD_OR_PREDICTION"] += 1
            continue
        label = claim.get("unified_label")
        if label is None or label == "MIXED":
            exclusions["AMBIGUOUS_LABEL"] += 1
            continue
        candidates = prediction.get("retrieved", prediction.get("results"))
        if not isinstance(candidates, list):
            exclusions["INVALID_PREDICTION"] += 1
            continue
        gold_sets = _gold_sets(gold_record)
        if not gold_sets and label != "NOT_ENOUGH_INFO":
            exclusions["UNAVAILABLE_GOLD_SEMANTICS"] += 1
            continue
        retrieved_ids = {
            item.get("passage_id") for item in candidates if isinstance(item, dict)
        }
        sufficient = label != "NOT_ENOUGH_INFO" and any(
            item <= retrieved_ids for item in gold_sets
        )
        scores: list[float] = []
        invalid = False
        for item in candidates:
            value = item.get(score_field) if isinstance(item, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                invalid = True
                break
            scores.append(float(value))
        if invalid:
            exclusions["INVALID_RERANKER_SCORE"] += 1
            continue
        verification = prediction.get("verification")
        predicted_label = verification.get("verdict") if isinstance(verification, dict) else None
        examples.append(
            {
                "claim_id": claim_id,
                "gold_sufficient": sufficient,
                "gold_label": label,
                "predicted_label": predicted_label,
                "candidates": candidates,
                "scores": scores,
                "corpus_version": prediction.get("corpus_version", "unknown"),
                "reranker_model": prediction.get("reranker_model", "unknown"),
            }
        )
    thresholds = sorted({score for example in examples for score in example["scores"]})
    if not examples or not thresholds:
        raise EvidenceGateError("GATE_CALIBRATION_NO_ELIGIBLE_CLAIMS: No usable development examples or scores.")
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        configuration = EvidenceGateConfiguration(
            version=version,
            enabled=True,
            minimum_score=threshold,
            minimum_relevant_passages=minimum_relevant_passages,
            minimum_unique_documents=minimum_unique_documents,
            score_field=score_field,
        )
        gate = EvidenceGate(configuration)
        proceeded = [gate.decide(example["candidates"]).status == "PROCEED" for example in examples]
        metrics = _binary_metrics([example["gold_sufficient"] for example in examples], proceeded)
        classified = [
            example for example, proceed in zip(examples, proceeded, strict=True)
            if proceed and isinstance(example["predicted_label"], str)
        ]
        accuracy = (
            sum(item["gold_label"] == item["predicted_label"] for item in classified) / len(classified)
            if classified else None
        )
        results.append(
            {
                "threshold": threshold,
                **metrics,
                "coverage": sum(proceeded) / len(proceeded),
                "classification_accuracy_among_proceeded": accuracy,
            }
        )
    selected = max(
        results,
        key=lambda row: (
            row["sufficiency_macro_f1"],
            row["insufficient_evidence_recall"],
            row["threshold"],
        ),
    )
    configuration = EvidenceGateConfiguration(
        version=version,
        enabled=True,
        minimum_score=selected["threshold"],
        minimum_relevant_passages=minimum_relevant_passages,
        minimum_unique_documents=minimum_unique_documents,
        score_field=score_field,
    )
    metrics = {
        "development_split": normalized_split,
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "eligible_claims": len(examples),
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
    }
    manifest = {
        "artifact_type": "evidence_gate",
        "version": version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "development_split": normalized_split,
        "split_manifest_version": split_version,
        "corpus_version": examples[0]["corpus_version"],
        "retrieval_configuration": "hybrid-reranked",
        "reranker_model": examples[0]["reranker_model"],
        "selected_threshold": selected["threshold"],
        "minimum_relevant_passages": minimum_relevant_passages,
        "minimum_unique_documents": minimum_unique_documents,
        "eligible_claims": len(examples),
        "excluded_claims": sum(exclusions.values()),
        "outputs": {
            "config": "config.json",
            "threshold_results": "threshold_results.csv",
            "metrics": "metrics.json",
        },
    }
    temporary: Path | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root))
        _write_json(temporary / "config.json", configuration.to_dict())
        with (temporary / "threshold_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output_dir)
    except FileExistsError as exc:
        raise EvidenceGateError(f"GATE_CALIBRATION_OUTPUT_EXISTS: Version {version!r} already exists.") from exc
    except (OSError, ValueError) as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise EvidenceGateError(f"GATE_CONFIG_INVALID: Could not write calibration artifacts: {exc}.") from exc
    return output_dir
