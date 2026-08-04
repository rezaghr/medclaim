"""Automated explanation checks and deterministic manual-review sampling."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .validation import ExplanationValidationError, ExplanationValidator


def evaluate_explanation(
    result: dict[str, Any],
    supplied_passages: list[dict[str, Any]],
    validator: ExplanationValidator | None = None,
    *,
    gate_abstained: bool = False,
) -> dict[str, Any]:
    validator = validator or ExplanationValidator()
    try:
        validation = validator.validate(
            result, supplied_passages, gate_abstained=gate_abstained
        )
        return {
            "status": "valid",
            **validation.to_dict(),
            "evidence_consistency": "deterministic_checks_passed",
            "unknown_fact_risk": "manual_review_required",
            "readability": {
                "average_words_per_sentence": _average_words_per_sentence(
                    result["explanation"]
                )
            },
            "llm_judge": {"enabled": False, "status": "not_run"},
        }
    except ExplanationValidationError as exc:
        return {
            "status": "invalid",
            "valid": False,
            "error": str(exc),
            "evidence_consistency": "failed_or_unavailable",
            "unknown_fact_risk": "manual_review_required",
            "llm_judge": {"enabled": False, "status": "not_run"},
        }


def _average_words_per_sentence(text: str) -> float:
    sentences = [part.strip() for part in text.replace("!", ".").replace("?", ".").split(".") if part.strip()]
    return round(sum(len(part.split()) for part in sentences) / len(sentences), 2) if sentences else 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Prediction line {number} is not an object.")
            rows.append(value)
    return rows


def export_explanation_review(
    predictions_path: Path,
    output_path: Path,
    sample_size: int = 100,
    seed: int = 42,
    stratify_by: tuple[str, ...] = ("dataset", "label", "correctness"),
) -> Path:
    if output_path.exists():
        raise ValueError(f"EXPLANATION_REVIEW_OUTPUT_EXISTS: {output_path} already exists.")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
        raise ValueError("EXPLANATION_REVIEW_INVALID: sample_size must be positive.")
    allowed = {"dataset", "label", "correctness"}
    if not stratify_by or not set(stratify_by) <= allowed:
        raise ValueError("EXPLANATION_REVIEW_INVALID: Unsupported stratification field.")
    rows = _load_jsonl(predictions_path)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result = row.get("result") if isinstance(row.get("result"), dict) else row
        gold = row.get("gold_label", row.get("unified_label"))
        predicted = row.get("predicted_label", result.get("verdict"))
        values = {
            "dataset": row.get("dataset"),
            "label": gold,
            "correctness": gold == predicted if gold is not None else None,
        }
        groups[tuple(values[field] for field in stratify_by)].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda row: str(row.get("claim_id", "")))
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups, key=lambda key: tuple(str(value) for value in key))
    while len(selected) < min(sample_size, len(rows)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < sample_size:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: str(row.get("claim_id", "")))
    fieldnames = (
        "claim_id", "dataset", "claim", "gold_label", "predicted_label",
        "explanation", "evidence_ids", "evidence_text",
        "evidence_consistency_rating", "label_consistency_rating",
        "coverage_rating", "readability_rating", "unsupported_content",
        "reviewer_notes",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            result = row.get("result") if isinstance(row.get("result"), dict) else row
            attributions = row.get("attributions", [])
            writer.writerow(
                {
                    "claim_id": row.get("claim_id"),
                    "dataset": row.get("dataset"),
                    "claim": row.get("claim", row.get("claim_text")),
                    "gold_label": row.get("gold_label", row.get("unified_label")),
                    "predicted_label": row.get("predicted_label", result.get("verdict")),
                    "explanation": result.get("explanation"),
                    "evidence_ids": " | ".join(result.get("evidence_used", [])),
                    "evidence_text": " | ".join(
                        str(item.get("text", "")) for item in attributions if isinstance(item, dict)
                    ),
                    "evidence_consistency_rating": "",
                    "label_consistency_rating": "",
                    "coverage_rating": "",
                    "readability_rating": "",
                    "unsupported_content": "",
                    "reviewer_notes": "",
                }
            )
    return output_path
