"""Unified medical claim labels and adapter mapping validation."""

from __future__ import annotations

from typing import Any

LABEL_SCHEMA_VERSION = "1.0.0"
UNIFIED_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")

LABEL_DESCRIPTIONS = {
    "SUPPORTS": "The available evidence directly supports the claim.",
    "REFUTES": "The available evidence directly contradicts the claim.",
    "NOT_ENOUGH_INFO": (
        "The available evidence does not establish support or contradiction."
    ),
    "MIXED": (
        "Different meaningful claim components have different relationships."
    ),
}


def extract_source_mapping(value: Any, dataset: str) -> dict[str, str]:
    """Read a mapping from supported adapter mapping-file shapes."""
    if not isinstance(value, dict):
        raise ValueError(f"Mapping for {dataset} must be a JSON object.")
    declared_dataset = value.get("dataset")
    if declared_dataset is not None and declared_dataset != dataset:
        raise ValueError(
            f"Mapping declares dataset {declared_dataset!r}, expected {dataset!r}."
        )
    candidate: Any = value
    for key in ("mappings", "label_mapping", "source_to_unified"):
        if key in value:
            candidate = value[key]
            break
    if not isinstance(candidate, dict) or not candidate:
        raise ValueError(f"Mapping for {dataset} is empty or invalid.")
    mapping: dict[str, str] = {}
    for source_label, unified_label in candidate.items():
        if not isinstance(source_label, str) or not source_label:
            raise ValueError(f"Mapping for {dataset} has an invalid source label.")
        if unified_label not in UNIFIED_LABELS:
            raise ValueError(
                f"Mapping for {dataset} maps {source_label!r} to unknown label "
                f"{unified_label!r}."
            )
        mapping[source_label] = unified_label
    return mapping


def build_label_schema(source_mappings: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "labels": {
            label: {"description": LABEL_DESCRIPTIONS[label]}
            for label in UNIFIED_LABELS
        },
        "source_mappings": source_mappings,
    }
