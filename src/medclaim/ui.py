"""Pure helpers for presenting verification evidence in UI clients."""

from __future__ import annotations

from typing import Any


def collect_used_evidence(verification: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve model-cited passage IDs to their full API evidence records."""
    evidence_used = verification.get("evidence_used", [])
    if not isinstance(evidence_used, list):
        return []

    records_by_id: dict[str, dict[str, Any]] = {}
    components = verification.get("component_results", [])
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            for field in ("retrieved_candidates", "model_input_evidence"):
                records = component.get(field, [])
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    passage_id = record.get("passage_id")
                    if not isinstance(passage_id, str) or not passage_id:
                        continue
                    records_by_id.setdefault(passage_id, {}).update(record)

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for passage_id in evidence_used:
        if not isinstance(passage_id, str) or passage_id in seen:
            continue
        seen.add(passage_id)
        resolved.append(
            {
                "passage_id": passage_id,
                **records_by_id.get(passage_id, {}),
            }
        )
    return resolved
