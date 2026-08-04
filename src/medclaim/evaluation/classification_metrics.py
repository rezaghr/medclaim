"""Dependency-free multiclass classification metrics for verifier ablations."""

from __future__ import annotations

from typing import Any


def classification_metrics(
    gold_labels: list[str], predicted_labels: list[str]
) -> dict[str, Any]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted label counts must match.")
    if not gold_labels:
        return {"status": "not_available", "evaluated_claims": 0}
    labels = sorted(set(gold_labels) | set(predicted_labels))
    matrix = [
        [
            sum(
                gold == gold_label and predicted == predicted_label
                for gold, predicted in zip(
                    gold_labels, predicted_labels, strict=True
                )
            )
            for predicted_label in labels
        ]
        for gold_label in labels
    ]
    per_label: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted_count = sum(row[index] for row in matrix)
        gold_count = sum(matrix[index])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gold_count,
        }
    return {
        "status": "available",
        "evaluated_claims": len(gold_labels),
        "accuracy": sum(
            gold == predicted
            for gold, predicted in zip(gold_labels, predicted_labels, strict=True)
        )
        / len(gold_labels),
        "macro_precision": sum(item["precision"] for item in per_label.values())
        / len(labels),
        "macro_recall": sum(item["recall"] for item in per_label.values())
        / len(labels),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "per_label": per_label,
        "confusion_matrix": {"labels": labels, "matrix": matrix},
    }
