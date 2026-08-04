"""Calibration, reliability, and risk-coverage metrics."""

from __future__ import annotations

import math
from typing import Any


def calibration_metrics(
    probabilities: list[float], targets: list[int], bins: int = 10
) -> dict[str, Any]:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("CALIBRATION_METRICS_INVALID: Non-empty aligned inputs are required.")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("CALIBRATION_METRICS_INVALID: bins must be positive.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        for value in probabilities
    ) or any(value not in (0, 1) for value in targets):
        raise ValueError("CALIBRATION_METRICS_INVALID: Invalid probability or target.")
    reliability = []
    gaps = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position for position, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1)
        ]
        mean_confidence = sum(probabilities[position] for position in selected) / len(selected) if selected else None
        observed = sum(targets[position] for position in selected) / len(selected) if selected else None
        gap = abs(mean_confidence - observed) if selected else None
        reliability.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": len(selected),
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed,
                "calibration_gap": gap,
            }
        )
        if gap is not None:
            gaps.append((len(selected), gap))
    epsilon = 1e-15
    clipped = [min(1 - epsilon, max(epsilon, float(value))) for value in probabilities]
    brier = sum((probability - target) ** 2 for probability, target in zip(probabilities, targets, strict=True)) / len(targets)
    log_loss = -sum(
        target * math.log(probability) + (1 - target) * math.log(1 - probability)
        for probability, target in zip(clipped, targets, strict=True)
    ) / len(targets)
    ece = sum(count * gap for count, gap in gaps) / len(targets)
    mce = max((gap for _, gap in gaps), default=0.0)
    accuracy = sum(targets) / len(targets)
    coverage_rows = accuracy_coverage(probabilities, targets)
    return {
        "sample_count": len(targets),
        "brier_score": brier,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "coverage": 1.0,
        "risk": 1 - accuracy,
        "reliability_bins": reliability,
        "accuracy_coverage": coverage_rows,
        "risk_coverage": [
            {"coverage": row["coverage"], "risk": 1 - row["accuracy"], "threshold": row["threshold"]}
            for row in coverage_rows
        ],
    }


def accuracy_coverage(probabilities: list[float], targets: list[int]) -> list[dict[str, float]]:
    rows = []
    for threshold in sorted(set(probabilities)):
        selected = [target for probability, target in zip(probabilities, targets, strict=True) if probability >= threshold]
        if selected:
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": len(selected) / len(targets),
                    "accuracy": sum(selected) / len(selected),
                }
            )
    return rows
