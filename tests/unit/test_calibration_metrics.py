import pytest

from medclaim.calibration.metrics import calibration_metrics


def test_brier_ece_mce_and_empty_bins():
    metrics = calibration_metrics([0.1, 0.4, 0.8, 0.9], [0, 0, 1, 1], bins=5)
    assert metrics["brier_score"] == pytest.approx(0.055)
    assert metrics["expected_calibration_error"] == pytest.approx(0.2)
    assert metrics["maximum_calibration_error"] == pytest.approx(0.4)
    assert len(metrics["reliability_bins"]) == 5
    assert any(row["sample_count"] == 0 and row["calibration_gap"] is None for row in metrics["reliability_bins"])


def test_probabilities_must_be_valid():
    with pytest.raises(ValueError, match="INVALID"):
        calibration_metrics([1.2], [1])
