import pytest

from medclaim.evaluation.classification_metrics import classification_metrics


def test_perfect_classification_metrics():
    metrics = classification_metrics(
        ["SUPPORTS", "REFUTES"], ["SUPPORTS", "REFUTES"]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == {
        "labels": ["REFUTES", "SUPPORTS"],
        "matrix": [[1, 0], [0, 1]],
    }


def test_imperfect_macro_metrics():
    metrics = classification_metrics(
        ["SUPPORTS", "REFUTES"], ["SUPPORTS", "SUPPORTS"]
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_precision"] == pytest.approx(0.25)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
