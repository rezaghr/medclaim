import json

from medclaim.evaluation.gate_decomposition import evaluate_gate_and_decomposition


def test_evaluation_writes_gate_classification_mixed_and_curve_artifacts(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "claim_id": "c1", "gold_label": "SUPPORTS", "predicted_label": "SUPPORTS",
            "gold_sufficient": True,
            "gate_decision": {"status": "PROCEED", "threshold": 0.5, "top_score": 0.8},
        },
        {
            "claim_id": "c2", "gold_label": "NOT_ENOUGH_INFO", "predicted_label": "NOT_ENOUGH_INFO",
            "gold_sufficient": False,
            "gate_decision": {"status": "ABSTAIN", "threshold": 0.5, "top_score": 0.2},
        },
        {
            "claim_id": "c3", "gold_label": "MIXED", "predicted_label": "MIXED",
            "result": {"component_results": [{"verdict": "SUPPORTS"}, {"verdict": "REFUTES"}]},
        },
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = evaluate_gate_and_decomposition(predictions, tmp_path / "evaluation")
    assert {path.name for path in output.iterdir()} == {
        "predictions.jsonl", "gate_metrics.json", "classification_metrics.json",
        "mixed_metrics.json", "abstention_curve.csv", "errors.jsonl", "manifest.json",
    }
    gate = json.loads((output / "gate_metrics.json").read_text())
    mixed = json.loads((output / "mixed_metrics.json").read_text())
    assert gate["sufficiency_macro_f1"] == 1.0
    assert gate["abstention_rate"] == 0.5
    assert mixed["mixed_f1"] == 1.0
    assert mixed["claims_decomposed"] == 1
