import csv
import json

from medclaim.explanation.evaluation import export_explanation_review


def test_review_export_is_deterministic_and_stratified(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "claim_id": f"c{i}", "dataset": "scifact" if i % 2 else "healthver",
            "claim": f"Claim {i}", "gold_label": "SUPPORTS" if i % 3 else "REFUTES",
            "predicted_label": "SUPPORTS" if i % 4 else "REFUTES",
            "explanation": "The selected evidence supports or contradicts the claim.",
            "evidence_used": [f"p{i}"], "attributions": [{"text": f"Evidence {i}"}],
        }
        for i in range(1, 13)
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = export_explanation_review(predictions, tmp_path / "first.csv", 8, 42)
    second = export_explanation_review(predictions, tmp_path / "second.csv", 8, 42)
    assert first.read_bytes() == second.read_bytes()
    output = list(csv.DictReader(first.open()))
    assert len(output) == 8
    assert {row["dataset"] for row in output} == {"scifact", "healthver"}
    assert output[0]["reviewer_notes"] == ""
