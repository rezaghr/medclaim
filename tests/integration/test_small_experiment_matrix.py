import json

from medclaim.experiments.configuration import load_experiment_configuration
from medclaim.experiments.runner import ExperimentRunner
from tests.unit.test_experiment_runner import config_value


def test_small_offline_four_mode_matrix(tmp_path):
    modes = (
        ("bm25", "llm", False),
        ("hybrid", "llm", False),
        ("hybrid-rerank", "llm", True),
        ("classifier", "classifier", True),
    )
    configurations = []
    for mode, verifier, reranking in modes:
        value = config_value(f"exp-{mode}")
        value["retrieval"]["mode"] = "bm25" if mode == "bm25" else "hybrid"
        value["reranking"]["enabled"] = reranking
        value["verifier"]["implementation"] = verifier
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(value) + "\n")
        configurations.append(load_experiment_configuration(path))

    calls = []

    def fake_pipeline(configuration):
        calls.append(configuration.experiment_id)
        return {
            "metrics": {
                "retrieval": {"recall_at_5": 1.0},
                "classification": {"accuracy": 1.0, "macro_f1": 1.0},
                "calibration": {"brier_score": 0.01, "ece": 0.02},
                "latency_cost": {"total_latency_ms": 1.0, "approximate_cost": 0.0},
                "per_dataset": {dataset: {"macro_f1": 1.0} for dataset in ("scifact", "healthver", "pubhealth")},
                "per_label": {label: {"f1": 1.0} for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")},
            },
            "predictions": [{"claim_id": configuration.experiment_id, "gold_label": "SUPPORTS", "predicted_label": "SUPPORTS", "confidence": 0.9}],
            "errors": [],
        }

    output, reused = ExperimentRunner({"fake": fake_pipeline}).run(
        configurations, tmp_path / "experiments", "tiny-matrix"
    )
    assert not reused
    assert calls == [configuration.experiment_id for configuration in configurations]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["completed_runs"] == 4
    assert len(list((output / "runs").iterdir())) == 4
