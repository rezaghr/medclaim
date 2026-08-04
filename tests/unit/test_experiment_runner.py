import json

import pytest

from medclaim.experiments.configuration import (
    ExperimentConfigurationError,
    load_experiment_configuration,
)
from medclaim.experiments.runner import ExperimentRunner, ExperimentRunnerError


def config_value(experiment_id="exp-test", **overrides):
    value = {
        "experiment_id": experiment_id,
        "dataset_version": "dataset-v1",
        "split_manifest_version": "splits-v1",
        "corpus_version": "corpus-v1",
        "corpus_content_hash": "sha256:" + "a" * 64,
        "datasets": ["scifact", "healthver", "pubhealth"],
        "split": "test",
        "labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED"],
        "retrieval": {
            "mode": "hybrid", "sparse_top_k": 50, "dense_top_k": 50,
            "fusion_top_k": 30, "rrf_k": 60, "bm25_index_version": "bm25-v1",
            "dense_index_version": "dense-v1",
        },
        "reranking": {"enabled": True, "candidate_count": 20, "final_evidence_k": 5, "model_version": "reranker-v1"},
        "gate": {"enabled": True, "version": "gate-v1"},
        "decomposition": {"mode": "auto"},
        "verifier": {"implementation": "llm", "model_version": "llm-v1", "prompt_version": "prompt-v1"},
        "calibration": {"version": "cal-v1", "method": "logistic"},
        "seed": 42,
        "oracle_evidence": False,
        "executor": "fake",
    }
    value.update(overrides)
    return value


def load(tmp_path, value, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value) + "\n")
    return load_experiment_configuration(path)


def result(configuration):
    return {
        "metrics": {
            "retrieval": {"recall_at_5": 0.8, "mrr": 0.7},
            "classification": {"accuracy": 0.75, "macro_f1": 0.7},
            "calibration": {"brier_score": 0.2, "ece": 0.1, "risk_coverage": [{"coverage": 1.0, "risk": 0.25}]},
            "latency_cost": {"total_latency_ms": 20.0, "approximate_cost": 0.01},
            "per_dataset": {"scifact": {"macro_f1": 0.8}, "healthver": {"macro_f1": 0.7}, "pubhealth": {"macro_f1": 0.6}},
            "per_label": {"SUPPORTS": {"f1": 0.8}, "MIXED": {"f1": 0.5}},
        },
        "predictions": [{
            "claim_id": f"{configuration.experiment_id}:c1", "gold_label": "SUPPORTS",
            "predicted_label": "SUPPORTS", "confidence": 0.9,
        }],
        "errors": [],
    }


def test_config_validation_unknown_field_and_hash_stability(tmp_path):
    first = load(tmp_path, config_value(), "first.json")
    second = load(tmp_path, config_value(), "second.json")
    assert first.configuration_hash == second.configuration_hash
    invalid = config_value()
    invalid["mystery"] = True
    with pytest.raises(ExperimentConfigurationError, match="UNKNOWN_FIELD"):
        load(tmp_path, invalid, "invalid.json")


def test_three_label_run_must_exclude_mixed_instead_of_remapping(tmp_path):
    value = config_value()
    value["labels"] = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    with pytest.raises(ExperimentConfigurationError, match="explicitly exclude MIXED"):
        load(tmp_path, value, "missing-policy.json")
    value["mixed_policy"] = "exclude"
    assert load(tmp_path, value, "excluded.json").value["mixed_policy"] == "exclude"


def test_multi_run_expansion(tmp_path):
    value = config_value()
    value["variants"] = [
        {"suffix": "llm", "overrides": {}},
        {"suffix": "classifier", "overrides": {"verifier": {"implementation": "classifier", "model_version": "classifier-v1"}}},
    ]
    expanded = load(tmp_path, value).expand()
    assert [item.experiment_id for item in expanded] == ["exp-test-llm", "exp-test-classifier"]


def test_single_run_reports_reuse_and_figures(tmp_path):
    configuration = load(tmp_path, config_value())
    runner = ExperimentRunner({"fake": result})
    output, reused = runner.run([configuration], tmp_path / "experiments", "group-v1")
    assert reused is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["completed_runs"] == 1
    assert (output / "figures" / "macro_f1_comparison.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "per_dataset_metrics.csv").read_text().count("scifact") == 1
    assert (output / "per_label_metrics.csv").read_text().count("SUPPORTS") == 1
    reused_output, reused = runner.run([configuration], tmp_path / "experiments", "group-v1")
    assert reused and reused_output == output


@pytest.mark.parametrize("field", ["model_version", "prompt_version"])
def test_reuse_rejected_after_verifier_compatibility_change(tmp_path, field):
    configuration = load(tmp_path, config_value(), "base.json")
    runner = ExperimentRunner({"fake": result})
    runner.run([configuration], tmp_path / "experiments", "group-v1")
    changed = config_value()
    changed["verifier"][field] = "changed-v2"
    changed_config = load(tmp_path, changed, "changed.json")
    with pytest.raises(ExperimentRunnerError, match="RUN_GROUP_EXISTS"):
        runner.run([changed_config], tmp_path / "experiments", "group-v1")


def test_continue_on_error_and_failure_manifests(tmp_path):
    first = load(tmp_path, config_value("exp-fail"), "fail.json")
    second = load(tmp_path, config_value("exp-ok"), "ok.json")

    def failing(configuration):
        raise RuntimeError("provider unavailable")

    output, _ = ExperimentRunner({"fake": failing}).run(
        [first, second], tmp_path / "experiments", "stop-group", continue_on_error=False
    )
    assert json.loads((output / "runs" / "exp-fail" / "manifest.json").read_text())["status"] == "FAILED"
    assert json.loads((output / "runs" / "exp-ok" / "manifest.json").read_text())["status"] == "SKIPPED"

    calls = []

    def selective(configuration):
        calls.append(configuration.experiment_id)
        if configuration.experiment_id == "exp-fail":
            raise RuntimeError("provider unavailable")
        return result(configuration)

    output, _ = ExperimentRunner({"fake": selective}).run(
        [first, second], tmp_path / "experiments", "continue-group", continue_on_error=True
    )
    assert calls == ["exp-fail", "exp-ok"]
    assert json.loads((output / "manifest.json").read_text())["failed_runs"] == 1
    assert json.loads((output / "runs" / "exp-ok" / "manifest.json").read_text())["status"] == "COMPLETED"


def test_oracle_run_is_marked_separately(tmp_path):
    value = config_value(
        retrieval={
            "mode": "gold", "sparse_top_k": 50, "dense_top_k": 50,
            "fusion_top_k": 30, "rrf_k": 60, "bm25_index_version": "n/a",
            "dense_index_version": "n/a",
        },
        oracle_evidence=True,
    )
    configuration = load(tmp_path, value)
    output, _ = ExperimentRunner({"fake": result}).run(
        [configuration], tmp_path / "experiments", "oracle-group"
    )
    run = json.loads((output / "runs" / "exp-test" / "manifest.json").read_text())
    aggregate = json.loads((output / "aggregate_metrics.json").read_text())
    assert run["oracle_evidence"] is True
    assert aggregate["runs"]["exp-test"]["oracle_evidence"] is True


def test_qualitative_sample_is_deterministic(tmp_path):
    first = load(tmp_path, config_value("exp-a"), "a.json")
    second = load(tmp_path, config_value("exp-b"), "b.json")
    runner = ExperimentRunner({"fake": result})
    out1, _ = runner.run([first, second], tmp_path / "one", "group", qualitative_seed=7)
    out2, _ = runner.run([first, second], tmp_path / "two", "group", qualitative_seed=7)
    assert (out1 / "qualitative_examples.jsonl").read_bytes() == (out2 / "qualitative_examples.jsonl").read_bytes()
