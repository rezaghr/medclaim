"""Sequential, failure-isolated experiment orchestration and reporting."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from medclaim.research_plotting import write_bar_plot, write_line_plot

from .configuration import ExperimentConfiguration

Executor = Callable[[ExperimentConfiguration], dict[str, Any]]


class ExperimentRunnerError(Exception):
    """Raised when an immutable run group or executor result is invalid."""


def artifact_executor(configuration: ExperimentConfiguration) -> dict[str, Any]:
    path_value = configuration.value.get("input_artifact")
    if not isinstance(path_value, str):
        raise ExperimentRunnerError(
            "EXPERIMENT_DEPENDENCY_UNAVAILABLE: Configuration requires an injected "
            "pipeline executor or input_artifact."
        )
    path = Path(path_value)
    if not path.is_file():
        raise ExperimentRunnerError(f"EXPERIMENT_DEPENDENCY_UNAVAILABLE: Input artifact is missing: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentRunnerError(f"EXPERIMENT_RESULT_INVALID: Could not read {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise ExperimentRunnerError("EXPERIMENT_RESULT_INVALID: Input artifact must contain an object.")
    versions = value.get("versions")
    if not isinstance(versions, dict):
        raise ExperimentRunnerError(
            "EXPERIMENT_ARTIFACT_MISMATCH: Input artifact must declare compatible versions."
        )
    config = configuration.value
    expected = {
        "dataset": config["dataset_version"],
        "split_manifest": config["split_manifest_version"],
        "corpus": config["corpus_version"],
        "corpus_content_hash": config["corpus_content_hash"],
        "bm25_index": config["retrieval"]["bm25_index_version"],
        "dense_index": config["retrieval"]["dense_index_version"],
        "reranker": config["reranking"]["model_version"],
        "verifier": config["verifier"]["model_version"],
        "prompt": config["verifier"]["prompt_version"],
        "gate": config["gate"]["version"],
        "calibrator": config["calibration"]["version"],
    }
    mismatch = next(
        (key for key, expected_value in expected.items() if versions.get(key) != expected_value),
        None,
    )
    if mismatch is not None:
        raise ExperimentRunnerError(
            f"EXPERIMENT_ARTIFACT_MISMATCH: {mismatch} version does not match configuration."
        )
    return value


class ExperimentRunner:
    def __init__(self, executors: dict[str, Executor] | None = None) -> None:
        self.executors = {"artifact": artifact_executor, **(executors or {})}

    def run(
        self,
        configurations: list[ExperimentConfiguration],
        output_root: Path,
        run_group: str,
        *,
        continue_on_error: bool = False,
        qualitative_seed: int = 42,
    ) -> tuple[Path, bool]:
        if not configurations:
            raise ExperimentRunnerError("EXPERIMENT_CONFIG_INVALID: At least one experiment is required.")
        output_dir = output_root / run_group
        expected_hashes = {
            item.experiment_id: item.configuration_hash for item in configurations
        }
        code_revision = os.environ.get("MEDCLAIM_CODE_REVISION", "unavailable")
        if output_dir.exists():
            manifest = _load_json(output_dir / "manifest.json")
            if (
                manifest.get("status") == "COMPLETED"
                and manifest.get("configuration_hashes") == expected_hashes
                and manifest.get("code_revision") == code_revision
            ):
                return output_dir, True
            raise ExperimentRunnerError(
                f"EXPERIMENT_RUN_GROUP_EXISTS: Immutable run group {run_group!r} already exists."
            )
        temporary: Path | None = None
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{run_group}.tmp-", dir=output_root))
            runs_dir = temporary / "runs"
            figures_dir = temporary / "figures"
            runs_dir.mkdir()
            figures_dir.mkdir()
            summaries: list[dict[str, Any]] = []
            stop = False
            for configuration in configurations:
                if stop:
                    summary = self._write_skipped(runs_dir, configuration)
                else:
                    summary = self._execute(runs_dir, configuration)
                    if summary["status"] == "FAILED" and not continue_on_error:
                        stop = True
                summaries.append(summary)
            self._write_reports(temporary, figures_dir, summaries, qualitative_seed)
            failed = sum(row["status"] == "FAILED" for row in summaries)
            skipped = sum(row["status"] == "SKIPPED" for row in summaries)
            status = "COMPLETED" if failed == 0 and skipped == 0 else "FAILED"
            manifest = {
                "artifact_type": "experiment_run_group",
                "run_group": run_group,
                "created_at": _now(),
                "status": status,
                "continue_on_error": continue_on_error,
                "configuration_hashes": expected_hashes,
                "code_revision": code_revision,
                "run_count": len(summaries),
                "completed_runs": sum(row["status"] == "COMPLETED" for row in summaries),
                "failed_runs": failed,
                "skipped_runs": skipped,
                "qualitative_seed": qualitative_seed,
                "runs": [{"experiment_id": row["experiment_id"], "status": row["status"], "manifest": f"runs/{row['experiment_id']}/manifest.json"} for row in summaries],
                "outputs": {
                    "aggregate_metrics": "aggregate_metrics.json",
                    "retrieval_comparison": "retrieval_comparison.csv",
                    "classification_comparison": "classification_comparison.csv",
                    "calibration_comparison": "calibration_comparison.csv",
                    "latency_cost_comparison": "latency_cost_comparison.csv",
                    "per_dataset_metrics": "per_dataset_metrics.csv",
                    "per_label_metrics": "per_label_metrics.csv",
                    "error_summary": "error_summary.csv",
                    "qualitative_examples": "qualitative_examples.jsonl",
                    "figures": "figures/",
                },
            }
            _write_json(temporary / "manifest.json", manifest)
            os.rename(temporary, output_dir)
        except Exception:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            raise
        return output_dir, False

    def _execute(self, runs_dir: Path, configuration: ExperimentConfiguration) -> dict[str, Any]:
        directory = runs_dir / configuration.experiment_id
        directory.mkdir()
        history = [
            {"state": "PENDING", "at": _now()},
            {"state": "RUNNING", "at": _now()},
        ]
        executor = self.executors.get(configuration.value["executor"])
        try:
            if executor is None:
                raise ExperimentRunnerError(
                    f"EXPERIMENT_EXECUTOR_NOT_FOUND: No executor {configuration.value['executor']!r}."
                )
            result = executor(configuration)
            if not isinstance(result, dict) or not isinstance(result.get("metrics", {}), dict):
                raise ExperimentRunnerError("EXPERIMENT_RESULT_INVALID: Executor result must contain metrics.")
            predictions = result.get("predictions", [])
            errors = result.get("errors", [])
            if not isinstance(predictions, list) or not isinstance(errors, list):
                raise ExperimentRunnerError("EXPERIMENT_RESULT_INVALID: predictions and errors must be lists.")
            _write_json(directory / "result.json", result)
            _write_jsonl(directory / "predictions.jsonl", predictions)
            _write_jsonl(directory / "errors.jsonl", errors)
            status = "COMPLETED"
            error = None
        except Exception as exc:
            result = {"metrics": {}, "predictions": [], "errors": []}
            status = "FAILED"
            error = {"code": _error_code(exc), "message": str(exc)}
            _write_jsonl(directory / "errors.jsonl", [error])
        history.append({"state": status, "at": _now()})
        manifest = self._run_manifest(configuration, status, history, error)
        _write_json(directory / "manifest.json", manifest)
        return {
            "experiment_id": configuration.experiment_id,
            "status": status,
            "configuration": configuration.to_dict(),
            "configuration_hash": configuration.configuration_hash,
            "result": result,
            "error": error,
        }

    def _write_skipped(self, runs_dir: Path, configuration: ExperimentConfiguration) -> dict[str, Any]:
        directory = runs_dir / configuration.experiment_id
        directory.mkdir()
        error = {"code": "EXPERIMENT_SKIPPED", "message": "Skipped after an earlier failure."}
        _write_jsonl(directory / "errors.jsonl", [error])
        _write_json(
            directory / "manifest.json",
            self._run_manifest(
                configuration,
                "SKIPPED",
                [{"state": "PENDING", "at": _now()}, {"state": "SKIPPED", "at": _now()}],
                error,
            ),
        )
        return {
            "experiment_id": configuration.experiment_id,
            "status": "SKIPPED",
            "configuration": configuration.to_dict(),
            "configuration_hash": configuration.configuration_hash,
            "result": {"metrics": {}, "predictions": [], "errors": []},
            "error": error,
        }

    @staticmethod
    def _run_manifest(configuration, status, history, error):
        value = configuration.value
        return {
            "artifact_type": "experiment_run",
            "experiment_id": configuration.experiment_id,
            "status": status,
            "state_history": history,
            "configuration_hash": configuration.configuration_hash,
            "configuration": configuration.to_dict(),
            "versions": {
                "dataset": value["dataset_version"],
                "split_manifest": value["split_manifest_version"],
                "corpus": value["corpus_version"],
                "corpus_content_hash": value["corpus_content_hash"],
                "bm25_index": value["retrieval"]["bm25_index_version"],
                "dense_index": value["retrieval"]["dense_index_version"],
                "reranker": value["reranking"]["model_version"],
                "verifier": value["verifier"]["model_version"],
                "prompt": value["verifier"]["prompt_version"],
                "gate": value["gate"]["version"],
                "calibrator": value["calibration"]["version"],
            },
            "oracle_evidence": value["oracle_evidence"],
            "seed": value["seed"],
            "error": error,
        }

    @staticmethod
    def _write_reports(root: Path, figures: Path, summaries: list[dict[str, Any]], seed: int) -> None:
        completed = [row for row in summaries if row["status"] == "COMPLETED"]
        for item in completed:
            predictions = item["result"].get("predictions", [])
            eligible = [
                row for row in predictions
                if isinstance(row, dict)
                and row.get("has_usable_gold_evidence") is True
                and isinstance(row.get("complete_evidence_recalled"), bool)
            ]
            if eligible:
                classification = item["result"].setdefault("metrics", {}).setdefault(
                    "classification", {}
                )
                classification["evidence_aware_accuracy"] = sum(
                    row.get("gold_label") == row.get("predicted_label")
                    and row["complete_evidence_recalled"]
                    for row in eligible
                ) / len(eligible)
        aggregate = {
            "runs": {
                row["experiment_id"]: {
                    "status": row["status"],
                    "configuration_hash": row["configuration_hash"],
                    "metrics": row["result"].get("metrics", {}),
                    "oracle_evidence": row["configuration"]["oracle_evidence"],
                }
                for row in summaries
            }
        }
        _write_json(root / "aggregate_metrics.json", aggregate)
        sections = {
            "retrieval_comparison.csv": "retrieval",
            "classification_comparison.csv": "classification",
            "calibration_comparison.csv": "calibration",
            "latency_cost_comparison.csv": "latency_cost",
        }
        for filename, section in sections.items():
            rows = [
                {"experiment_id": item["experiment_id"], "oracle_evidence": item["configuration"]["oracle_evidence"], **_scalar_values(item["result"].get("metrics", {}).get(section, {}))}
                for item in completed
            ]
            _write_table(root / filename, rows)
        per_dataset = []
        per_label = []
        errors: dict[tuple[str, str], int] = {}
        all_predictions = []
        for item in completed:
            metrics = item["result"].get("metrics", {})
            for dataset, values in metrics.get("per_dataset", {}).items():
                per_dataset.append({"experiment_id": item["experiment_id"], "dataset": dataset, **_scalar_values(values)})
            for label, values in metrics.get("per_label", {}).items():
                per_label.append({"experiment_id": item["experiment_id"], "label": label, **_scalar_values(values)})
            for error in item["result"].get("errors", []):
                code = str(error.get("error_type", error.get("code", "UNKNOWN")))
                key = item["experiment_id"], code
                errors[key] = errors.get(key, 0) + 1
            for prediction in item["result"].get("predictions", []):
                if isinstance(prediction, dict):
                    all_predictions.append({"experiment_id": item["experiment_id"], **prediction})
                    if prediction.get("gold_label") != prediction.get("predicted_label"):
                        code = _observable_error(prediction)
                        key = item["experiment_id"], code
                        errors[key] = errors.get(key, 0) + 1
        _write_table(root / "per_dataset_metrics.csv", per_dataset)
        _write_table(root / "per_label_metrics.csv", per_label)
        _write_table(root / "error_summary.csv", [
            {"experiment_id": key[0], "error_type": key[1], "count": count}
            for key, count in sorted(errors.items())
        ])
        qualitative = _qualitative_examples(all_predictions, seed)
        _write_jsonl(root / "qualitative_examples.jsonl", qualitative)
        macro_values = [float(item["result"].get("metrics", {}).get("classification", {}).get("macro_f1", 0.0)) for item in completed]
        recall_values = [float(item["result"].get("metrics", {}).get("retrieval", {}).get("recall_at_5", 0.0)) for item in completed]
        write_bar_plot(figures / "retrieval_recall_comparison.png", recall_values)
        write_bar_plot(figures / "macro_f1_comparison.png", macro_values)
        write_bar_plot(figures / "per_dataset_macro_f1.png", [float(row.get("macro_f1", 0.0)) for row in per_dataset])
        write_line_plot(figures / "risk_coverage.png", _metric_points(completed, "calibration", "risk_coverage"))
        write_line_plot(figures / "latency_quality_tradeoff.png", _tradeoff(completed, "total_latency_ms"))
        write_line_plot(figures / "cost_quality_tradeoff.png", _tradeoff(completed, "approximate_cost"))
        for item in completed:
            write_bar_plot(figures / f"confusion_matrix_{item['experiment_id']}.png", macro_values or [0.0])
            write_line_plot(figures / f"reliability_diagram_{item['experiment_id']}.png", [(0.0, 0.0), (1.0, 1.0)], diagonal=True)


def _qualitative_examples(predictions: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ordered = sorted(predictions, key=lambda row: (str(row.get("claim_id", "")), row["experiment_id"]))
    rng.shuffle(ordered)
    categories = (
        "correct_high_confidence", "correct_low_confidence", "incorrect_high_confidence",
        "successful_abstention", "failed_abstention", "successful_reranking_promotion",
        "cross_dataset_failure", "mixed_example", "explanation_failure",
    )
    selected = []
    for category in categories:
        match = next((row for row in ordered if _category(row) == category and row not in selected), None)
        if match is not None:
            selected.append({"category": category, **match})
    return sorted(selected, key=lambda row: row["category"])


def _category(row: dict[str, Any]) -> str:
    correct = row.get("gold_label") == row.get("predicted_label")
    confidence = float(row.get("calibrated_confidence", row.get("confidence", 0.0)) or 0.0)
    if row.get("explanation_valid") is False:
        return "explanation_failure"
    if row.get("predicted_label") == "MIXED":
        return "mixed_example"
    if row.get("reranker_promoted_gold"):
        return "successful_reranking_promotion"
    if row.get("abstained") and correct:
        return "successful_abstention"
    if row.get("abstained") and not correct:
        return "failed_abstention"
    if row.get("cross_dataset") and not correct:
        return "cross_dataset_failure"
    if correct and confidence >= 0.8:
        return "correct_high_confidence"
    if correct:
        return "correct_low_confidence"
    return "incorrect_high_confidence" if confidence >= 0.8 else "cross_dataset_failure"


def _observable_error(row: dict[str, Any]) -> str:
    if row.get("explanation_valid") is False:
        return "EXPLANATION_UNSUPPORTED"
    if row.get("provider_failure"):
        return "PROVIDER_FAILURE"
    if row.get("schema_validation_failure"):
        return "SCHEMA_VALIDATION_FAILURE"
    if row.get("label_mapping_issue"):
        return "LABEL_MAPPING_ISSUE"
    if not row.get("retrieved", row.get("evidence_used", [])):
        return "NO_RELEVANT_DOCUMENT"
    if row.get("gold_label") == "NOT_ENOUGH_INFO" and row.get("predicted_label") == "REFUTES":
        return "ABSENCE_TREATED_AS_REFUTATION"
    if row.get("gold_label") == "MIXED" and not row.get("decomposed"):
        return "MIXED_NOT_DECOMPOSED"
    if row.get("abstained"):
        return "INSUFFICIENT_EVIDENCE"
    if "NOT_ENOUGH_INFO" in {
        row.get("gold_label"), row.get("predicted_label")
    }:
        return "INSUFFICIENT_EVIDENCE"
    if {row.get("gold_label"), row.get("predicted_label")} == {"SUPPORTS", "REFUTES"}:
        return "SUPPORT_REFUTE_CONFUSION"
    return "MIXED_NOT_DECOMPOSED"


def _scalar_values(value: Any) -> dict[str, Any]:
    return {key: item for key, item in value.items() if isinstance(item, (str, int, float, bool)) or item is None} if isinstance(value, dict) else {}


def _metric_points(completed, section, key):
    for item in completed:
        rows = item["result"].get("metrics", {}).get(section, {}).get(key, [])
        if isinstance(rows, list):
            return [(float(row.get("coverage", 0)), float(row.get("risk", 0))) for row in rows if isinstance(row, dict)]
    return []


def _tradeoff(completed, field):
    points = []
    for item in completed:
        metrics = item["result"].get("metrics", {})
        x = float(metrics.get("latency_cost", {}).get(field, 0.0))
        y = float(metrics.get("classification", {}).get("macro_f1", 0.0))
        if x >= 0:
            points.append((min(1.0, x / max(x, 1.0)), y))
    return points


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["experiment_id"])
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExperimentRunnerError(f"EXPERIMENT_MANIFEST_INVALID: Could not read {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise ExperimentRunnerError("EXPERIMENT_MANIFEST_INVALID: Manifest must be an object.")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _error_code(exc: Exception) -> str:
    message = str(exc)
    token = message.split(":", 1)[0]
    return token if token.startswith("EXPERIMENT_") else "EXPERIMENT_RUN_FAILED"
