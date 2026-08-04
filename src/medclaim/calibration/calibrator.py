"""Versioned confidence calibrator fitting, loading, and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medclaim.research_plotting import write_line_plot

from .features import FEATURE_NAMES, CalibrationFeatureError, extract_confidence_features
from .metrics import calibration_metrics

CALIBRATOR_VERSION = "1.0.0"
METHODS = {"none", "logistic", "isotonic"}
CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}")


class CalibrationError(Exception):
    """Raised for controlled calibration data, model, or artifact errors."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"CALIBRATION_ARTIFACT_INVALID: Could not read {path}: {exc}.") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                rows.append(value)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise CalibrationError(f"CALIBRATION_DATA_INVALID: Could not read {path}: {exc}.") from exc
    return rows


def _assignments(value: Any) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise CalibrationError("CALIBRATION_SPLIT_INVALID: Split manifest must be an object.")
    output: dict[str, str] = {}
    for row in value.get("assignments", []):
        if isinstance(row, dict) and isinstance(row.get("claim_id"), str):
            split = row.get("project_split", row.get("split"))
            if isinstance(split, str) and row.get("calibration_eligible", True) is not False:
                output[row["claim_id"]] = split
    if isinstance(value.get("claim_splits"), dict):
        output.update({str(key): str(split) for key, split in value["claim_splits"].items()})
    if isinstance(value.get("splits"), dict):
        for split, ids in value["splits"].items():
            if isinstance(ids, list):
                output.update({str(claim_id): str(split) for claim_id in ids})
    if not output:
        raise CalibrationError("CALIBRATION_SPLIT_INVALID: No split assignments were found.")
    return output, str(value.get("version", value.get("split_version", "unknown")))


@dataclass
class ConfidenceCalibrator:
    method: str
    version: str
    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict(self, records: list[dict[str, Any]]) -> list[float]:
        features = []
        raw = []
        for record in records:
            vector, metadata = extract_confidence_features(record)
            features.append(vector)
            raw.append(metadata["raw_confidence"])
        if self.method == "none":
            probabilities = np.asarray(raw, dtype=float)
        elif self.method == "logistic":
            probabilities = self.model.predict_proba(np.asarray(features, dtype=float))[:, 1]
        elif self.method == "isotonic":
            probabilities = self.model.predict(np.asarray(raw, dtype=float))
        else:
            raise CalibrationError(f"CALIBRATION_METHOD_INVALID: Unknown method {self.method!r}.")
        return [max(0.0, min(1.0, float(value))) for value in probabilities]

    def result(self, record: dict[str, Any]) -> dict[str, Any]:
        vector, metadata = extract_confidence_features(record)
        del vector
        return {
            "raw_confidence": metadata["raw_confidence"],
            "calibrated_confidence": self.predict([record])[0],
            "confidence_method": self.method,
            "calibrator_version": self.version,
            "confidence_warning": (
                "Estimated from validation performance; not clinical certainty."
                if self.method != "none"
                else "Uncalibrated model confidence estimate."
            ),
        }


def raw_confidence_result(record: dict[str, Any]) -> dict[str, Any]:
    _, metadata = extract_confidence_features(record)
    return {
        "raw_confidence": metadata["raw_confidence"],
        "calibrated_confidence": None,
        "confidence_method": "raw",
        "calibrator_version": None,
        "confidence_warning": "Uncalibrated model confidence estimate.",
    }


def fit_confidence_calibrator(
    predictions_path: Path,
    split_manifest_path: Path,
    method: str,
    output_root: Path,
    version: str,
    bins: int = 10,
    seed: int = 42,
) -> Path:
    if method not in METHODS:
        raise CalibrationError(f"CALIBRATION_METHOD_INVALID: Unsupported method {method!r}.")
    output_dir = output_root / version
    if output_dir.exists():
        raise CalibrationError(f"CALIBRATION_OUTPUT_EXISTS: {output_dir} already exists.")
    assignments, split_version = _assignments(_load_json(split_manifest_path))
    records: list[dict[str, Any]] = []
    targets: list[int] = []
    exclusions: dict[str, int] = {}
    for record in _load_jsonl(predictions_path):
        claim_id = record.get("claim_id")
        assigned = assignments.get(claim_id)
        if assigned == "validation":
            assigned = "dev"
        if assigned != "dev":
            continue
        gold = record.get("gold_label", record.get("unified_label"))
        result = record.get("result") if isinstance(record.get("result"), dict) else record
        predicted = record.get("predicted_label", result.get("verdict"))
        if not isinstance(gold, str) or not isinstance(predicted, str):
            exclusions["MISSING_LABEL"] = exclusions.get("MISSING_LABEL", 0) + 1
            continue
        if record.get("provider_failure") or record.get("oracle_evidence"):
            reason = "PROVIDER_FAILURE" if record.get("provider_failure") else "ORACLE_EVIDENCE"
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        try:
            extract_confidence_features(record)
        except CalibrationFeatureError:
            exclusions["INVALID_FEATURES"] = exclusions.get("INVALID_FEATURES", 0) + 1
            continue
        records.append(record)
        targets.append(int(predicted == gold))
    if not records:
        raise CalibrationError("CALIBRATION_NO_ELIGIBLE_RECORDS: No development predictions are eligible.")
    features = np.asarray([extract_confidence_features(row)[0] for row in records], dtype=float)
    raw = np.asarray([extract_confidence_features(row)[1]["raw_confidence"] for row in records], dtype=float)
    target_array = np.asarray(targets, dtype=int)
    if method in {"logistic", "isotonic"} and len(set(targets)) < 2:
        raise CalibrationError("CALIBRATION_NO_ELIGIBLE_RECORDS: Fitting requires correct and incorrect examples.")
    if method == "logistic":
        model: Any = LogisticRegression(random_state=seed, max_iter=1000)
        model.fit(features, target_array)
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw, target_array)
    else:
        model = None
    calibrator = ConfidenceCalibrator(method, version, model)
    calibrated = calibrator.predict(records)
    raw_metrics = calibration_metrics(raw.tolist(), targets, bins)
    calibrated_metrics = calibration_metrics(calibrated, targets, bins)
    feature_schema = {
        "schema_version": "1.0.0",
        "dataset_agnostic": True,
        "target": "prediction_correct",
        "feature_names": list(FEATURE_NAMES if method != "isotonic" else ("raw_confidence",)),
        "prohibited_features": ["gold_label", "claim_id", "dataset_test_outcome", "gold_evidence_presence"],
    }
    metrics = {
        "method": method,
        "development_records": len(records),
        "excluded_records": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "before_calibration": raw_metrics,
        "after_calibration": calibrated_metrics,
    }
    temporary: Path | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root))
        with (temporary / "calibrator.pkl").open("wb") as handle:
            pickle.dump(
                {"builder_version": CALIBRATOR_VERSION, "method": method, "version": version, "model": model},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _write_json(temporary / "feature_schema.json", feature_schema)
        _write_json(temporary / "metrics.json", metrics)
        manifest = {
            "artifact_type": "confidence_calibrator",
            "version": version,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "builder_version": CALIBRATOR_VERSION,
            "method": method,
            "development_split": "dev",
            "split_manifest_version": split_version,
            "seed": seed,
            "bins": bins,
            "eligible_records": len(records),
            "files": {
                "calibrator": {"path": "calibrator.pkl", "sha256": _sha256(temporary / "calibrator.pkl")},
                "feature_schema": {"path": "feature_schema.json", "sha256": _sha256(temporary / "feature_schema.json")},
                "metrics": {"path": "metrics.json", "sha256": _sha256(temporary / "metrics.json")},
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, output_dir)
    except FileExistsError as exc:
        raise CalibrationError(f"CALIBRATION_OUTPUT_EXISTS: {output_dir} already exists.") from exc
    except Exception as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(f"CALIBRATION_OUTPUT_INVALID: Could not write artifacts: {exc}.") from exc
    return output_dir


def load_confidence_calibrator(directory: Path) -> ConfidenceCalibrator:
    manifest = _load_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != "confidence_calibrator":
        raise CalibrationError("CALIBRATION_ARTIFACT_INVALID: Manifest type is invalid.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CalibrationError("CALIBRATION_ARTIFACT_INVALID: Manifest files are missing.")
    for key in ("calibrator", "feature_schema", "metrics"):
        entry = files.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str) or CHECKSUM.fullmatch(entry["sha256"]) is None:
            raise CalibrationError("CALIBRATION_ARTIFACT_INVALID: File metadata is invalid.")
        path = directory / entry["path"]
        if path.parent != directory or _sha256(path) != entry["sha256"]:
            raise CalibrationError("CALIBRATION_CHECKSUM_MISMATCH: Calibrator artifact checksum mismatch.")
    schema = _load_json(directory / files["feature_schema"]["path"])
    expected = list(FEATURE_NAMES if manifest.get("method") != "isotonic" else ("raw_confidence",))
    if not isinstance(schema, dict) or schema.get("feature_names") != expected:
        raise CalibrationError("CALIBRATION_FEATURE_SCHEMA_MISMATCH: Feature schema is incompatible.")
    try:
        with (directory / files["calibrator"]["path"]).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise CalibrationError(f"CALIBRATION_ARTIFACT_INVALID: Could not load project calibrator: {exc}.") from exc
    if not isinstance(payload, dict) or payload.get("builder_version") != CALIBRATOR_VERSION or payload.get("method") != manifest.get("method") or payload.get("version") != manifest.get("version"):
        raise CalibrationError("CALIBRATION_ARTIFACT_INVALID: Serialized calibrator metadata mismatch.")
    return ConfidenceCalibrator(payload["method"], payload["version"], payload.get("model"))


def evaluate_confidence_calibrator(
    predictions_path: Path,
    calibrator_dir: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise CalibrationError(f"CALIBRATION_EVALUATION_OUTPUT_EXISTS: {output_dir} already exists.")
    calibrator = load_confidence_calibrator(calibrator_dir)
    records = []
    targets = []
    for record in _load_jsonl(predictions_path):
        result = record.get("result") if isinstance(record.get("result"), dict) else record
        gold = record.get("gold_label", record.get("unified_label"))
        predicted = record.get("predicted_label", result.get("verdict"))
        if isinstance(gold, str) and isinstance(predicted, str) and not record.get("provider_failure"):
            try:
                extract_confidence_features(record)
            except CalibrationFeatureError as exc:
                raise CalibrationError(str(exc)) from exc
            records.append(record)
            targets.append(int(gold == predicted))
    if not records:
        raise CalibrationError("CALIBRATION_NO_ELIGIBLE_RECORDS: No evaluable predictions.")
    raw = [extract_confidence_features(record)[1]["raw_confidence"] for record in records]
    calibrated = calibrator.predict(records)
    before = calibration_metrics(raw, targets, 10)
    after = calibration_metrics(calibrated, targets, 10)
    temporary: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
        enriched = []
        for record, probability in zip(records, calibrated, strict=True):
            copy = dict(record)
            copy["calibration"] = {
                **calibrator.result(record),
                "calibrated_confidence": probability,
            }
            enriched.append(copy)
        _write_jsonl(temporary / "predictions.jsonl", enriched)
        _write_json(temporary / "metrics.json", {"before_calibration": before, "after_calibration": after})
        _write_csv(temporary / "reliability_bins.csv", after["reliability_bins"])
        _write_csv(temporary / "accuracy_coverage.csv", after["accuracy_coverage"])
        _write_csv(temporary / "risk_coverage.csv", after["risk_coverage"])
        points = [
            (row["mean_confidence"], row["observed_accuracy"])
            for row in after["reliability_bins"] if row["sample_count"]
        ]
        write_line_plot(temporary / "reliability_diagram.png", points, diagonal=True)
        write_line_plot(
            temporary / "risk_coverage.png",
            [(row["coverage"], row["risk"]) for row in after["risk_coverage"]],
        )
        _write_json(
            temporary / "manifest.json",
            {
                "artifact_type": "calibration_evaluation",
                "version": output_dir.name,
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "calibrator_version": calibrator.version,
                "method": calibrator.method,
                "evaluated_records": len(records),
                "calibrator_unchanged": True,
                "outputs": {
                    "predictions": "predictions.jsonl",
                    "metrics": "metrics.json",
                    "reliability_bins": "reliability_bins.csv",
                    "reliability_diagram": "reliability_diagram.png",
                    "accuracy_coverage": "accuracy_coverage.csv",
                    "risk_coverage": "risk_coverage.csv",
                    "risk_coverage_figure": "risk_coverage.png",
                },
            },
        )
        os.rename(temporary, output_dir)
    except Exception as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(f"CALIBRATION_EVALUATION_INVALID: {exc}.") from exc
    return output_dir


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
