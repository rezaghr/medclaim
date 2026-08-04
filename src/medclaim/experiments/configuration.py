"""Strict JSON experiment configurations and deterministic hashes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medclaim.datasets.constants import SUPPORTED_DATASETS

TOP_LEVEL = {
    "experiment_id", "dataset_version", "split_manifest_version",
    "corpus_version", "corpus_content_hash", "datasets", "split", "labels",
    "retrieval", "reranking", "gate", "decomposition", "verifier",
    "calibration", "seed", "oracle_evidence", "executor", "input_artifact",
    "variants", "mixed_policy",
}
NESTED = {
    "retrieval": {
        "mode", "sparse_top_k", "dense_top_k", "fusion_top_k", "rrf_k",
        "bm25_index_version", "dense_index_version",
    },
    "reranking": {"enabled", "candidate_count", "final_evidence_k", "model_version"},
    "gate": {"enabled", "version"},
    "decomposition": {"mode"},
    "verifier": {"implementation", "model_version", "prompt_version"},
    "calibration": {"version", "method"},
}
REQUIRED = TOP_LEVEL - {"input_artifact", "variants", "mixed_policy"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ExperimentConfigurationError(Exception):
    """Raised for unknown, missing, or incompatible experiment settings."""


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


@dataclass(frozen=True)
class ExperimentConfiguration:
    value: dict[str, Any]
    source: str

    @property
    def experiment_id(self) -> str:
        return self.value["experiment_id"]

    @property
    def configuration_hash(self) -> str:
        return _canonical_hash(self.value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)

    def expand(self) -> list["ExperimentConfiguration"]:
        variants = self.value.get("variants", [])
        if not variants:
            value = self.to_dict()
            value.pop("variants", None)
            return [ExperimentConfiguration(_validate(value), self.source)]
        expanded = []
        base = self.to_dict()
        base.pop("variants", None)
        for variant in variants:
            if not isinstance(variant, dict) or set(variant) != {"suffix", "overrides"}:
                raise ExperimentConfigurationError(
                    "EXPERIMENT_CONFIG_INVALID: Each variant requires suffix and overrides."
                )
            suffix = variant["suffix"]
            overrides = variant["overrides"]
            if not isinstance(suffix, str) or not ID_PATTERN.fullmatch(suffix) or not isinstance(overrides, dict):
                raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: Variant is invalid.")
            value = _deep_merge(base, overrides)
            value["experiment_id"] = f"{base['experiment_id']}-{suffix}"
            expanded.append(ExperimentConfiguration(_validate(value), self.source))
        return expanded


def _validate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: Configuration must be an object.")
    unknown = sorted(set(value) - TOP_LEVEL)
    missing = sorted(REQUIRED - set(value))
    if unknown:
        raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_UNKNOWN_FIELD: Unknown field {unknown[0]!r}.")
    if missing:
        raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_INVALID: Missing field {missing[0]!r}.")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or ID_PATTERN.fullmatch(experiment_id) is None:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: experiment_id is invalid.")
    for field in ("dataset_version", "split_manifest_version", "corpus_version", "corpus_content_hash", "executor"):
        if not isinstance(value[field], str) or not value[field]:
            raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_INVALID: {field} must be non-empty.")
    if value["split"] not in {"train", "dev", "test"}:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: split must be train, dev, or test.")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: seed must be an integer.")
    if not isinstance(value["oracle_evidence"], bool):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: oracle_evidence must be boolean.")
    if not isinstance(value["datasets"], list) or not value["datasets"] or not set(value["datasets"]) <= SUPPORTED_DATASETS:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: datasets are invalid.")
    allowed_labels = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED"}
    if not isinstance(value["labels"], list) or not value["labels"] or not set(value["labels"]) <= allowed_labels:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: labels are invalid.")
    mixed_policy = value.get("mixed_policy", "include")
    if mixed_policy not in {"include", "exclude"} or (
        "MIXED" not in value["labels"] and mixed_policy != "exclude"
    ):
        raise ExperimentConfigurationError(
            "EXPERIMENT_CONFIG_INVALID: Three-label runs must explicitly exclude MIXED."
        )
    for section, allowed in NESTED.items():
        section_value = value.get(section)
        if not isinstance(section_value, dict) or set(section_value) != allowed:
            extra = sorted(set(section_value or {}) - allowed)
            reason = f"unknown field {extra[0]!r}" if extra else "missing or invalid fields"
            raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_INVALID: {section} has {reason}.")
    if value["retrieval"]["mode"] not in {"none", "bm25", "dense", "hybrid", "gold"}:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: retrieval mode is invalid.")
    if value["decomposition"]["mode"] not in {"off", "auto", "always"}:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: decomposition mode is invalid.")
    if value["verifier"]["implementation"] not in {"llm", "classifier"}:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: verifier is invalid.")
    if value["calibration"]["method"] not in {"none", "logistic", "isotonic"}:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: calibration method is invalid.")
    for section in ("retrieval", "reranking"):
        for key, item in value[section].items():
            if key.endswith("top_k") or key in {"rrf_k", "candidate_count", "final_evidence_k"}:
                if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                    raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_INVALID: {section}.{key} must be positive.")
    if value["reranking"]["final_evidence_k"] > value["reranking"]["candidate_count"]:
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: final_evidence_k exceeds candidates.")
    if not isinstance(value["reranking"]["enabled"], bool) or not isinstance(
        value["gate"]["enabled"], bool
    ):
        raise ExperimentConfigurationError(
            "EXPERIMENT_CONFIG_INVALID: reranking.enabled and gate.enabled must be boolean."
        )
    if value["oracle_evidence"] != (value["retrieval"]["mode"] == "gold"):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: Oracle flag and gold retrieval mode disagree.")
    if value.get("input_artifact") is not None and not isinstance(value["input_artifact"], str):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: input_artifact must be a string.")
    return copy.deepcopy(value)


def load_experiment_configuration(path: Path) -> ExperimentConfiguration:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"EXPERIMENT_CONFIG_INVALID: Could not read {path}: {exc}.") from exc
    return ExperimentConfiguration(_validate(value), path.as_posix())


def load_experiment_configurations(paths: list[Path]) -> list[ExperimentConfiguration]:
    expanded = [item for path in sorted(paths) for item in load_experiment_configuration(path).expand()]
    ids = [item.experiment_id for item in expanded]
    if len(ids) != len(set(ids)):
        raise ExperimentConfigurationError("EXPERIMENT_CONFIG_INVALID: Expanded experiment IDs are duplicated.")
    return expanded
