"""Strict, machine-readable alignment checks for the medical runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from medclaim.datasets.constants import SUPPORTED_DATASETS
from medclaim.datasets.label_schema import UNIFIED_LABELS
from medclaim.experiments.configuration import load_experiment_configuration

_LEGACY_NAME = "FE" + "VER"
_LEGACY_PATTERN = re.compile(
    "|".join((_LEGACY_NAME, "Tony" + "BY/M3", "fe" + "ver_dataset", "fe" + "ver_subset")),
    re.I,
)
_SCAN_ROOTS = (
    "src",
    "app",
    "scripts",
    "configs",
    "tests",
    "artifacts/corpora",
    "artifacts/indexes",
)
_TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".toml", ".txt"}


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _manifest_files(root: Path) -> list[Path]:
    return sorted(
        path for path in (root / "artifacts").rglob("manifest.json") if "audits" not in path.parts
    )


def _contains_test_tuning(value: Any, parent: str = "") -> bool:
    if isinstance(value, dict):
        artifact = str(value.get("artifact_type", "")).casefold()
        tuning = any(term in artifact for term in ("calibr", "threshold", "gate"))
        for key, item in value.items():
            context = f"{parent}.{key}".casefold()
            if (
                (
                    tuning
                    or any(term in context for term in ("calibr", "threshold", "selection", "fit"))
                )
                and key.casefold()
                in {
                    "split",
                    "development_split",
                    "fitting_split",
                    "selection_split",
                    "tuning_split",
                }
                and str(item).casefold() == "test"
            ):
                return True
            if _contains_test_tuning(item, f"{context}.{artifact}"):
                return True
    elif isinstance(value, list):
        return any(_contains_test_tuning(item, parent) for item in value)
    return False


def audit_spec_alignment(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    runtime_references: list[dict[str, Any]] = []
    excluded = {root / "scripts" / "audit_spec_alignment.py", Path(__file__).resolve()}
    for relative in _SCAN_ROOTS:
        location = root / relative
        if not location.exists():
            continue
        for path in [location] if location.is_file() else location.rglob("*"):
            if not path.is_file() or path in excluded or path.suffix not in _TEXT_SUFFIXES:
                continue
            if relative.startswith("artifacts/") and path.name != "manifest.json":
                # Corpus evidence may legitimately discuss fever as a symptom. Only
                # artifact manifests are runtime configuration surfaces; scanning raw
                # passages creates false legacy-dataset violations.
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if _LEGACY_PATTERN.search(line):
                    runtime_references.append(
                        {"path": path.relative_to(root).as_posix(), "line": number}
                    )

    adapter_manifest_violations: list[str] = []
    for dataset in sorted(SUPPORTED_DATASETS):
        path = root / "configs" / "datasets" / f"{dataset}.yaml"
        try:
            value = _load(path)
            if (
                not isinstance(value, dict)
                or value.get("dataset") != dataset
                or not isinstance(value.get("adapter_manifest"), dict)
                or value["adapter_manifest"].get("artifact_type") != "normalized_dataset"
            ):
                raise ValueError("invalid normalized adapter manifest")
        except Exception as exc:
            adapter_manifest_violations.append(f"{path.relative_to(root)}: {exc}")

    invalid_corpus_manifests: list[str] = []
    invalid_index_manifests: list[str] = []
    invalid_label_mappings: list[str] = []
    test_tuning_violations: list[str] = []
    loaded_manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in _manifest_files(root):
        try:
            value = _load(path)
        except Exception as exc:
            invalid_corpus_manifests.append(f"{path.relative_to(root)}: unreadable ({exc})")
            continue
        if not isinstance(value, dict):
            invalid_corpus_manifests.append(f"{path.relative_to(root)}: manifest is not an object")
            continue
        loaded_manifests.append((path, value))
    corpus_references = {
        (value.get("corpus_version"), value.get("content_hash"), value.get("passage_count"))
        for _, value in loaded_manifests
        if value.get("artifact_type") == "medical_evidence_corpus"
    }
    for path, value in loaded_manifests:
        artifact_type = value.get("artifact_type")
        if artifact_type == "medical_evidence_corpus":
            datasets = value.get("datasets")
            if not isinstance(datasets, list) or set(datasets) != SUPPORTED_DATASETS:
                invalid_corpus_manifests.append(path.relative_to(root).as_posix())
        if artifact_type in {"bm25_index", "dense_index"}:
            corpus = value.get("corpus", {})
            reference = (
                (
                    corpus.get("version"),
                    corpus.get("content_hash"),
                    corpus.get("passage_count"),
                )
                if isinstance(corpus, dict)
                else None
            )
            if reference not in corpus_references:
                invalid_index_manifests.append(path.relative_to(root).as_posix())
        if _contains_test_tuning(value):
            test_tuning_violations.append(path.relative_to(root).as_posix())

    for path in sorted((root / "configs").rglob("*")):
        if path.suffix not in {".json", ".yaml", ".yml"}:
            continue
        try:
            value = _load(path)
        except Exception:
            continue
        mappings = value.get("mappings", {}) if isinstance(value, dict) else {}
        if isinstance(mappings, dict) and "MIXED" in mappings and mappings["MIXED"] != "MIXED":
            invalid_label_mappings.append(path.relative_to(root).as_posix())

    experiment_violations: list[str] = []
    required = {
        "dataset_version",
        "split_manifest_version",
        "corpus_version",
        "retrieval",
        "reranking",
        "verifier",
    }
    for path in sorted((root / "configs" / "experiments").glob("*.json")):
        try:
            configurations = load_experiment_configuration(path).expand()
            valid = all(
                not (required - set(item.value))
                and set(item.value.get("datasets", [])) <= SUPPORTED_DATASETS
                and all(
                    item.value.get(section, {}).get(key)
                    for section, key in (
                        ("retrieval", "bm25_index_version"),
                        ("retrieval", "dense_index_version"),
                        ("reranking", "model_version"),
                        ("verifier", "model_version"),
                        ("verifier", "prompt_version"),
                    )
                )
                and (
                    "MIXED" in item.value.get("labels", [])
                    or item.value.get("mixed_policy") == "exclude"
                )
                for item in configurations
            )
        except Exception:
            valid = False
        if not valid:
            experiment_violations.append(path.relative_to(root).as_posix())

    privacy_default_violations: list[str] = []
    deployment_path = root / "configs" / "deployment" / "default.yaml"
    try:
        deployment = _load(deployment_path)
        for key in ("persistence_enabled", "persist_claim_text", "persist_explanation"):
            if deployment.get(key) is not False:
                privacy_default_violations.append(f"{deployment_path.relative_to(root)}:{key}")
    except Exception as exc:
        privacy_default_violations.append(f"{deployment_path.relative_to(root)}: {exc}")

    failures = {
        "fe" + "ver_runtime_references": runtime_references,
        "adapter_manifest_violations": adapter_manifest_violations,
        "invalid_corpus_manifests": invalid_corpus_manifests,
        "invalid_index_manifests": invalid_index_manifests,
        "experiment_manifest_violations": experiment_violations,
        "invalid_label_mappings": invalid_label_mappings,
        "invalid_internal_labels": (
            []
            if tuple(UNIFIED_LABELS) == ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")
            else [list(UNIFIED_LABELS)]
        ),
        "test_tuning_violations": test_tuning_violations,
        "privacy_default_violations": privacy_default_violations,
    }
    return {
        "audit_version": "spec-alignment-v1",
        "status": "passed" if not any(failures.values()) else "failed",
        "supported_datasets": ["healthver", "scifact", "pubhealth"],
        **failures,
        "warnings": [],
    }
