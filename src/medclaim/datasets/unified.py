"""Build an immutable unified dataset from normalized medical datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import DATASET_ORDER, DATASET_RANK
from .label_schema import (
    LABEL_SCHEMA_VERSION,
    UNIFIED_LABELS,
    build_label_schema,
    extract_source_mapping,
)

BUILDER_VERSION = "1.0.0"
SPLIT_ORDER = {"train": 0, "dev": 1, "validation": 1, "test": 2}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REQUIRED_INPUTS = (
    "claims.jsonl",
    "documents.jsonl",
    "label_mapping.json",
    "quality_report.json",
    "manifest.json",
)


class UnifiedDatasetError(Exception):
    """Raised when normalized datasets cannot be merged safely."""


def _canonical_hash(values: Iterable[Any]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


def normalized_dataset_content_hash(
    claims: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> str:
    """Hash normalized claim and document records in their file order."""
    return _source_content_hash(claims, documents)


def _source_content_hash(
    claims: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> str:
    return _canonical_hash(
        [
            *({"kind": "claim", "record": row} for row in claims),
            *({"kind": "document", "record": row} for row in documents),
        ]
    )


def unified_content_hash(
    claims: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    evidence_relations: list[dict[str, Any]],
    label_schema: dict[str, Any],
) -> str:
    return _canonical_hash(
        [
            {"kind": "claim", "record": row} for row in claims
        ]
        + [{"kind": "document", "record": row} for row in documents]
        + [{"kind": "evidence", "record": row} for row in evidence_relations]
        + [{"kind": "label_schema", "record": label_schema}]
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UnifiedDatasetError(f"UNIFIED_MISSING_INPUT: {path} does not exist.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_JSON: Could not read {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_JSON: {path} must contain an object.")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise UnifiedDatasetError(
                        f"UNIFIED_INVALID_JSON: Could not parse {path} line {line_number}: {exc.msg}."
                    ) from exc
                if not isinstance(row, dict):
                    raise UnifiedDatasetError(
                        f"UNIFIED_INVALID_JSON: {path} line {line_number} must be an object."
                    )
                rows.append(row)
    except FileNotFoundError as exc:
        raise UnifiedDatasetError(f"UNIFIED_MISSING_INPUT: {path} does not exist.") from exc
    except OSError as exc:
        raise UnifiedDatasetError(f"UNIFIED_INPUT_READ_ERROR: Could not read {path}: {exc}.") from exc
    return rows


def _required(row: dict[str, Any], field: str, kind: str) -> Any:
    if field not in row:
        raise UnifiedDatasetError(f"UNIFIED_MISSING_FIELD: {kind} is missing {field!r}.")
    return row[field]


def _validate_version(version: str) -> None:
    if not isinstance(version, str) or not version or version in {".", ".."} or VERSION_PATTERN.fullmatch(version) is None:
        raise UnifiedDatasetError("UNIFIED_INVALID_VERSION: Version contains invalid characters.")


def _id_sort_key(value: Any) -> tuple[str, int, int | str]:
    rendered = str(value)
    namespace, _, suffix = rendered.rpartition(":")
    try:
        return namespace, 0, int(suffix)
    except ValueError:
        return namespace, 1, suffix


def _normalize_document(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    document_id = _required(row, "document_id", "Document")
    prefix = f"{dataset}:document:"
    if not isinstance(document_id, str) or not document_id.startswith(prefix):
        raise UnifiedDatasetError(
            f"UNIFIED_INVALID_NAMESPACE: Document ID {document_id!r} must start with {prefix!r}."
        )
    if row.get("dataset") != dataset:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DATASET: Document {document_id} has the wrong dataset.")
    text = _required(row, "text", f"Document {document_id}")
    if not isinstance(text, str) or not text.strip():
        raise UnifiedDatasetError(f"UNIFIED_EMPTY_DOCUMENT: Document {document_id} has empty text.")
    source_type = _required(row, "source_type", f"Document {document_id}")
    if not isinstance(source_type, str) or not source_type.strip():
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} has invalid source_type.")
    sentences = row.get("sentences", row.get("abstract_sentences", []))
    if not isinstance(sentences, list) or any(not isinstance(item, str) or not item.strip() for item in sentences):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} has invalid sentences.")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} metadata must be an object.")
    source_document_id = _required(row, "source_document_id", f"Document {document_id}")
    title = row.get("title")
    source_url = row.get("source_url")
    publication_year = row.get("publication_year")
    if title is not None and not isinstance(title, str):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} has invalid title.")
    if source_url is not None and not isinstance(source_url, str):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} has invalid source_url.")
    if publication_year is not None and (
        not isinstance(publication_year, int) or isinstance(publication_year, bool)
    ):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DOCUMENT: Document {document_id} has invalid publication_year.")
    return {
        "document_id": document_id,
        "dataset": dataset,
        "source_document_id": str(source_document_id),
        "title": title,
        "source_type": source_type,
        "source_url": source_url,
        "publication_year": publication_year,
        "text": text,
        "sentences": list(sentences),
        "metadata": dict(metadata),
    }


def _normalize_claim(
    row: dict[str, Any],
    dataset: str,
    mapping: dict[str, str],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    claim_id = _required(row, "claim_id", "Claim")
    prefix = f"{dataset}:claim:"
    if not isinstance(claim_id, str) or not claim_id.startswith(prefix):
        raise UnifiedDatasetError(
            f"UNIFIED_INVALID_NAMESPACE: Claim ID {claim_id!r} must start with {prefix!r}."
        )
    if row.get("dataset") != dataset:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_DATASET: Claim {claim_id} has the wrong dataset.")
    text = _required(row, "claim_text", f"Claim {claim_id}")
    if not isinstance(text, str) or not text.strip():
        raise UnifiedDatasetError(f"UNIFIED_EMPTY_CLAIM: Claim {claim_id} has empty text.")
    original_label = _required(row, "original_label", f"Claim {claim_id}")
    unified_label = _required(row, "unified_label", f"Claim {claim_id}")
    if unified_label is not None and unified_label not in UNIFIED_LABELS:
        raise UnifiedDatasetError(f"UNIFIED_UNKNOWN_LABEL: Claim {claim_id} uses {unified_label!r}.")
    if original_label is None:
        if unified_label not in (None, "MIXED"):
            raise UnifiedDatasetError(f"UNIFIED_LABEL_MISMATCH: Claim {claim_id} has no source label.")
    elif not isinstance(original_label, str) or original_label not in mapping:
        raise UnifiedDatasetError(
            f"UNIFIED_UNKNOWN_SOURCE_LABEL: Claim {claim_id} uses unmapped source label {original_label!r}."
        )
    elif mapping[original_label] != unified_label:
        raise UnifiedDatasetError(
            f"UNIFIED_LABEL_MISMATCH: Claim {claim_id} maps {original_label!r} to "
            f"{mapping[original_label]!r}, not {unified_label!r}."
        )
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_CLAIM: Claim {claim_id} metadata must be an object.")
    source_sets = _required(row, "evidence_sets", f"Claim {claim_id}")
    if not isinstance(source_sets, list):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_EVIDENCE: Claim {claim_id} evidence_sets must be a list.")
    evidence_sets: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    for source_set in source_sets:
        if not isinstance(source_set, dict):
            raise UnifiedDatasetError(f"UNIFIED_INVALID_EVIDENCE: Claim {claim_id} has invalid evidence.")
        evidence_id = _required(source_set, "evidence_set_id", f"Claim {claim_id} evidence")
        if not isinstance(evidence_id, str) or not evidence_id.startswith(f"{claim_id}:evidence:"):
            raise UnifiedDatasetError(f"UNIFIED_INVALID_NAMESPACE: Evidence ID {evidence_id!r} is invalid.")
        if evidence_id in seen_evidence_ids:
            raise UnifiedDatasetError(f"UNIFIED_NAMESPACE_COLLISION: Evidence ID {evidence_id} is duplicated.")
        seen_evidence_ids.add(evidence_id)
        document_id = _required(source_set, "document_id", f"Evidence {evidence_id}")
        if document_id not in documents_by_id:
            raise UnifiedDatasetError(
                f"UNIFIED_MISSING_DOCUMENT: Claim {claim_id} references unknown document {document_id}."
            )
        relationship = _required(source_set, "relationship", f"Evidence {evidence_id}")
        if relationship not in UNIFIED_LABELS:
            raise UnifiedDatasetError(f"UNIFIED_UNKNOWN_LABEL: Evidence {evidence_id} uses {relationship!r}.")
        indices = source_set.get("source_sentence_indices", source_set.get("sentence_indices", []))
        if not isinstance(indices, list) or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
            raise UnifiedDatasetError(f"UNIFIED_INVALID_EVIDENCE: Evidence {evidence_id} has invalid sentence indices.")
        sentence_count = len(documents_by_id[document_id]["sentences"])
        if any(index >= sentence_count for index in indices):
            raise UnifiedDatasetError(
                f"UNIFIED_INVALID_EVIDENCE: Evidence {evidence_id} has an out-of-range sentence index."
            )
        evidence_sets.append(
            {
                "evidence_set_id": evidence_id,
                "relationship": relationship,
                "document_id": document_id,
                "source_sentence_indices": list(indices),
                "passage_ids": [],
            }
        )
    original_split = _required(row, "original_split", f"Claim {claim_id}")
    language = _required(row, "language", f"Claim {claim_id}")
    explanation = row.get("gold_explanation")
    if not isinstance(original_split, str) or not original_split:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_CLAIM: Claim {claim_id} has invalid original_split.")
    if not isinstance(language, str) or not language:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_CLAIM: Claim {claim_id} has invalid language.")
    if explanation is not None and not isinstance(explanation, str):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_CLAIM: Claim {claim_id} has invalid gold_explanation.")
    return {
        "claim_id": claim_id,
        "dataset": dataset,
        "source_claim_id": str(_required(row, "source_claim_id", f"Claim {claim_id}")),
        "claim_text": text,
        "original_split": original_split,
        "original_label": original_label,
        "unified_label": unified_label,
        "language": language,
        "evidence_sets": evidence_sets,
        "gold_explanation": explanation,
        "metadata": dict(metadata),
    }


def _validate_source_manifest(
    manifest: dict[str, Any], dataset: str, claims: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> None:
    if manifest.get("artifact_type") != "normalized_dataset" or manifest.get("dataset") != dataset:
        raise UnifiedDatasetError(f"UNIFIED_INVALID_MANIFEST: Expected normalized {dataset} dataset.")
    if manifest.get("claim_count") != len(claims) or manifest.get("document_count") != len(documents):
        raise UnifiedDatasetError(f"UNIFIED_COUNT_MISMATCH: {dataset} manifest counts do not match its files.")
    declared_hash = manifest.get("content_hash")
    if declared_hash is not None:
        if not isinstance(declared_hash, str) or SHA256_PATTERN.fullmatch(declared_hash) is None:
            raise UnifiedDatasetError(f"UNIFIED_INVALID_MANIFEST: {dataset} content_hash is invalid.")
        actual_hash = _source_content_hash(claims, documents)
        if declared_hash != actual_hash:
            raise UnifiedDatasetError(f"UNIFIED_CONTENT_HASH_MISMATCH: {dataset} source files do not match its manifest.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise UnifiedDatasetError(f"UNIFIED_INVALID_MANIFEST: {dataset} outputs are invalid.")
    expected_outputs = {
        "claims": "claims.jsonl",
        "documents": "documents.jsonl",
        "label_mapping": "label_mapping.json",
        "quality_report": "quality_report.json",
    }
    for key, filename in expected_outputs.items():
        if outputs.get(key) != filename:
            raise UnifiedDatasetError(f"UNIFIED_INVALID_MANIFEST: {dataset} output {key!r} is missing.")


def build_unified_dataset(
    scifact_dir: Path,
    healthver_dir: Path,
    pubhealth_dir: Path,
    output_root: Path,
    version: str,
) -> Path:
    """Validate and merge three adapter outputs into an immutable artifact."""
    _validate_version(version)
    version_dir = output_root / version
    if version_dir.exists():
        raise UnifiedDatasetError(f"UNIFIED_VERSION_EXISTS: Dataset version {version!r} already exists.")
    source_dirs = {
        "scifact": scifact_dir,
        "healthver": healthver_dir,
        "pubhealth": pubhealth_dir,
    }
    all_claims: list[dict[str, Any]] = []
    all_documents: list[dict[str, Any]] = []
    source_mappings: dict[str, dict[str, str]] = {}
    sources: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    seen_document_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()
    dataset_stats: dict[str, dict[str, int]] = {}
    for dataset in DATASET_ORDER:
        source_dir = source_dirs[dataset]
        for filename in REQUIRED_INPUTS:
            if not (source_dir / filename).is_file():
                raise UnifiedDatasetError(f"UNIFIED_MISSING_INPUT: Required file does not exist: {source_dir / filename}.")
        source_claims = _load_jsonl(source_dir / "claims.jsonl")
        source_documents = _load_jsonl(source_dir / "documents.jsonl")
        manifest = _load_json(source_dir / "manifest.json")
        _validate_source_manifest(manifest, dataset, source_claims, source_documents)
        try:
            mapping = extract_source_mapping(_load_json(source_dir / "label_mapping.json"), dataset)
        except ValueError as exc:
            raise UnifiedDatasetError(f"UNIFIED_INVALID_LABEL_MAPPING: {exc}") from exc
        source_mappings[dataset] = mapping
        documents = [_normalize_document(row, dataset) for row in source_documents]
        local_document_ids = {row["document_id"] for row in documents}
        if len(local_document_ids) != len(documents):
            raise UnifiedDatasetError(f"UNIFIED_NAMESPACE_COLLISION: Duplicate document ID in {dataset}.")
        local_documents_by_id = {row["document_id"]: row for row in documents}
        claims = [_normalize_claim(row, dataset, mapping, local_documents_by_id) for row in source_claims]
        local_claim_ids = {row["claim_id"] for row in claims}
        if len(local_claim_ids) != len(claims):
            raise UnifiedDatasetError(f"UNIFIED_NAMESPACE_COLLISION: Duplicate claim ID in {dataset}.")
        evidence_ids = [item["evidence_set_id"] for claim in claims for item in claim["evidence_sets"]]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise UnifiedDatasetError(f"UNIFIED_NAMESPACE_COLLISION: Duplicate evidence-set ID in {dataset}.")
        if seen_claim_ids & local_claim_ids or seen_document_ids & local_document_ids or seen_evidence_ids & set(evidence_ids):
            raise UnifiedDatasetError("UNIFIED_NAMESPACE_COLLISION: IDs collide across datasets.")
        seen_claim_ids.update(local_claim_ids)
        seen_document_ids.update(local_document_ids)
        seen_evidence_ids.update(evidence_ids)
        all_claims.extend(claims)
        all_documents.extend(documents)
        dataset_stats[dataset] = {
            "claims": len(claims),
            "documents": len(documents),
            "evidence_sets": len(evidence_ids),
        }
        sources.append(
            {
                "dataset": dataset,
                "adapter_version": str(manifest.get("adapter_version", "unknown")),
                "content_hash": manifest.get("content_hash", _source_content_hash(source_claims, source_documents)),
                "claim_count": len(claims),
                "document_count": len(documents),
            }
        )
    all_claims.sort(key=lambda row: (DATASET_RANK[row["dataset"]], SPLIT_ORDER.get(row["original_split"], 99), row["original_split"], _id_sort_key(row["claim_id"])))
    all_documents.sort(key=lambda row: (DATASET_RANK[row["dataset"]], _id_sort_key(row["document_id"])))
    evidence_relations = [
        {
            "claim_id": claim["claim_id"],
            "evidence_set_id": evidence["evidence_set_id"],
            "dataset": claim["dataset"],
            "relationship": evidence["relationship"],
            "document_id": evidence["document_id"],
            "source_sentence_indices": evidence["source_sentence_indices"],
            "passage_ids": [],
        }
        for claim in all_claims
        for evidence in claim["evidence_sets"]
    ]
    label_schema = build_label_schema(source_mappings)
    label_counts = Counter(claim["unified_label"] or "UNLABELED" for claim in all_claims)
    split_counts = Counter(claim["original_split"] for claim in all_claims)
    quality_report = {
        "artifact_version": version,
        "status": "success",
        "datasets": dataset_stats,
        "total_claims": len(all_claims),
        "total_documents": len(all_documents),
        "total_evidence_sets": len(evidence_relations),
        "label_distribution": {label: label_counts[label] for label in (*UNIFIED_LABELS, "UNLABELED")},
        "split_distribution": dict(sorted(split_counts.items())),
        "claims_with_evidence": sum(bool(claim["evidence_sets"]) for claim in all_claims),
        "claims_without_evidence": sum(not claim["evidence_sets"] for claim in all_claims),
        "claims_with_explanations": sum(bool(claim["gold_explanation"]) for claim in all_claims),
        "missing_document_references": 0,
        "namespace_collisions": 0,
        "warnings": [],
    }
    manifest = {
        "artifact_type": "unified_medical_dataset",
        "version": version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "builder_version": BUILDER_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "sources": sources,
        "claim_count": len(all_claims),
        "document_count": len(all_documents),
        "evidence_set_count": len(evidence_relations),
        "content_hash": unified_content_hash(all_claims, all_documents, evidence_relations, label_schema),
        "outputs": {
            "claims": "claims.jsonl",
            "documents": "documents.jsonl",
            "evidence_relations": "evidence_relations.jsonl",
            "label_schema": "label_schema.json",
            "quality_report": "quality_report.json",
        },
        "warnings": [],
    }
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir()
        _write_jsonl(all_claims, version_dir / "claims.jsonl")
        _write_jsonl(all_documents, version_dir / "documents.jsonl")
        _write_jsonl(evidence_relations, version_dir / "evidence_relations.jsonl")
        _write_json(label_schema, version_dir / "label_schema.json")
        _write_json(quality_report, version_dir / "quality_report.json")
        _write_json(manifest, version_dir / "manifest.json")
    except FileExistsError as exc:
        raise UnifiedDatasetError(f"UNIFIED_VERSION_EXISTS: Dataset version {version!r} already exists.") from exc
    except (OSError, ValueError) as exc:
        raise UnifiedDatasetError(f"UNIFIED_OUTPUT_WRITE_ERROR: Could not write {version_dir}: {exc}.") from exc
    return version_dir


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(value: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
