"""Normalize local SciFact source files into MedClaim's canonical format."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schemas import CanonicalClaim, CanonicalDocument, EvidenceSet
from .unified import normalized_dataset_content_hash

ADAPTER_VERSION = "1.0.0"
SCIFACT_LABEL_MAPPING = {
    "SUPPORT": "SUPPORTS",
    "CONTRADICT": "REFUTES",
}
OUTPUT_FILENAMES = (
    "claims.jsonl",
    "documents.jsonl",
    "label_mapping.json",
    "quality_report.json",
    "manifest.json",
)
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}


class SciFactPreparationError(Exception):
    """Raised when SciFact source data cannot be normalized safely."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric value {value}")


def claim_id_for(source_id: Any) -> str:
    return f"scifact:claim:{source_id}"


def document_id_for(source_id: Any) -> str:
    return f"scifact:document:{source_id}"


def clean_claim_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in value
    )
    return re.sub(r"\s+", " ", without_controls).strip()


def map_scifact_label(label: str | None, claim_id: str | None = None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        rendered_claim = claim_id or "unknown claim"
        raise SciFactPreparationError(
            "SCIFACT_UNKNOWN_LABEL: "
            f"Unsupported source label {label!r} for claim {rendered_claim}."
        )
    normalized = label.strip()
    try:
        return SCIFACT_LABEL_MAPPING[normalized]
    except KeyError as exc:
        rendered_claim = claim_id or "unknown claim"
        raise SciFactPreparationError(
            "SCIFACT_UNKNOWN_LABEL: "
            f"Unsupported source label {normalized!r} for claim {rendered_claim}."
        ) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    reason = getattr(exc, "msg", str(exc))
                    raise SciFactPreparationError(
                        "SCIFACT_INVALID_JSON: "
                        f"Could not parse {path} at line {line_number}: {reason}."
                    ) from exc
                if not isinstance(record, dict):
                    raise SciFactPreparationError(
                        "SCIFACT_INVALID_JSON: "
                        f"Expected an object in {path} at line {line_number}."
                    )
                records.append(record)
    except FileNotFoundError as exc:
        raise SciFactPreparationError(
            f"SCIFACT_MISSING_INPUT: Required input file does not exist: {path}."
        ) from exc
    except OSError as exc:
        raise SciFactPreparationError(
            f"SCIFACT_INPUT_READ_ERROR: Could not read {path}: {exc}."
        ) from exc
    return records


def _required(record: dict[str, Any], field: str, record_kind: str) -> Any:
    if field not in record or record[field] is None:
        raise SciFactPreparationError(
            "SCIFACT_MISSING_FIELD: "
            f"{record_kind} is missing required field {field!r}."
        )
    return record[field]


def normalize_scifact_document(record: dict[str, Any]) -> CanonicalDocument:
    source_id = str(_required(record, "doc_id", "Document"))
    abstract = _required(record, "abstract", f"Document {source_id}")
    if not isinstance(abstract, list) or any(
        not isinstance(sentence, str) for sentence in abstract
    ):
        raise SciFactPreparationError(
            "SCIFACT_INVALID_DOCUMENT: "
            f"Document {source_id} field 'abstract' must be a list of strings."
        )

    title = record.get("title")
    if title is not None and not isinstance(title, str):
        raise SciFactPreparationError(
            "SCIFACT_INVALID_DOCUMENT: "
            f"Document {source_id} field 'title' must be a string or null."
        )

    metadata: dict[str, Any] = {}
    if "structured" in record:
        if not isinstance(record["structured"], bool):
            raise SciFactPreparationError(
                "SCIFACT_INVALID_DOCUMENT: "
                f"Document {source_id} field 'structured' must be a boolean."
            )
        metadata["structured"] = record["structured"]

    return {
        "document_id": document_id_for(source_id),
        "dataset": "scifact",
        "source_document_id": source_id,
        "title": title,
        "source_type": "scientific_abstract",
        "abstract_sentences": list(abstract),
        "text": " ".join(abstract),
        "metadata": metadata,
    }


def _normalize_cited_document_ids(record: dict[str, Any], claim_id: str) -> list[str]:
    cited_ids = record.get("cited_doc_ids", [])
    if not isinstance(cited_ids, list):
        raise SciFactPreparationError(
            "SCIFACT_INVALID_CLAIM: "
            f"Claim {claim_id} field 'cited_doc_ids' must be a list."
        )
    return [str(source_id) for source_id in cited_ids]


def normalize_scifact_claim(
    record: dict[str, Any],
    split: str,
    documents_by_id: dict[str, CanonicalDocument],
    *,
    validate_references: bool = True,
) -> CanonicalClaim:
    source_id = str(_required(record, "id", "Claim"))
    canonical_claim_id = claim_id_for(source_id)
    claim_text = clean_claim_text(_required(record, "claim", f"Claim {source_id}"))
    if not claim_text:
        raise SciFactPreparationError(
            f"SCIFACT_EMPTY_CLAIM: Claim {canonical_claim_id} has empty text."
        )

    source_evidence = record.get("evidence", {})
    if source_evidence is None:
        source_evidence = {}
    if not isinstance(source_evidence, dict):
        raise SciFactPreparationError(
            "SCIFACT_INVALID_CLAIM: "
            f"Claim {canonical_claim_id} field 'evidence' must be an object."
        )

    evidence_sets: list[EvidenceSet] = []
    original_labels: list[str] = []
    evidence_index = 0
    for source_document_id in sorted(source_evidence, key=_source_id_sort_key):
        rationales = source_evidence[source_document_id]
        if not isinstance(rationales, list):
            raise SciFactPreparationError(
                "SCIFACT_INVALID_CLAIM: "
                f"Evidence for claim {canonical_claim_id} and document "
                f"{source_document_id} must be a list."
            )
        for rationale in rationales:
            if not isinstance(rationale, dict):
                raise SciFactPreparationError(
                    "SCIFACT_INVALID_CLAIM: "
                    f"Evidence for claim {canonical_claim_id} must contain objects."
                )
            source_label = _required(
                rationale,
                "label",
                f"Evidence for claim {canonical_claim_id}",
            )
            if not isinstance(source_label, str):
                map_scifact_label(source_label, canonical_claim_id)
            source_label = source_label.strip()
            relationship = map_scifact_label(source_label, canonical_claim_id)

            sentence_indices = _required(
                rationale,
                "sentences",
                f"Evidence for claim {canonical_claim_id}",
            )
            if not isinstance(sentence_indices, list) or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in sentence_indices
            ):
                raise SciFactPreparationError(
                    "SCIFACT_INVALID_CLAIM: "
                    f"Evidence sentence indexes for claim {canonical_claim_id} "
                    "must be a list of integers."
                )

            evidence_sets.append(
                {
                    "evidence_set_id": (
                        f"{canonical_claim_id}:evidence:{evidence_index}"
                    ),
                    "relationship": relationship,
                    "document_id": document_id_for(source_document_id),
                    "sentence_indices": list(sentence_indices),
                }
            )
            original_labels.append(source_label)
            evidence_index += 1

    unique_labels = set(original_labels)
    if len(unique_labels) == 1:
        original_label: str | None = original_labels[0]
        unified_label: str | None = map_scifact_label(
            original_label, canonical_claim_id
        )
    elif len(unique_labels) > 1:
        # SciFact labels individual rationales, so conflicting rationale labels
        # have no single original claim label.
        original_label = None
        unified_label = "MIXED"
    else:
        original_label = None
        unified_label = None

    claim: CanonicalClaim = {
        "claim_id": canonical_claim_id,
        "dataset": "scifact",
        "source_claim_id": source_id,
        "claim_text": claim_text,
        "original_split": split,
        "original_label": original_label,
        "unified_label": unified_label,
        "language": "en",
        "evidence_sets": evidence_sets,
        "metadata": {
            "cited_document_ids": _normalize_cited_document_ids(
                record, canonical_claim_id
            )
        },
    }
    if validate_references:
        validate_evidence_references(claim, documents_by_id)
    return claim


def validate_evidence_references(
    claim: CanonicalClaim,
    documents_by_id: dict[str, CanonicalDocument],
) -> None:
    seen_evidence_ids: set[str] = set()
    for evidence_set in claim["evidence_sets"]:
        evidence_id = evidence_set["evidence_set_id"]
        if evidence_id in seen_evidence_ids:
            raise SciFactPreparationError(
                "SCIFACT_DUPLICATE_EVIDENCE_ID: "
                f"Evidence-set ID {evidence_id} appears more than once."
            )
        seen_evidence_ids.add(evidence_id)

        document_id = evidence_set["document_id"]
        if document_id not in documents_by_id:
            raise SciFactPreparationError(
                "SCIFACT_MISSING_DOCUMENT: "
                f"Claim {claim['claim_id']} references document {document_id}, "
                "but that document does not exist in the loaded corpus."
            )
        sentence_count = len(documents_by_id[document_id]["abstract_sentences"])
        for sentence_index in evidence_set["sentence_indices"]:
            if not 0 <= sentence_index < sentence_count:
                raise SciFactPreparationError(
                    "SCIFACT_INVALID_SENTENCE_INDEX: "
                    f"Claim {claim['claim_id']} references sentence index "
                    f"{sentence_index} in document {document_id}, which has "
                    f"{sentence_count} abstract sentences (valid zero-based range: "
                    f"0 through {sentence_count - 1})."
                )


def _source_id_sort_key(source_id: Any) -> tuple[int, int | str]:
    rendered = str(source_id)
    try:
        return (0, int(rendered))
    except ValueError:
        return (1, rendered)


def _normalize_documents(
    source_records: Iterable[dict[str, Any]],
) -> list[CanonicalDocument]:
    documents: list[CanonicalDocument] = []
    seen_ids: set[str] = set()
    for source_record in source_records:
        document = normalize_scifact_document(source_record)
        source_id = document["source_document_id"]
        if source_id in seen_ids:
            raise SciFactPreparationError(
                "SCIFACT_DUPLICATE_DOCUMENT_ID: "
                f"Source document ID {source_id} appears more than once."
            )
        seen_ids.add(source_id)
        documents.append(document)
    return sorted(
        documents,
        key=lambda document: _source_id_sort_key(document["source_document_id"]),
    )


def _normalize_claims(
    split_records: dict[str, list[dict[str, Any]]],
    documents_by_id: dict[str, CanonicalDocument],
    *,
    validate_references: bool = True,
) -> list[CanonicalClaim]:
    claims: list[CanonicalClaim] = []
    seen_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        for source_record in split_records.get(split, []):
            source_id = str(_required(source_record, "id", "Claim"))
            if source_id in seen_ids:
                raise SciFactPreparationError(
                    "SCIFACT_DUPLICATE_CLAIM_ID: "
                    f"Source claim ID {source_id} appears more than once."
                )
            seen_ids.add(source_id)
            claims.append(
                normalize_scifact_claim(
                    source_record,
                    split,
                    documents_by_id,
                    validate_references=validate_references,
                )
            )
    return sorted(
        claims,
        key=lambda claim: (
            SPLIT_ORDER[claim["original_split"]],
            _source_id_sort_key(claim["source_claim_id"]),
        ),
    )


def _reference_validation_errors(
    claims: list[CanonicalClaim],
    documents_by_id: dict[str, CanonicalDocument],
) -> tuple[list[str], int, int]:
    errors: list[str] = []
    missing_documents = 0
    invalid_sentences = 0
    for claim in claims:
        for evidence_set in claim["evidence_sets"]:
            document_id = evidence_set["document_id"]
            if document_id not in documents_by_id:
                missing_documents += 1
                errors.append(
                    "SCIFACT_MISSING_DOCUMENT: "
                    f"Claim {claim['claim_id']} references document {document_id}, "
                    "but that document does not exist in the loaded corpus."
                )
                continue

            sentence_count = len(
                documents_by_id[document_id]["abstract_sentences"]
            )
            for sentence_index in evidence_set["sentence_indices"]:
                if not 0 <= sentence_index < sentence_count:
                    invalid_sentences += 1
                    errors.append(
                        "SCIFACT_INVALID_SENTENCE_INDEX: "
                        f"Claim {claim['claim_id']} references sentence index "
                        f"{sentence_index} in document {document_id}, which has "
                        f"{sentence_count} abstract sentences (valid zero-based "
                        f"range: 0 through {sentence_count - 1})."
                    )
    return errors, missing_documents, invalid_sentences


def build_quality_report(
    documents: list[CanonicalDocument],
    claims: list[CanonicalClaim],
    warnings: list[str],
) -> dict[str, Any]:
    split_counts = Counter(claim["original_split"] for claim in claims)
    original_counts = Counter(
        claim["original_label"] or "UNLABELED" for claim in claims
    )
    unified_counts = Counter(
        claim["unified_label"] or "UNLABELED" for claim in claims
    )
    claims_with_evidence = sum(bool(claim["evidence_sets"]) for claim in claims)
    claims_with_citations_without_evidence = sum(
        bool(claim["metadata"]["cited_document_ids"])
        and not claim["evidence_sets"]
        for claim in claims
    )
    return {
        "dataset": "scifact",
        "status": "success",
        "document_count": len(documents),
        "claim_count": len(claims),
        "split_counts": {
            split: split_counts[split] for split in ("train", "dev", "test")
        },
        "original_label_counts": dict(sorted(original_counts.items())),
        "unified_label_counts": dict(sorted(unified_counts.items())),
        "unlabeled_claims": unified_counts["UNLABELED"],
        "claims_with_evidence": claims_with_evidence,
        "claims_without_evidence": len(claims) - claims_with_evidence,
        "claims_with_citations_without_evidence": (
            claims_with_citations_without_evidence
        ),
        "evidence_set_count": sum(
            len(claim["evidence_sets"]) for claim in claims
        ),
        "missing_document_references": 0,
        "invalid_sentence_references": 0,
        "duplicate_claim_ids": 0,
        "duplicate_document_ids": 0,
        "warnings": warnings,
    }


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
    except (OSError, ValueError) as exc:
        raise SciFactPreparationError(
            f"SCIFACT_OUTPUT_WRITE_ERROR: Could not write {path}: {exc}."
        ) from exc


def _write_json(data: dict[str, Any], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(
                data,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            output_file.write("\n")
    except (OSError, ValueError) as exc:
        raise SciFactPreparationError(
            f"SCIFACT_OUTPUT_WRITE_ERROR: Could not write {path}: {exc}."
        ) from exc


def _resolve_inputs(input_dir: Path) -> tuple[dict[str, Path], list[str]]:
    corpus = input_dir / "corpus.jsonl"
    train = input_dir / "claims_train.jsonl"
    dev_candidates = (
        input_dir / "claims_dev.jsonl",
        input_dir / "claims_validation.jsonl",
    )
    test = input_dir / "claims_test.jsonl"

    for kind, path in (("corpus", corpus), ("training claims", train)):
        if not path.is_file():
            raise SciFactPreparationError(
                f"SCIFACT_MISSING_INPUT: Required {kind} file does not exist: {path}."
            )
    dev = next((path for path in dev_candidates if path.is_file()), None)
    if dev is None:
        raise SciFactPreparationError(
            "SCIFACT_MISSING_INPUT: Required development claims file does not "
            f"exist (looked for {dev_candidates[0]} and {dev_candidates[1]})."
        )

    inputs = {"corpus": corpus, "train": train, "dev": dev}
    warnings: list[str] = []
    if test.is_file():
        inputs["test"] = test
    else:
        warnings.append(
            f"Optional test split is unavailable: {test.name}."
        )
    return inputs, warnings


def prepare_scifact(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize local SciFact files and write deterministic artifacts."""
    output_paths = {
        filename: output_dir / filename for filename in OUTPUT_FILENAMES
    }
    if not overwrite and any(path.exists() for path in output_paths.values()):
        raise SciFactPreparationError(
            "Output files already exist. Use --overwrite to replace them."
        )

    inputs, warnings = _resolve_inputs(input_dir)
    documents = _normalize_documents(load_jsonl(inputs["corpus"]))
    documents_by_id = {document["document_id"]: document for document in documents}
    split_records = {
        split: load_jsonl(path)
        for split, path in inputs.items()
        if split in SPLIT_ORDER
    }
    claims = _normalize_claims(
        split_records,
        documents_by_id,
        validate_references=False,
    )
    quality_report = build_quality_report(documents, claims, warnings)

    reference_errors, missing_documents, invalid_sentences = (
        _reference_validation_errors(claims, documents_by_id)
    )
    if reference_errors:
        quality_report["status"] = "failed"
        quality_report["missing_document_references"] = missing_documents
        quality_report["invalid_sentence_references"] = invalid_sentences
        quality_report["errors"] = reference_errors
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SciFactPreparationError(
                f"SCIFACT_OUTPUT_WRITE_ERROR: Could not create {output_dir}: {exc}."
            ) from exc
        _write_json(quality_report, output_paths["quality_report.json"])
        raise SciFactPreparationError(reference_errors[0])

    input_files = {
        "corpus": inputs["corpus"].name,
        "train_claims": inputs["train"].name,
        "dev_claims": inputs["dev"].name,
    }
    if "test" in inputs:
        input_files["test_claims"] = inputs["test"].name
    manifest = {
        "artifact_type": "normalized_dataset",
        "dataset": "scifact",
        "adapter_version": ADAPTER_VERSION,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "input_files": input_files,
        "outputs": {
            "claims": "claims.jsonl",
            "documents": "documents.jsonl",
            "label_mapping": "label_mapping.json",
            "quality_report": "quality_report.json",
        },
        "document_count": len(documents),
        "claim_count": len(claims),
        "content_hash": normalized_dataset_content_hash(claims, documents),
    }
    label_mapping = {
        "dataset": "scifact",
        "schema_version": "1.0.0",
        "mappings": SCIFACT_LABEL_MAPPING,
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SciFactPreparationError(
            f"SCIFACT_OUTPUT_WRITE_ERROR: Could not create {output_dir}: {exc}."
        ) from exc
    _write_jsonl(claims, output_paths["claims.jsonl"])
    _write_jsonl(documents, output_paths["documents.jsonl"])
    _write_json(label_mapping, output_paths["label_mapping.json"])
    _write_json(quality_report, output_paths["quality_report.json"])
    _write_json(manifest, output_paths["manifest.json"])
    return quality_report
