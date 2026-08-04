"""Build an immutable, versioned passage corpus from normalized SciFact data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

BUILDER_VERSION = "1.0.0"
REQUIRED_INPUTS = ("documents.jsonl", "claims.jsonl", "manifest.json")
OUTPUT_FILENAMES = (
    "documents.jsonl",
    "passages.jsonl",
    "gold_evidence.jsonl",
    "quality_report.json",
    "manifest.json",
)
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class CorpusBuildError(Exception):
    """Raised when a corpus cannot be built without losing traceability."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric value {value}")


def sha256_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def passage_id_for(document_id: str, passage_index: int) -> str:
    return f"{document_id}:p:{passage_index}"


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def validate_version(version: str) -> str:
    if (
        not isinstance(version, str)
        or not version
        or version in {".", ".."}
        or VERSION_PATTERN.fullmatch(version) is None
    ):
        raise CorpusBuildError(
            "CORPUS_INVALID_VERSION: Version must be non-empty and contain only "
            "letters, numbers, dots, underscores, and hyphens."
        )
    return version


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    reason = getattr(exc, "msg", str(exc))
                    raise CorpusBuildError(
                        "CORPUS_INVALID_JSON: "
                        f"Could not parse {path} at line {line_number}: {reason}."
                    ) from exc
                if not isinstance(record, dict):
                    raise CorpusBuildError(
                        "CORPUS_INVALID_JSON: "
                        f"Expected an object in {path} at line {line_number}."
                    )
                records.append(record)
    except FileNotFoundError as exc:
        raise CorpusBuildError(
            f"CORPUS_MISSING_INPUT: Required input file does not exist: {path}."
        ) from exc
    except OSError as exc:
        raise CorpusBuildError(
            f"CORPUS_INPUT_READ_ERROR: Could not read {path}: {exc}."
        ) from exc
    return records


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file, parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise CorpusBuildError(
            f"CORPUS_MISSING_INPUT: Required input file does not exist: {path}."
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        reason = getattr(exc, "msg", str(exc))
        raise CorpusBuildError(
            f"CORPUS_INVALID_JSON: Could not parse {path}: {reason}."
        ) from exc
    except OSError as exc:
        raise CorpusBuildError(
            f"CORPUS_INPUT_READ_ERROR: Could not read {path}: {exc}."
        ) from exc
    if not isinstance(value, dict):
        raise CorpusBuildError(
            f"CORPUS_INVALID_JSON: Expected an object in {path}."
        )
    return value


def _required(record: dict[str, Any], field: str, record_kind: str) -> Any:
    if field not in record or record[field] is None:
        raise CorpusBuildError(
            "CORPUS_MISSING_FIELD: "
            f"{record_kind} is missing required field {field!r}."
        )
    return record[field]


def _namespaced_id_sort_key(value: Any) -> tuple[str, int, int | str]:
    rendered = str(value)
    namespace, _, suffix = rendered.rpartition(":")
    try:
        return (namespace, 0, int(suffix))
    except ValueError:
        return (namespace, 1, suffix)


def _validate_document_source(source: dict[str, Any]) -> tuple[str, list[str], str]:
    document_id = _required(source, "document_id", "Document")
    if not isinstance(document_id, str) or not document_id.strip():
        raise CorpusBuildError("CORPUS_EMPTY_DOCUMENT_ID: Document ID is empty.")
    if source.get("dataset") != "scifact":
        raise CorpusBuildError(
            "CORPUS_INVALID_DOCUMENT: "
            f"Document {document_id} must have dataset='scifact'."
        )

    sentences = _required(source, "abstract_sentences", f"Document {document_id}")
    if not isinstance(sentences, list) or not sentences:
        raise CorpusBuildError(
            "CORPUS_INVALID_SENTENCES: "
            f"Document {document_id} must contain a non-empty sentence list."
        )
    for sentence_index, sentence in enumerate(sentences):
        if not isinstance(sentence, str):
            raise CorpusBuildError(
                "CORPUS_INVALID_SENTENCES: "
                f"Document {document_id} sentence {sentence_index} is not a string."
            )
        if not sentence.strip():
            raise CorpusBuildError(
                "CORPUS_EMPTY_SENTENCE: "
                f"Document {document_id} contains an empty sentence at index "
                f"{sentence_index}. The sentence cannot be removed because doing "
                "so would change evidence indexes."
            )

    text = " ".join(sentences)
    source_text = _required(source, "text", f"Document {document_id}")
    if not isinstance(source_text, str) or source_text != text:
        raise CorpusBuildError(
            "CORPUS_DOCUMENT_TEXT_MISMATCH: "
            f"Document {document_id} text does not match its abstract sentences."
        )
    return document_id, sentences, text


def build_document_record(
    source_document: dict[str, Any], corpus_version: str
) -> dict[str, Any]:
    document_id, sentences, text = _validate_document_source(source_document)
    source_document_id = _required(
        source_document, "source_document_id", f"Document {document_id}"
    )
    title = source_document.get("title")
    if title is not None and not isinstance(title, str):
        raise CorpusBuildError(
            "CORPUS_INVALID_DOCUMENT: "
            f"Document {document_id} title must be a string or null."
        )
    source_type = _required(
        source_document, "source_type", f"Document {document_id}"
    )
    if not isinstance(source_type, str):
        raise CorpusBuildError(
            f"CORPUS_INVALID_DOCUMENT: Document {document_id} source_type is invalid."
        )
    metadata = source_document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CorpusBuildError(
            f"CORPUS_INVALID_DOCUMENT: Document {document_id} metadata is invalid."
        )
    output_metadata = dict(metadata)
    output_metadata["sentence_count"] = len(sentences)

    return {
        "document_id": document_id,
        "dataset": "scifact",
        "source_document_id": str(source_document_id),
        "title": title,
        "source_type": source_type,
        "source_url": source_document.get("source_url"),
        "publication_year": source_document.get("publication_year"),
        "text": text,
        "content_hash": sha256_text(text),
        "corpus_version": corpus_version,
        "metadata": output_metadata,
    }


def _sentence_spans(sentences: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence_index, sentence in enumerate(sentences):
        start_char = cursor
        end_char = start_char + len(sentence)
        spans.append((start_char, end_char))
        cursor = end_char
        if sentence_index < len(sentences) - 1:
            cursor += 1
    return spans


def _build_passage(
    document_id: str,
    document_text: str,
    passage_index: int,
    sentence_indices: list[int],
    spans: list[tuple[int, int]],
    corpus_version: str,
) -> dict[str, Any]:
    start_char = spans[sentence_indices[0]][0]
    end_char = spans[sentence_indices[-1]][1]
    text = document_text[start_char:end_char]
    passage = {
        "passage_id": passage_id_for(document_id, passage_index),
        "document_id": document_id,
        "dataset": "scifact",
        "passage_index": passage_index,
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
        "token_count": whitespace_token_count(text),
        "content_hash": sha256_text(text),
        "corpus_version": corpus_version,
        "metadata": {
            "section": "abstract",
            "source_sentence_indices": sentence_indices,
            "is_gold_for_any_claim": False,
        },
    }
    if document_text[start_char:end_char] != text:
        raise CorpusBuildError(
            "CORPUS_INVALID_OFFSETS: "
            f"Passage {passage['passage_id']} does not resolve to its text."
        )
    return passage


def build_sentence_passages(
    source_document: dict[str, Any],
    corpus_version: str,
    max_passage_words: int,
) -> list[dict[str, Any]]:
    document_id, sentences, document_text = _validate_document_source(
        source_document
    )
    spans = _sentence_spans(sentences)
    passages: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(sentences):
        word_count = whitespace_token_count(sentence)
        if word_count > max_passage_words:
            raise CorpusBuildError(
                "CORPUS_PASSAGE_TOO_LONG: "
                f"Document {document_id} sentence {sentence_index} contains "
                f"{word_count} words, which exceeds "
                f"max_passage_words={max_passage_words}."
            )
        passages.append(
            _build_passage(
                document_id,
                document_text,
                sentence_index,
                [sentence_index],
                spans,
                corpus_version,
            )
        )
    return passages


def build_merged_passages(
    source_document: dict[str, Any],
    corpus_version: str,
    short_sentence_word_threshold: int,
    max_passage_words: int,
) -> list[dict[str, Any]]:
    document_id, sentences, document_text = _validate_document_source(
        source_document
    )
    spans = _sentence_spans(sentences)
    passages: list[dict[str, Any]] = []
    sentence_index = 0
    while sentence_index < len(sentences):
        current_words = whitespace_token_count(sentences[sentence_index])
        if current_words > max_passage_words:
            raise CorpusBuildError(
                "CORPUS_PASSAGE_TOO_LONG: "
                f"Document {document_id} sentence {sentence_index} contains "
                f"{current_words} words, which exceeds "
                f"max_passage_words={max_passage_words}."
            )

        source_indices = [sentence_index]
        if (
            current_words < short_sentence_word_threshold
            and sentence_index + 1 < len(sentences)
        ):
            next_words = whitespace_token_count(sentences[sentence_index + 1])
            if current_words + next_words <= max_passage_words:
                source_indices.append(sentence_index + 1)

        passages.append(
            _build_passage(
                document_id,
                document_text,
                len(passages),
                source_indices,
                spans,
                corpus_version,
            )
        )
        sentence_index += len(source_indices)
    return passages


def _validate_claim_source(claim: dict[str, Any]) -> None:
    claim_id = _required(claim, "claim_id", "Claim")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise CorpusBuildError("CORPUS_INVALID_CLAIM: Claim ID is empty.")
    if claim.get("dataset") != "scifact":
        raise CorpusBuildError(
            f"CORPUS_INVALID_CLAIM: Claim {claim_id} must have dataset='scifact'."
        )
    _required(claim, "original_split", f"Claim {claim_id}")
    if "unified_label" not in claim:
        raise CorpusBuildError(
            f"CORPUS_MISSING_FIELD: Claim {claim_id} is missing 'unified_label'."
        )
    evidence_sets = _required(claim, "evidence_sets", f"Claim {claim_id}")
    if not isinstance(evidence_sets, list):
        raise CorpusBuildError(
            f"CORPUS_INVALID_CLAIM: Claim {claim_id} evidence_sets must be a list."
        )


def resolve_gold_evidence(
    claims: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    passage_by_source_sentence: dict[tuple[str, int], dict[str, Any]] = {}
    passages_by_id = {passage["passage_id"]: passage for passage in passages}
    document_ids = {passage["document_id"] for passage in passages}
    for passage in passages:
        for sentence_index in passage["metadata"]["source_sentence_indices"]:
            key = (passage["document_id"], sentence_index)
            if key in passage_by_source_sentence:
                raise CorpusBuildError(
                    "CORPUS_AMBIGUOUS_SENTENCE_MAPPING: "
                    f"Document {key[0]} sentence {key[1]} maps to multiple passages."
                )
            passage_by_source_sentence[key] = passage

    gold_records: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    for claim in claims:
        _validate_claim_source(claim)
        claim_id = claim["claim_id"]
        if claim_id in seen_claim_ids:
            raise CorpusBuildError(
                f"CORPUS_DUPLICATE_CLAIM_ID: Claim ID {claim_id} appears more than once."
            )
        seen_claim_ids.add(claim_id)
        output_evidence: list[dict[str, Any]] = []
        seen_evidence_ids: set[str] = set()
        for evidence_set in claim["evidence_sets"]:
            if not isinstance(evidence_set, dict):
                raise CorpusBuildError(
                    f"CORPUS_INVALID_CLAIM: Claim {claim_id} has invalid evidence."
                )
            evidence_set_id = _required(
                evidence_set, "evidence_set_id", f"Claim {claim_id} evidence"
            )
            if evidence_set_id in seen_evidence_ids:
                raise CorpusBuildError(
                    "CORPUS_DUPLICATE_EVIDENCE_ID: "
                    f"Evidence-set ID {evidence_set_id} appears more than once."
                )
            seen_evidence_ids.add(evidence_set_id)
            document_id = _required(
                evidence_set, "document_id", f"Evidence {evidence_set_id}"
            )
            if document_id not in document_ids:
                raise CorpusBuildError(
                    "CORPUS_MISSING_EVIDENCE_DOCUMENT: "
                    f"Claim {claim_id} references unknown document {document_id}."
                )
            sentence_indices = _required(
                evidence_set, "sentence_indices", f"Evidence {evidence_set_id}"
            )
            if not isinstance(sentence_indices, list) or not sentence_indices or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in sentence_indices
            ):
                raise CorpusBuildError(
                    "CORPUS_INVALID_EVIDENCE: "
                    f"Evidence {evidence_set_id} sentence_indices must be a "
                    "non-empty list of integers."
                )

            resolved_passages: dict[str, dict[str, Any]] = {}
            for sentence_index in sentence_indices:
                key = (document_id, sentence_index)
                if key not in passage_by_source_sentence:
                    raise CorpusBuildError(
                        "CORPUS_UNRESOLVED_SENTENCE_REFERENCE: "
                        f"Claim {claim_id} references sentence index {sentence_index} "
                        f"in document {document_id}, but it cannot be mapped to a passage."
                    )
                passage = passage_by_source_sentence[key]
                resolved_passages[passage["passage_id"]] = passage

            ordered_passage_ids = sorted(
                resolved_passages,
                key=lambda passage_id: (
                    _namespaced_id_sort_key(
                        passages_by_id[passage_id]["document_id"]
                    ),
                    passages_by_id[passage_id]["passage_index"],
                ),
            )
            for passage_id in ordered_passage_ids:
                passages_by_id[passage_id]["metadata"][
                    "is_gold_for_any_claim"
                ] = True
            output_evidence.append(
                {
                    "evidence_set_id": evidence_set_id,
                    "relationship": _required(
                        evidence_set, "relationship", f"Evidence {evidence_set_id}"
                    ),
                    "document_id": document_id,
                    "source_sentence_indices": list(sentence_indices),
                    "passage_ids": ordered_passage_ids,
                }
            )
        gold_records.append(
            {
                "claim_id": claim_id,
                "dataset": "scifact",
                "original_split": claim["original_split"],
                "unified_label": claim["unified_label"],
                "evidence_sets": output_evidence,
            }
        )

    return sorted(
        gold_records,
        key=lambda record: (
            SPLIT_ORDER.get(record["original_split"], len(SPLIT_ORDER)),
            _namespaced_id_sort_key(record["claim_id"]),
        ),
    )


def build_quality_report(
    documents: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
    configuration: dict[str, Any],
    corpus_version: str,
) -> dict[str, Any]:
    word_counts = [passage["token_count"] for passage in passages]
    content_counts = Counter(passage["content_hash"] for passage in passages)
    duplicate_sizes = [count for count in content_counts.values() if count > 1]
    warnings: list[str] = []
    if duplicate_sizes:
        warnings.append(
            f"Found {len(duplicate_sizes)} exact duplicate passage groups; "
            "all passages were retained."
        )

    gold_passage_ids = {
        passage_id
        for claim in gold_evidence
        for evidence_set in claim["evidence_sets"]
        for passage_id in evidence_set["passage_ids"]
    }
    return {
        "dataset": "scifact",
        "corpus_version": corpus_version,
        "status": "success",
        "chunking": {
            "mode": "sentence",
            "merge_short_sentences": configuration["merge_short_sentences"],
            "short_sentence_word_threshold": configuration[
                "short_sentence_word_threshold"
            ],
            "max_passage_words": configuration["max_passage_words"],
        },
        "document_count": len(documents),
        "passage_count": len(passages),
        "documents_without_sentences": 0,
        "empty_sentences": 0,
        "minimum_passage_words": min(word_counts, default=0),
        "maximum_passage_words": max(word_counts, default=0),
        "average_passage_words": round(mean(word_counts), 2) if word_counts else 0.0,
        "gold_claim_count": sum(
            bool(claim["evidence_sets"]) for claim in gold_evidence
        ),
        "gold_evidence_set_count": sum(
            len(claim["evidence_sets"]) for claim in gold_evidence
        ),
        "unique_gold_passage_count": len(gold_passage_ids),
        "unresolved_document_references": 0,
        "unresolved_sentence_references": 0,
        "exact_duplicate_passage_groups": len(duplicate_sizes),
        "exact_duplicate_passage_count": sum(duplicate_sizes),
        "warnings": warnings,
    }


def corpus_content_hash(passages: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for passage in passages:
        serialized = json.dumps(
            passage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        hasher.update(serialized.encode("utf-8"))
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(
            f"CORPUS_OUTPUT_WRITE_ERROR: Could not write {path}: {exc}."
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
        raise CorpusBuildError(
            f"CORPUS_OUTPUT_WRITE_ERROR: Could not write {path}: {exc}."
        ) from exc


def _portable_source_manifest(path: Path) -> str:
    return path.name if path.is_absolute() else path.as_posix()


def build_scifact_corpus(
    input_dir: Path,
    output_root: Path,
    version: str,
    merge_short_sentences: bool = False,
    short_sentence_word_threshold: int = 5,
    max_passage_words: int = 120,
) -> Path:
    """Build an immutable SciFact evidence corpus and return its directory."""
    validate_version(version)
    if short_sentence_word_threshold < 1:
        raise CorpusBuildError(
            "CORPUS_INVALID_CONFIGURATION: "
            "short_sentence_word_threshold must be at least 1."
        )
    if max_passage_words < 1:
        raise CorpusBuildError(
            "CORPUS_INVALID_CONFIGURATION: max_passage_words must be at least 1."
        )
    version_dir = output_root / version
    if version_dir.exists():
        raise CorpusBuildError(
            "CORPUS_VERSION_EXISTS: "
            f"Corpus version {version!r} already exists. Choose a new version or "
            "remove the existing development artifact manually."
        )

    input_paths = {filename: input_dir / filename for filename in REQUIRED_INPUTS}
    for path in input_paths.values():
        if not path.is_file():
            raise CorpusBuildError(
                f"CORPUS_MISSING_INPUT: Required input file does not exist: {path}."
            )

    source_documents = _load_jsonl(input_paths["documents.jsonl"])
    claims = _load_jsonl(input_paths["claims.jsonl"])
    source_manifest = _load_json(input_paths["manifest.json"])
    source_adapter_version = _required(
        source_manifest, "adapter_version", "Source manifest"
    )

    seen_document_ids: set[str] = set()
    for source_document in source_documents:
        document_id, _, _ = _validate_document_source(source_document)
        if document_id in seen_document_ids:
            raise CorpusBuildError(
                "CORPUS_DUPLICATE_DOCUMENT_ID: "
                f"Document ID {document_id} appears more than once."
            )
        seen_document_ids.add(document_id)
    source_documents.sort(
        key=lambda document: _namespaced_id_sort_key(document["document_id"])
    )

    documents = [
        build_document_record(source_document, version)
        for source_document in source_documents
    ]
    passages: list[dict[str, Any]] = []
    for source_document in source_documents:
        if merge_short_sentences:
            passages.extend(
                build_merged_passages(
                    source_document,
                    version,
                    short_sentence_word_threshold,
                    max_passage_words,
                )
            )
        else:
            passages.extend(
                build_sentence_passages(
                    source_document,
                    version,
                    max_passage_words,
                )
            )
    passages.sort(
        key=lambda passage: (
            _namespaced_id_sort_key(passage["document_id"]),
            passage["passage_index"],
        )
    )
    gold_evidence = resolve_gold_evidence(claims, passages)
    configuration = {
        "chunking_mode": "sentence",
        "merge_short_sentences": merge_short_sentences,
        "short_sentence_word_threshold": short_sentence_word_threshold,
        "max_passage_words": max_passage_words,
    }
    quality_report = build_quality_report(
        documents, passages, gold_evidence, configuration, version
    )
    manifest = {
        "artifact_type": "medical_evidence_corpus",
        "dataset": "scifact",
        "corpus_version": version,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "builder_version": BUILDER_VERSION,
        "source_adapter_version": str(source_adapter_version),
        "source_manifest": _portable_source_manifest(input_paths["manifest.json"]),
        "configuration": configuration,
        "document_count": len(documents),
        "passage_count": len(passages),
        "claim_count": len(gold_evidence),
        "gold_evidence_set_count": quality_report["gold_evidence_set_count"],
        "content_hash": corpus_content_hash(passages),
        "outputs": {
            "documents": "documents.jsonl",
            "passages": "passages.jsonl",
            "gold_evidence": "gold_evidence.jsonl",
            "quality_report": "quality_report.json",
        },
        "warnings": quality_report["warnings"],
    }

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir()
    except FileExistsError as exc:
        raise CorpusBuildError(
            "CORPUS_VERSION_EXISTS: "
            f"Corpus version {version!r} already exists. Choose a new version or "
            "remove the existing development artifact manually."
        ) from exc
    except OSError as exc:
        raise CorpusBuildError(
            f"CORPUS_OUTPUT_WRITE_ERROR: Could not create {version_dir}: {exc}."
        ) from exc

    _write_jsonl(documents, version_dir / "documents.jsonl")
    _write_jsonl(passages, version_dir / "passages.jsonl")
    _write_jsonl(gold_evidence, version_dir / "gold_evidence.jsonl")
    _write_json(quality_report, version_dir / "quality_report.json")
    _write_json(manifest, version_dir / "manifest.json")
    return version_dir
