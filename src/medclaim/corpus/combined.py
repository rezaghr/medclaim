"""Build a deterministic evidence corpus from a unified medical dataset."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from medclaim.corpus.scifact_corpus import (
    CorpusBuildError,
    corpus_content_hash,
    passage_id_for,
    sha256_text,
    validate_version,
    whitespace_token_count,
)
from medclaim.datasets.constants import DATASET_ORDER, DATASET_RANK
from medclaim.datasets.label_schema import UNIFIED_LABELS
from medclaim.datasets.unified import (
    SPLIT_ORDER,
    unified_content_hash,
)

BUILDER_VERSION = "1.1.0"
REQUIRED_INPUTS = (
    "claims.jsonl",
    "documents.jsonl",
    "evidence_relations.jsonl",
    "label_schema.json",
    "manifest.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusBuildError(f"CORPUS_MISSING_INPUT: Required input does not exist: {path}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"CORPUS_INVALID_JSON: Could not read {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise CorpusBuildError(f"CORPUS_INVALID_JSON: {path} must contain an object.")
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
                    raise CorpusBuildError(
                        f"CORPUS_INVALID_JSON: Could not parse {path} line {line_number}: {exc.msg}."
                    ) from exc
                if not isinstance(row, dict):
                    raise CorpusBuildError(f"CORPUS_INVALID_JSON: {path} line {line_number} must be an object.")
                rows.append(row)
    except FileNotFoundError as exc:
        raise CorpusBuildError(f"CORPUS_MISSING_INPUT: Required input does not exist: {path}.") from exc
    except OSError as exc:
        raise CorpusBuildError(f"CORPUS_INPUT_READ_ERROR: Could not read {path}: {exc}.") from exc
    return rows


def _id_sort_key(value: Any) -> tuple[str, int, int | str]:
    rendered = str(value)
    namespace, _, suffix = rendered.rpartition(":")
    try:
        return namespace, 0, int(suffix)
    except ValueError:
        return namespace, 1, suffix


def _validate_unified_input(
    manifest: dict[str, Any],
    claims: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    label_schema: dict[str, Any],
) -> None:
    if manifest.get("artifact_type") != "unified_medical_dataset":
        raise CorpusBuildError("CORPUS_INVALID_SOURCE: Expected a unified medical dataset.")
    expected_counts = {
        "claim_count": len(claims),
        "document_count": len(documents),
        "evidence_set_count": len(evidence),
    }
    for field, actual in expected_counts.items():
        if manifest.get(field) != actual:
            raise CorpusBuildError(f"CORPUS_SOURCE_COUNT_MISMATCH: {field} does not match source files.")
    actual_hash = unified_content_hash(claims, documents, evidence, label_schema)
    if manifest.get("content_hash") != actual_hash:
        raise CorpusBuildError("CORPUS_SOURCE_HASH_MISMATCH: Unified dataset files do not match its manifest.")
    expected_relations = [
        {
            "claim_id": claim.get("claim_id"),
            "evidence_set_id": item.get("evidence_set_id"),
            "dataset": claim.get("dataset"),
            "relationship": item.get("relationship"),
            "document_id": item.get("document_id"),
            "source_sentence_indices": item.get("source_sentence_indices"),
            "passage_ids": item.get("passage_ids"),
        }
        for claim in claims
        for item in claim.get("evidence_sets", [])
    ]
    if evidence != expected_relations:
        raise CorpusBuildError(
            "CORPUS_SOURCE_EVIDENCE_MISMATCH: evidence_relations.jsonl does not "
            "match the nested claim evidence."
        )
    source_mappings = label_schema.get("source_mappings")
    if not isinstance(source_mappings, dict) or set(source_mappings) != set(DATASET_ORDER):
        raise CorpusBuildError(
            "CORPUS_INVALID_SOURCE: Label schema must contain all dataset mappings."
        )


def _sentence_spans(document: dict[str, Any]) -> list[tuple[int, int, int]]:
    text = document["text"]
    cursor = 0
    spans: list[tuple[int, int, int]] = []
    for index, sentence in enumerate(document["sentences"]):
        start = text.find(sentence, cursor)
        if start < 0:
            raise CorpusBuildError(
                f"CORPUS_DOCUMENT_TEXT_MISMATCH: Sentence {index} of {document['document_id']} "
                "cannot be located in document text."
            )
        end = start + len(sentence)
        spans.append((start, end, index))
        cursor = end
    return spans


def _rule_based_spans(text: str, max_words: int) -> list[tuple[int, int, list[int]]]:
    sentence_spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        if text[start:match.start()].strip():
            sentence_spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        sentence_spans.append((start, len(text)))
    if not sentence_spans:
        sentence_spans = [(0, len(text))]

    chunks: list[tuple[int, int, list[int]]] = []
    for start, end in sentence_spans:
        words = list(re.finditer(r"\S+", text[start:end]))
        if not words:
            continue
        for offset in range(0, len(words), max_words):
            group = words[offset : offset + max_words]
            chunk_start = start + group[0].start()
            chunk_end = start + group[-1].end()
            chunks.append((chunk_start, chunk_end, []))
    return chunks


def _split_source_span(
    text: str,
    start: int,
    end: int,
    source_sentence_index: int,
    max_words: int,
) -> list[tuple[int, int, list[int]]]:
    """Split an oversized source sentence without losing its provenance."""
    words = list(re.finditer(r"\S+", text[start:end]))
    groups = [words[offset : offset + max_words] for offset in range(0, len(words), max_words)]
    for group_index, group in enumerate(groups):
        if (
            group
            and not any(character.isalnum() for word in group for character in word.group())
            and group_index > 0
            and len(groups[group_index - 1]) > 1
        ):
            group.insert(0, groups[group_index - 1].pop())

    chunks: list[tuple[int, int, list[int]]] = []
    for group in groups:
        if group:
            chunks.append(
                (
                    start + group[0].start(),
                    start + group[-1].end(),
                    [source_sentence_index],
                )
            )
    return chunks


def build_document_passages(
    document: dict[str, Any], corpus_version: str, max_passage_words: int
) -> list[dict[str, Any]]:
    """Create source-aware passages with stable offsets and IDs."""
    dataset = document.get("dataset")
    if dataset not in DATASET_ORDER:
        raise CorpusBuildError(f"CORPUS_INVALID_DOCUMENT: Unknown dataset {dataset!r}.")
    document_id = document.get("document_id")
    text = document.get("text")
    source_type = document.get("source_type")
    metadata = document.get("metadata")
    sentences = document.get("sentences")
    if not isinstance(document_id, str) or not document_id.startswith(f"{dataset}:document:"):
        raise CorpusBuildError("CORPUS_INVALID_DOCUMENT: Invalid document namespace.")
    if not isinstance(text, str) or not text.strip() or not isinstance(source_type, str) or not source_type:
        raise CorpusBuildError(f"CORPUS_INVALID_DOCUMENT: Document {document_id} is incomplete.")
    if not isinstance(metadata, dict) or not isinstance(sentences, list):
        raise CorpusBuildError(f"CORPUS_INVALID_DOCUMENT: Document {document_id} has invalid metadata or sentences.")
    if sentences:
        spans = [
            chunk
            for start, end, index in _sentence_spans(document)
            for chunk in _split_source_span(
                text, start, end, index, max_passage_words
            )
        ]
    elif dataset == "pubhealth":
        spans = _rule_based_spans(text, max_passage_words)
    else:
        spans = [(0, len(text), [])]
    # Some source abstracts contain punctuation-only "sentences" caused by noisy
    # upstream segmentation. Keep their provenance, but attach them to a neighboring
    # passage so every emitted passage remains indexable.
    merged_spans: list[tuple[int, int, list[int]]] = []
    pending: tuple[int, list[int]] | None = None
    for start, end, indices in spans:
        if any(character.isalnum() for character in text[start:end]):
            if pending is not None:
                start = pending[0]
                indices = [*pending[1], *indices]
                pending = None
            merged_spans.append((start, end, indices))
        elif merged_spans:
            prior_start, _, prior_indices = merged_spans[-1]
            merged_spans[-1] = (prior_start, end, [*prior_indices, *indices])
        elif pending is None:
            pending = (start, list(indices))
        else:
            pending = (pending[0], [*pending[1], *indices])
    spans = merged_spans
    passages: list[dict[str, Any]] = []
    for passage_index, (start, end, indices) in enumerate(spans):
        passage_text = text[start:end]
        if not passage_text.strip():
            raise CorpusBuildError(f"CORPUS_EMPTY_PASSAGE: Document {document_id} produced empty text.")
        passages.append(
            {
                "passage_id": passage_id_for(document_id, passage_index),
                "document_id": document_id,
                "dataset": dataset,
                "passage_index": passage_index,
                "text": passage_text,
                "start_char": start,
                "end_char": end,
                "token_count": whitespace_token_count(passage_text),
                "content_hash": sha256_text(passage_text),
                "corpus_version": corpus_version,
                "metadata": {
                    **metadata,
                    "source_type": source_type,
                    "source_sentence_indices": indices,
                    "is_gold_for_any_claim": False,
                },
            }
        )
    return passages


def _resolve_gold_evidence(
    claims: list[dict[str, Any]], passages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    passages_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passages_by_sentence: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for passage in passages:
        passages_by_document[passage["document_id"]].append(passage)
        for index in passage["metadata"]["source_sentence_indices"]:
            key = passage["document_id"], index
            passages_by_sentence[key].append(passage)
    records: list[dict[str, Any]] = []
    unresolved = 0
    seen_claim_ids: set[str] = set()
    for claim in claims:
        claim_id = claim.get("claim_id")
        dataset = claim.get("dataset")
        if not isinstance(claim_id, str) or dataset not in DATASET_ORDER or claim_id in seen_claim_ids:
            raise CorpusBuildError(f"CORPUS_INVALID_CLAIM: Invalid or duplicate claim {claim_id!r}.")
        seen_claim_ids.add(claim_id)
        output_sets: list[dict[str, Any]] = []
        for evidence in claim.get("evidence_sets", []):
            document_id = evidence["document_id"]
            indices = evidence["source_sentence_indices"]
            resolved: list[dict[str, Any]] = []
            if indices:
                for index in indices:
                    matching_passages = passages_by_sentence.get((document_id, index))
                    if not matching_passages:
                        unresolved += 1
                        raise CorpusBuildError(
                            f"CORPUS_UNRESOLVED_EVIDENCE: {evidence['evidence_set_id']} sentence {index} cannot be resolved."
                        )
                    resolved.extend(matching_passages)
            else:
                resolved = passages_by_document.get(document_id, [])
                if not resolved:
                    unresolved += 1
                    raise CorpusBuildError(f"CORPUS_UNRESOLVED_EVIDENCE: {evidence['evidence_set_id']} has no passages.")
            passage_ids = list(dict.fromkeys(item["passage_id"] for item in resolved))
            for passage in resolved:
                passage["metadata"]["is_gold_for_any_claim"] = True
            output_sets.append(
                {
                    "evidence_set_id": evidence["evidence_set_id"],
                    "relationship": evidence["relationship"],
                    "document_id": document_id,
                    "source_sentence_indices": list(indices),
                    "passage_ids": passage_ids,
                }
            )
        records.append(
            {
                "claim_id": claim_id,
                "dataset": dataset,
                "original_split": claim["original_split"],
                "unified_label": claim["unified_label"],
                "evidence_sets": output_sets,
            }
        )
    records.sort(key=lambda row: (DATASET_RANK[row["dataset"]], SPLIT_ORDER.get(row["original_split"], 99), row["original_split"], _id_sort_key(row["claim_id"])))
    return records, unresolved


def build_combined_corpus(
    dataset_dir: Path,
    output_root: Path,
    version: str,
    max_passage_words: int = 120,
) -> Path:
    """Build an immutable multi-dataset medical evidence corpus."""
    validate_version(version)
    if not isinstance(max_passage_words, int) or isinstance(max_passage_words, bool) or max_passage_words < 1:
        raise CorpusBuildError("CORPUS_INVALID_CONFIGURATION: max_passage_words must be at least 1.")
    version_dir = output_root / version
    if version_dir.exists():
        raise CorpusBuildError(f"CORPUS_VERSION_EXISTS: Corpus version {version!r} already exists.")
    for filename in REQUIRED_INPUTS:
        if not (dataset_dir / filename).is_file():
            raise CorpusBuildError(f"CORPUS_MISSING_INPUT: Required input does not exist: {dataset_dir / filename}.")
    claims = _load_jsonl(dataset_dir / "claims.jsonl")
    source_documents = _load_jsonl(dataset_dir / "documents.jsonl")
    evidence_relations = _load_jsonl(dataset_dir / "evidence_relations.jsonl")
    label_schema = _load_json(dataset_dir / "label_schema.json")
    source_manifest = _load_json(dataset_dir / "manifest.json")
    _validate_unified_input(source_manifest, claims, source_documents, evidence_relations, label_schema)
    source_documents.sort(key=lambda row: (DATASET_RANK.get(row.get("dataset"), 99), _id_sort_key(row.get("document_id"))))
    documents: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    for source in source_documents:
        document_id = source.get("document_id")
        if document_id in seen_document_ids:
            raise CorpusBuildError(f"CORPUS_DUPLICATE_DOCUMENT_ID: {document_id} appears more than once.")
        seen_document_ids.add(document_id)
        metadata = dict(source["metadata"])
        metadata["sentence_count"] = len(source["sentences"])
        documents.append(
            {
                "document_id": document_id,
                "dataset": source["dataset"],
                "source_document_id": source["source_document_id"],
                "title": source["title"],
                "source_type": source["source_type"],
                "source_url": source["source_url"],
                "publication_year": source["publication_year"],
                "text": source["text"],
                "content_hash": sha256_text(source["text"]),
                "corpus_version": version,
                "metadata": metadata,
            }
        )
        passages.extend(build_document_passages(source, version, max_passage_words))
    passages.sort(key=lambda row: (DATASET_RANK[row["dataset"]], _id_sort_key(row["document_id"]), row["passage_index"]))
    gold_evidence, unresolved = _resolve_gold_evidence(claims, passages)
    document_counts = Counter(row["dataset"] for row in documents)
    passage_counts = Counter(row["dataset"] for row in passages)
    source_types = Counter(row["source_type"] for row in documents)
    word_counts = [row["token_count"] for row in passages]
    duplicate_counts = Counter(row["content_hash"] for row in passages)
    duplicate_groups = [count for count in duplicate_counts.values() if count > 1]
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in DATASET_ORDER:
        coverage[dataset] = {}
        for label in (*UNIFIED_LABELS, "UNLABELED"):
            matching = [row for row in gold_evidence if row["dataset"] == dataset and (row["unified_label"] or "UNLABELED") == label]
            coverage[dataset][label] = {
                "claims": len(matching),
                "claims_with_evidence": sum(bool(row["evidence_sets"]) for row in matching),
            }
    warnings = []
    if duplicate_groups:
        warnings.append(f"Found {len(duplicate_groups)} exact duplicate passage groups; all passages were retained.")
    quality_report = {
        "corpus_version": version,
        "status": "success",
        "documents_per_dataset": {dataset: document_counts[dataset] for dataset in DATASET_ORDER},
        "passages_per_dataset": {dataset: passage_counts[dataset] for dataset in DATASET_ORDER},
        "source_types": dict(sorted(source_types.items())),
        "passage_word_counts": {
            "minimum": min(word_counts, default=0),
            "maximum": max(word_counts, default=0),
            "average": round(mean(word_counts), 2) if word_counts else 0.0,
        },
        "claims_with_resolved_evidence": sum(bool(row["evidence_sets"]) for row in gold_evidence),
        "claims_without_evidence": sum(not row["evidence_sets"] for row in gold_evidence),
        "unresolved_evidence_references": unresolved,
        "exact_duplicate_passage_groups": len(duplicate_groups),
        "exact_duplicate_passage_count": sum(duplicate_groups),
        "corpus_coverage": coverage,
        "warnings": warnings,
    }
    manifest = {
        "artifact_type": "medical_evidence_corpus",
        "dataset": "multi_dataset",
        "datasets": list(DATASET_ORDER),
        "corpus_version": version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "builder_version": BUILDER_VERSION,
        "source_dataset_version": source_manifest["version"],
        "source_content_hash": source_manifest["content_hash"],
        "configuration": {"chunking_mode": "dataset_appropriate", "max_passage_words": max_passage_words},
        "document_count": len(documents),
        "passage_count": len(passages),
        "claim_count": len(gold_evidence),
        "gold_evidence_set_count": sum(len(row["evidence_sets"]) for row in gold_evidence),
        "content_hash": corpus_content_hash(passages),
        "outputs": {
            "documents": "documents.jsonl",
            "passages": "passages.jsonl",
            "gold_evidence": "gold_evidence.jsonl",
            "quality_report": "quality_report.json",
        },
        "warnings": warnings,
    }
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir()
        _write_jsonl(documents, version_dir / "documents.jsonl")
        _write_jsonl(passages, version_dir / "passages.jsonl")
        _write_jsonl(gold_evidence, version_dir / "gold_evidence.jsonl")
        _write_json(quality_report, version_dir / "quality_report.json")
        _write_json(manifest, version_dir / "manifest.json")
    except FileExistsError as exc:
        raise CorpusBuildError(f"CORPUS_VERSION_EXISTS: Corpus version {version!r} already exists.") from exc
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"CORPUS_OUTPUT_WRITE_ERROR: Could not write {version_dir}: {exc}.") from exc
    return version_dir


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(value: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
