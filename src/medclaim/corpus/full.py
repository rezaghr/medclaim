"""Download and normalize the complete SciFact, HealthVer, and PUBHEALTH releases."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .scifact_corpus import corpus_content_hash, sha256_text, whitespace_token_count

DATASET_ORDER = ("scifact", "healthver", "pubhealth")
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}
LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")


class FullCorpusError(Exception):
    """Raised when an official source cannot be downloaded or normalized safely."""


@dataclass(frozen=True)
class SourceFile:
    filename: str
    url: str
    sha256: str
    dataset: str
    license: str


SOURCE_FILES = (
    SourceFile(
        "scifact-data.tar.gz",
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
        "scifact",
        "CC-BY-4.0 claims; ODC-By-1.0 S2ORC abstracts",
    ),
    SourceFile(
        "healthver-train.csv",
        "https://raw.githubusercontent.com/sarrouti/healthver/"
        "b20ac99ceed62f5264a319fa25a854df1668d85b/data/healthver_train.csv",
        "57668949f79d0f46e66d74df4b0bc9812cad4d5ad7c5dd03dfba008a608b3f2a",
        "healthver",
        "Research release; repository does not declare an SPDX license",
    ),
    SourceFile(
        "healthver-dev.csv",
        "https://raw.githubusercontent.com/sarrouti/healthver/"
        "b20ac99ceed62f5264a319fa25a854df1668d85b/data/healthver_dev.csv",
        "6f8ea31cbcf2ea5a2d3c82d65587d66e96de4f70e72642bbf7f61fec6ef8a93a",
        "healthver",
        "Research release; repository does not declare an SPDX license",
    ),
    SourceFile(
        "healthver-test.csv",
        "https://raw.githubusercontent.com/sarrouti/healthver/"
        "b20ac99ceed62f5264a319fa25a854df1668d85b/data/healthver_test.csv",
        "cf457d2dcddd7a736408082d0037527884f7ca3bda06729e1da5ec9e3b135819",
        "healthver",
        "Research release; repository does not declare an SPDX license",
    ),
    SourceFile(
        "pubhealth.zip",
        "https://drive.google.com/uc?export=download&id=1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj",
        "3f0a5541f4a60c09a138a896621402893ce4b3a37060363d9257010c2c27fc3a",
        "pubhealth",
        "MIT repository release",
    ),
)

SCIFACT_LABELS = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}
HEALTHVER_LABELS = {
    "Supports": "SUPPORTS",
    "Refutes": "REFUTES",
    "Neutral": "NOT_ENOUGH_INFO",
}
PUBHEALTH_LABELS = {
    "true": "SUPPORTS",
    "false": "REFUTES",
    "unproven": "NOT_ENOUGH_INFO",
    "mixture": "MIXED",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", without_controls).strip()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_sources(download_dir: Path) -> dict[str, Path]:
    """Download immutable, checksum-pinned source files with atomic publication."""
    download_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for source in SOURCE_FILES:
        destination = download_dir / source.filename
        if destination.is_file() and _sha256_file(destination) == source.sha256:
            resolved[source.filename] = destination
            continue
        if destination.exists():
            destination.unlink()
        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    source.url, headers={"User-Agent": "MedClaimRAG/1.0 dataset downloader"}
                )
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                actual = _sha256_file(temporary)
                if actual != source.sha256:
                    raise FullCorpusError(
                        f"DATASET_CHECKSUM_MISMATCH: {source.filename} expected "
                        f"{source.sha256}, received {actual}."
                    )
                os.replace(temporary, destination)
                resolved[source.filename] = destination
                break
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(2**attempt)
        else:
            raise FullCorpusError(
                f"DATASET_DOWNLOAD_FAILED: Could not download {source.url}: {last_error}."
            ) from last_error
    return resolved


def _json_lines(handle: Iterable[bytes]) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullCorpusError(
                f"DATASET_INVALID_JSON: Invalid JSONL record at line {line_number}: {exc}."
            ) from exc
        if not isinstance(row, dict):
            raise FullCorpusError(
                f"DATASET_INVALID_JSON: JSONL line {line_number} is not an object."
            )
        rows.append(row)
    return rows


def _document(
    dataset: str,
    source_id: str,
    text: str,
    *,
    title: str | None,
    source_type: str,
    sentences: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "document_id": f"{dataset}:document:{source_id}",
        "dataset": dataset,
        "source_document_id": source_id,
        "title": title,
        "source_type": source_type,
        "source_url": None,
        "publication_year": None,
        "text": text,
        "sentences": sentences,
        "metadata": metadata,
    }


def _claim(
    dataset: str,
    source_id: str,
    text: str,
    split: str,
    original_label: str | None,
    unified_label: str | None,
    evidence_sets: list[dict[str, Any]],
    *,
    explanation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": f"{dataset}:claim:{source_id}",
        "dataset": dataset,
        "source_claim_id": source_id,
        "claim_text": text,
        "original_split": split,
        "original_label": original_label,
        "unified_label": unified_label,
        "language": "en",
        "evidence_sets": evidence_sets,
        "gold_explanation": explanation,
        "metadata": metadata or {},
    }


def _load_scifact(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with tarfile.open(path) as archive:
        corpus_handle = archive.extractfile("data/corpus.jsonl")
        if corpus_handle is None:
            raise FullCorpusError("SCIFACT_MISSING_INPUT: data/corpus.jsonl is absent.")
        source_documents = _json_lines(corpus_handle)
        split_rows: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "dev", "test"):
            handle = archive.extractfile(f"data/claims_{split}.jsonl")
            if handle is None:
                raise FullCorpusError(f"SCIFACT_MISSING_INPUT: claims_{split}.jsonl is absent.")
            split_rows[split] = _json_lines(handle)

    documents = []
    sentences_by_document: dict[str, list[str]] = {}
    for row in sorted(source_documents, key=lambda item: int(item["doc_id"])):
        source_id = str(row["doc_id"])
        sentences = [_clean_text(item) for item in row.get("abstract", [])]
        sentences = [item for item in sentences if item]
        if not sentences:
            raise FullCorpusError(f"SCIFACT_EMPTY_DOCUMENT: Document {source_id} is empty.")
        document = _document(
            "scifact",
            source_id,
            " ".join(sentences),
            title=_clean_text(row.get("title")) or None,
            source_type="scientific_abstract",
            sentences=sentences,
            metadata={"structured": bool(row.get("structured", False))},
        )
        documents.append(document)
        sentences_by_document[document["document_id"]] = sentences

    claims = []
    seen_claim_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        for row in sorted(split_rows[split], key=lambda item: int(item["id"])):
            source_id = str(row["id"])
            if source_id in seen_claim_ids:
                raise FullCorpusError(f"SCIFACT_DUPLICATE_CLAIM: Claim {source_id} is duplicated.")
            seen_claim_ids.add(source_id)
            claim_id = f"scifact:claim:{source_id}"
            evidence_sets = []
            source_labels: list[str] = []
            for document_source_id in sorted(row.get("evidence", {}), key=int):
                groups = row["evidence"][document_source_id]
                document_id = f"scifact:document:{document_source_id}"
                if document_id not in sentences_by_document:
                    raise FullCorpusError(
                        f"SCIFACT_MISSING_DOCUMENT: Claim {source_id} references {document_id}."
                    )
                for group in groups:
                    source_label = group.get("label")
                    if source_label not in SCIFACT_LABELS:
                        raise FullCorpusError(
                            f"SCIFACT_UNKNOWN_LABEL: Claim {source_id} uses {source_label!r}."
                        )
                    indices = group.get("sentences")
                    if not isinstance(indices, list) or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or not 0 <= index < len(sentences_by_document[document_id])
                        for index in indices
                    ):
                        raise FullCorpusError(
                            f"SCIFACT_INVALID_EVIDENCE: Claim {source_id} has invalid sentences."
                        )
                    source_labels.append(source_label)
                    evidence_sets.append(
                        {
                            "evidence_set_id": f"{claim_id}:evidence:{len(evidence_sets)}",
                            "relationship": SCIFACT_LABELS[source_label],
                            "document_id": document_id,
                            "source_sentence_indices": list(indices),
                        }
                    )
            unique_labels = set(source_labels)
            original_label = next(iter(unique_labels)) if len(unique_labels) == 1 else None
            unified_label = (
                SCIFACT_LABELS[original_label]
                if original_label is not None
                else "MIXED"
                if len(unique_labels) > 1
                else "NOT_ENOUGH_INFO"
                if split != "test"
                else None
            )
            claims.append(
                _claim(
                    "scifact",
                    source_id,
                    _clean_text(row.get("claim")),
                    split,
                    original_label,
                    unified_label,
                    evidence_sets,
                    metadata={
                        "cited_document_ids": [str(item) for item in row.get("cited_doc_ids", [])]
                    },
                )
            )
    return documents, claims


def _load_healthver(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents_by_hash: dict[str, dict[str, Any]] = {}
    claims = []
    seen_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        with paths[split].open(encoding="utf-8-sig", newline="") as source:
            rows = csv.DictReader(source)
            for row in rows:
                source_id = str(row.get("id", "")).strip()
                if not source_id or source_id in seen_ids:
                    raise FullCorpusError(
                        f"HEALTHVER_INVALID_ID: Missing or duplicate row ID {source_id!r}."
                    )
                seen_ids.add(source_id)
                evidence = _clean_text(row.get("evidence"))
                claim_text = _clean_text(row.get("claim"))
                source_label = _clean_text(row.get("label"))
                if not evidence or not claim_text or source_label not in HEALTHVER_LABELS:
                    raise FullCorpusError(f"HEALTHVER_INVALID_ROW: Row {source_id} is incomplete.")
                evidence_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
                document_source_id = evidence_hash[:24]
                document_id = f"healthver:document:{document_source_id}"
                existing = documents_by_hash.get(evidence_hash)
                if existing is None:
                    documents_by_hash[evidence_hash] = _document(
                        "healthver",
                        document_source_id,
                        evidence,
                        title=None,
                        source_type="scientific_evidence_statement",
                        sentences=[evidence],
                        metadata={"evidence_sha256": evidence_hash},
                    )
                elif existing["text"] != evidence:
                    raise FullCorpusError("HEALTHVER_HASH_COLLISION: Evidence hashes collided.")
                claim_id = f"healthver:claim:{source_id}"
                relationship = HEALTHVER_LABELS[source_label]
                claims.append(
                    _claim(
                        "healthver",
                        source_id,
                        claim_text,
                        split,
                        source_label,
                        relationship,
                        [
                            {
                                "evidence_set_id": f"{claim_id}:evidence:0",
                                "relationship": relationship,
                                "document_id": document_id,
                                "source_sentence_indices": [0],
                            }
                        ],
                        metadata={
                            "topic_id": _clean_text(row.get("topic_ip")),
                            "question": _clean_text(row.get("question")),
                        },
                    )
                )
    documents = sorted(documents_by_hash.values(), key=lambda item: item["source_document_id"])
    claims.sort(key=lambda item: (SPLIT_ORDER[item["original_split"]], int(item["source_claim_id"])))
    return documents, claims


def _load_pubhealth(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    documents = []
    claims = []
    dropped = Counter()
    with zipfile.ZipFile(path) as archive:
        for split in ("train", "dev", "test"):
            member = f"PUBHEALTH/{split}.tsv"
            with archive.open(member) as binary:
                source = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                rows = csv.reader(source, delimiter="\t")
                header = next(rows, [])
                for row_number, row in enumerate(rows, start=2):
                    if len(row) == 10 and header and header[0] == "":
                        row = row[1:]
                    if len(row) != 9:
                        dropped["malformed_rows"] += 1
                        continue
                    (
                        raw_id,
                        raw_claim,
                        date,
                        explanation,
                        fact_checkers,
                        main_text,
                        sources,
                        raw_label,
                        subjects,
                    ) = row
                    claim_text = _clean_text(raw_claim)
                    text = _clean_text(main_text)
                    if not claim_text or not text:
                        dropped["empty_rows"] += 1
                        continue
                    source_id = _clean_text(raw_id) or f"row-{row_number}"
                    namespaced_id = f"{split}:{source_id}"
                    source_label = _clean_text(raw_label).casefold() or None
                    unified_label = PUBHEALTH_LABELS.get(source_label)
                    if unified_label is None:
                        dropped["unlabeled_rows_retained"] += 1
                    document = _document(
                        "pubhealth",
                        namespaced_id,
                        text,
                        title=None,
                        source_type="fact_check_article",
                        sentences=[],
                        metadata={
                            "date_published": _clean_text(date),
                            "evidence_sources": _clean_text(sources),
                            "subjects": _clean_text(subjects),
                        },
                    )
                    documents.append(document)
                    claim_id = f"pubhealth:claim:{namespaced_id}"
                    evidence_sets = (
                        [
                            {
                                "evidence_set_id": f"{claim_id}:evidence:0",
                                "relationship": unified_label,
                                "document_id": document["document_id"],
                                "source_sentence_indices": [],
                            }
                        ]
                        if unified_label is not None
                        else []
                    )
                    claims.append(
                        _claim(
                            "pubhealth",
                            namespaced_id,
                            claim_text,
                            split,
                            source_label,
                            unified_label,
                            evidence_sets,
                            explanation=_clean_text(explanation) or None,
                            metadata={"fact_checkers": _clean_text(fact_checkers)},
                        )
                    )
    return documents, claims, dict(dropped)


def _source_spans(document: dict[str, Any]) -> list[tuple[int, int, list[int]]]:
    text = document["text"]
    sentences = document["sentences"]
    if sentences:
        cursor = 0
        spans = []
        for index, sentence in enumerate(sentences):
            start = text.find(sentence, cursor)
            if start < 0:
                raise FullCorpusError(
                    f"CORPUS_DOCUMENT_TEXT_MISMATCH: Cannot locate sentence {index} in "
                    f"{document['document_id']}."
                )
            end = start + len(sentence)
            spans.append((start, end, [index]))
            cursor = end
        return spans
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r"(?<=[.!?])\s+", text))
    boundaries.append(len(text))
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end, []))
    return spans or [(0, len(text), [])]


def _chunk_document(
    document: dict[str, Any], max_words: int
) -> list[tuple[int, int, list[int]]]:
    pieces: list[tuple[int, int, list[int]]] = []
    for start, end, indices in _source_spans(document):
        words = list(re.finditer(r"\S+", document["text"][start:end]))
        for offset in range(0, len(words), max_words):
            group = words[offset : offset + max_words]
            if group:
                pieces.append(
                    (start + group[0].start(), start + group[-1].end(), list(indices))
                )
    chunks: list[tuple[int, int, list[int]]] = []
    for start, end, indices in pieces:
        word_count = whitespace_token_count(document["text"][start:end])
        if chunks:
            prior_start, prior_end, prior_indices = chunks[-1]
            combined_count = whitespace_token_count(document["text"][prior_start:end])
            if combined_count <= max_words and word_count <= max_words:
                chunks[-1] = (
                    prior_start,
                    end,
                    list(dict.fromkeys([*prior_indices, *indices])),
                )
                continue
        chunks.append((start, end, indices))
    return chunks


def _build_passages(
    documents: list[dict[str, Any]], corpus_version: str, max_words: int
) -> tuple[list[dict[str, Any]], int]:
    passages = []
    skipped_nonsemantic = 0
    for document in documents:
        for passage_index, (start, end, sentence_indices) in enumerate(
            _chunk_document(document, max_words)
        ):
            text = document["text"][start:end]
            if not any(character.isalnum() for character in text):
                skipped_nonsemantic += 1
                continue
            passages.append(
                {
                    "passage_id": f"p{len(passages) + 1:07d}",
                    "document_id": document["document_id"],
                    "dataset": document["dataset"],
                    "passage_index": passage_index,
                    "text": text,
                    "start_char": start,
                    "end_char": end,
                    "token_count": whitespace_token_count(text),
                    "content_hash": sha256_text(text),
                    "corpus_version": corpus_version,
                    "metadata": {
                        "source_type": document["source_type"],
                        "source_sentence_indices": sentence_indices,
                        "is_gold_for_any_claim": False,
                    },
                }
            )
    return passages, skipped_nonsemantic


def _resolve_evidence(
    claims: list[dict[str, Any]], passages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sentence: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for passage in passages:
        by_document[passage["document_id"]].append(passage)
        for sentence_index in passage["metadata"]["source_sentence_indices"]:
            by_sentence[(passage["document_id"], sentence_index)].append(passage)
    gold = []
    for claim in claims:
        output_sets = []
        for evidence in claim["evidence_sets"]:
            indices = evidence["source_sentence_indices"]
            selected = (
                [
                    passage
                    for index in indices
                    for passage in by_sentence.get((evidence["document_id"], index), [])
                ]
                if indices
                else by_document.get(evidence["document_id"], [])
            )
            selected = list({item["passage_id"]: item for item in selected}.values())
            if not selected:
                raise FullCorpusError(
                    f"CORPUS_UNRESOLVED_EVIDENCE: {evidence['evidence_set_id']} has no passage."
                )
            for passage in selected:
                passage["metadata"]["is_gold_for_any_claim"] = True
            evidence["passage_ids"] = [item["passage_id"] for item in selected]
            output_sets.append(dict(evidence))
        gold.append(
            {
                "claim_id": claim["claim_id"],
                "dataset": claim["dataset"],
                "original_split": claim["original_split"],
                "unified_label": claim["unified_label"],
                "evidence_sets": output_sets,
            }
        )
    return gold


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_full_corpus(
    sources: dict[str, Path], output_root: Path, version: str, max_passage_words: int = 120
) -> Path:
    """Normalize all valid official records into one immutable retrieval corpus."""
    version_dir = output_root / version
    if version_dir.is_dir():
        return version_dir
    if not isinstance(max_passage_words, int) or not 1 <= max_passage_words <= 1000:
        raise FullCorpusError("CORPUS_INVALID_CONFIGURATION: max_passage_words is invalid.")

    scifact_documents, scifact_claims = _load_scifact(sources["scifact-data.tar.gz"])
    healthver_documents, healthver_claims = _load_healthver(
        {
            split: sources[f"healthver-{split}.csv"]
            for split in ("train", "dev", "test")
        }
    )
    pubhealth_documents, pubhealth_claims, pubhealth_dropped = _load_pubhealth(
        sources["pubhealth.zip"]
    )
    documents = [*scifact_documents, *healthver_documents, *pubhealth_documents]
    claims = [*scifact_claims, *healthver_claims, *pubhealth_claims]
    if len({item["document_id"] for item in documents}) != len(documents):
        raise FullCorpusError("CORPUS_DUPLICATE_DOCUMENT_ID: Document IDs are not unique.")
    if len({item["claim_id"] for item in claims}) != len(claims):
        raise FullCorpusError("CORPUS_DUPLICATE_CLAIM_ID: Claim IDs are not unique.")

    for document in documents:
        document["content_hash"] = sha256_text(document["text"])
        document["corpus_version"] = version
    passages, skipped_nonsemantic = _build_passages(
        documents, version, max_passage_words
    )
    gold_evidence = _resolve_evidence(claims, passages)
    document_counts = Counter(item["dataset"] for item in documents)
    passage_counts = Counter(item["dataset"] for item in passages)
    claim_counts = Counter(item["dataset"] for item in claims)
    label_counts = Counter(item["unified_label"] or "UNLABELED" for item in claims)
    source_metadata = [
        {
            "dataset": source.dataset,
            "filename": source.filename,
            "url": source.url,
            "sha256": f"sha256:{source.sha256}",
            "license": source.license,
        }
        for source in SOURCE_FILES
    ]
    warnings = []
    if pubhealth_dropped:
        warnings.append(
            "PUBHEALTH malformed or empty source rows were excluded; counts are in quality_report.json."
        )
    quality_report = {
        "status": "success",
        "corpus_version": version,
        "documents_per_dataset": dict(document_counts),
        "passages_per_dataset": dict(passage_counts),
        "claims_per_dataset": dict(claim_counts),
        "label_distribution": dict(label_counts),
        "claims_with_evidence": sum(bool(item["evidence_sets"]) for item in claims),
        "gold_evidence_set_count": sum(len(item["evidence_sets"]) for item in claims),
        "pubhealth_source_rows": {
            "retained": len(pubhealth_claims),
            **pubhealth_dropped,
        },
        "nonsemantic_source_fragments_excluded_from_passages": skipped_nonsemantic,
        "warnings": warnings,
    }
    manifest = {
        "artifact_type": "medical_evidence_corpus",
        "dataset": "multi_dataset",
        "datasets": list(DATASET_ORDER),
        "corpus_version": version,
        "created_at": _now(),
        "builder_version": "full-corpus-1.0.0",
        "source_dataset_version": "scifact-latest+healthver-b20ac99+pubhealth-official",
        "sources": source_metadata,
        "configuration": {
            "chunking_mode": "sentence_aware_max_words",
            "max_passage_words": max_passage_words,
            "healthver_exact_evidence_deduplication": True,
        },
        "document_count": len(documents),
        "passage_count": len(passages),
        "claim_count": len(claims),
        "gold_evidence_set_count": sum(len(item["evidence_sets"]) for item in claims),
        "content_hash": corpus_content_hash(passages),
        "outputs": {
            "documents": "documents.jsonl",
            "passages": "passages.jsonl",
            "claims": "claims.jsonl",
            "gold_evidence": "gold_evidence.jsonl",
            "quality_report": "quality_report.json",
        },
        "warnings": warnings,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root))
    try:
        _write_jsonl(temporary / "documents.jsonl", documents)
        _write_jsonl(temporary / "passages.jsonl", passages)
        _write_jsonl(temporary / "claims.jsonl", claims)
        _write_jsonl(temporary / "gold_evidence.jsonl", gold_evidence)
        _write_json(temporary / "quality_report.json", quality_report)
        _write_json(temporary / "manifest.json", manifest)
        temporary.chmod(0o755)
        os.rename(temporary, version_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return version_dir
