"""Strict, shared I/O primitives for immutable retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from medclaim.corpus.scifact_corpus import corpus_content_hash

CHECKSUM_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric value {value}")


def load_json(path: Path, error_code: str, *, error_type: type[Exception]) -> Any:
    try:
        with path.open(encoding="utf-8") as input_file:
            return json.load(input_file, parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise error_type(
            f"{error_code}: Required file does not exist: {path}."
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        reason = getattr(exc, "msg", str(exc))
        raise error_type(f"{error_code}: Could not parse {path}: {reason}.") from exc
    except OSError as exc:
        raise error_type(f"{error_code}: Could not read {path}: {exc}.") from exc


def load_jsonl_objects(
    path: Path,
    *,
    invalid_code: str,
    missing_code: str,
    error_type: type[Exception],
) -> list[dict[str, Any]]:
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
                    raise error_type(
                        f"{invalid_code}: Could not parse {path} at line "
                        f"{line_number}: {reason}."
                    ) from exc
                if not isinstance(record, dict):
                    raise error_type(
                        f"{invalid_code}: Expected an object in {path} at line {line_number}."
                    )
                records.append(record)
    except FileNotFoundError as exc:
        raise error_type(
            f"{missing_code}: Required file does not exist: {path}."
        ) from exc
    except OSError as exc:
        raise error_type(f"{missing_code}: Could not read {path}: {exc}.") from exc
    return records


def load_corpus(
    corpus_dir: Path, *, prefix: str, error_type: type[Exception]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the corpus contract shared by all retrievers."""
    invalid = f"{prefix}_CORPUS_MANIFEST_INVALID"
    if not corpus_dir.is_dir():
        raise error_type(
            f"{prefix}_CORPUS_NOT_FOUND: Corpus directory does not exist: {corpus_dir}."
        )
    manifest = load_json(corpus_dir / "manifest.json", invalid, error_type=error_type)
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != "medical_evidence_corpus":
        raise error_type(f"{invalid}: Expected a medical evidence corpus.")
    version = manifest.get("corpus_version")
    count = manifest.get("passage_count")
    content_hash = manifest.get("content_hash")
    if not isinstance(version, str) or not version:
        raise error_type(f"{invalid}: corpus_version must be non-empty.")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise error_type(f"{invalid}: passage_count must be positive.")
    if not isinstance(content_hash, str) or CHECKSUM_PATTERN.fullmatch(content_hash) is None:
        raise error_type(f"{invalid}: content_hash must be a SHA-256.")
    passages = load_jsonl_objects(
        corpus_dir / "passages.jsonl",
        invalid_code=invalid,
        missing_code=f"{prefix}_CORPUS_NOT_FOUND",
        error_type=error_type,
    )
    if len(passages) != count:
        raise error_type(f"{prefix}_CORPUS_COUNT_MISMATCH: Passage count does not match manifest.")
    seen_ids: set[str] = set()
    required = ("passage_id", "document_id", "dataset", "text", "corpus_version")
    for passage in passages:
        passage_id = passage.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id:
            raise error_type(f"{invalid}: Passage has no valid passage_id.")
        if passage_id in seen_ids:
            raise error_type(f"{prefix}_DUPLICATE_PASSAGE_ID: Passage ID {passage_id} is duplicated.")
        seen_ids.add(passage_id)
        if any(field not in passage or not isinstance(passage[field], str) for field in required):
            raise error_type(f"{invalid}: Passage {passage_id} has invalid fields.")
        if passage["corpus_version"] != version:
            raise error_type(f"{prefix}_INDEX_CORPUS_MISMATCH: Passage corpus version is invalid.")
    if corpus_content_hash(passages) != content_hash:
        raise error_type(f"{prefix}_CORPUS_HASH_MISMATCH: passages.jsonl does not match its manifest.")
    return manifest, passages


def sha256_file(path: Path, error_code: str, *, error_type: type[Exception]) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise error_type(f"{error_code}: Could not read {path}: {exc}.") from exc
    return f"sha256:{hasher.hexdigest()}"


def write_json(
    value: Any, path: Path, error_code: str, *, error_type: type[Exception]
) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, allow_nan=False, indent=2)
            output_file.write("\n")
    except (OSError, ValueError) as exc:
        raise error_type(f"{error_code}: Could not write {path}: {exc}.") from exc
