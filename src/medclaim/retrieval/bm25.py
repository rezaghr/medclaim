"""Build, validate, and query an immutable rank-bm25 index."""

from __future__ import annotations

import math
import os
import pickle
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .artifact_io import load_corpus, load_json, sha256_file, write_json
from .tokenization import TOKENIZER_NAME, tokenize_bm25

BUILDER_VERSION = "1.0.0"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class BM25Error(Exception):
    """Raised for controlled BM25 artifact or retrieval failures."""


_load_json = partial(load_json, error_type=BM25Error)
_load_corpus = partial(load_corpus, prefix="BM25", error_type=BM25Error)
_sha256_file = partial(
    sha256_file, error_code="BM25_INDEX_NOT_FOUND", error_type=BM25Error
)
_write_json = partial(
    write_json, error_code="BM25_OUTPUT_WRITE_FAILED", error_type=BM25Error
)


def _validate_version(index_version: str) -> None:
    if (
        not isinstance(index_version, str)
        or not index_version
        or index_version in {".", ".."}
        or VERSION_PATTERN.fullmatch(index_version) is None
    ):
        raise BM25Error(
            "BM25_INVALID_VERSION: Version must contain only letters, numbers, "
            "dots, underscores, and hyphens."
        )


def _validate_parameters(k1: float, b: float, epsilon: float) -> None:
    values = (k1, b, epsilon)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        )
        or any(not math.isfinite(float(value)) for value in values)
        or k1 <= 0
        or not 0 <= b <= 1
        or epsilon < 0
    ):
        raise BM25Error(
            "BM25_INVALID_PARAMETER: Require k1 > 0, 0 <= b <= 1, and epsilon >= 0."
        )


def _rank_bm25_version() -> str:
    try:
        return package_version("rank-bm25")
    except PackageNotFoundError:
        return "unknown"


def build_bm25_index(
    corpus_dir: Path,
    output_root: Path,
    version: str,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> Path:
    """Build an immutable BM25 index for one validated corpus."""
    _validate_version(version)
    _validate_parameters(k1, b, epsilon)
    index_dir = output_root / version
    if index_dir.exists():
        raise BM25Error(
            "BM25_INDEX_VERSION_EXISTS: "
            f"Index version {version!r} already exists. Use a new index version."
        )

    corpus_manifest, passages = _load_corpus(corpus_dir)
    tokenized_corpus: list[list[str]] = []
    passage_ids: list[str] = []
    for passage in passages:
        tokens = tokenize_bm25(passage["text"])
        if not tokens:
            raise BM25Error(
                "BM25_EMPTY_TOKENIZED_PASSAGE: "
                f"Passage {passage['passage_id']} contains no indexable tokens."
            )
        tokenized_corpus.append(tokens)
        passage_ids.append(passage["passage_id"])
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b, epsilon=epsilon)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root)
        )
    except OSError as exc:
        raise BM25Error(
            f"BM25_OUTPUT_WRITE_FAILED: Could not create index build directory: {exc}."
        ) from exc

    try:
        index_path = temporary_dir / "index.pkl"
        mapping_path = temporary_dir / "passage_ids.json"
        try:
            with index_path.open("wb") as output_file:
                pickle.dump(bm25, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise BM25Error(
                f"BM25_OUTPUT_WRITE_FAILED: Could not write {index_path}: {exc}."
            ) from exc
        _write_json(passage_ids, mapping_path)

        manifest = {
            "artifact_type": "bm25_index",
            "retrieval_type": "sparse",
            "index_version": version,
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "builder_version": BUILDER_VERSION,
            "library": {
                "name": "rank-bm25",
                "version": _rank_bm25_version(),
                "implementation": "BM25Okapi",
            },
            "corpus": {
                "version": corpus_manifest["corpus_version"],
                "content_hash": corpus_manifest["content_hash"],
                "passage_count": len(passages),
            },
            "configuration": {
                "indexed_fields": ["text"],
                "tokenizer": TOKENIZER_NAME,
                "unicode_normalization": "NFKC",
                "lowercase": True,
                "remove_stop_words": False,
                "stemming": False,
                "k1": float(k1),
                "b": float(b),
                "epsilon": float(epsilon),
            },
            "files": {
                "index": {"path": "index.pkl", "sha256": _sha256_file(index_path)},
                "passage_ids": {
                    "path": "passage_ids.json",
                    "sha256": _sha256_file(mapping_path),
                },
            },
        }
        _write_json(manifest, temporary_dir / "manifest.json")
        try:
            temporary_dir.chmod(0o755)
            os.rename(temporary_dir, index_dir)
        except FileExistsError as exc:
            raise BM25Error(
                "BM25_INDEX_VERSION_EXISTS: "
                f"Index version {version!r} already exists. Use a new index version."
            ) from exc
        except OSError as exc:
            raise BM25Error(
                f"BM25_OUTPUT_WRITE_FAILED: Could not finalize {index_dir}: {exc}."
            ) from exc
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return index_dir


def _validate_index_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != "bm25_index":
        raise BM25Error("BM25_INDEX_MANIFEST_INVALID: Expected a BM25 index manifest.")
    if manifest.get("retrieval_type") != "sparse":
        raise BM25Error("BM25_INDEX_MANIFEST_INVALID: retrieval_type must be 'sparse'.")
    for field in ("index_version", "corpus", "configuration", "files"):
        if field not in manifest:
            raise BM25Error(
                f"BM25_INDEX_MANIFEST_INVALID: Manifest is missing {field!r}."
            )
    if not isinstance(manifest["index_version"], str) or not manifest["index_version"]:
        raise BM25Error(
            "BM25_INDEX_MANIFEST_INVALID: index_version must be a non-empty string."
        )
    corpus = manifest["corpus"]
    configuration = manifest["configuration"]
    files = manifest["files"]
    if (
        not isinstance(corpus, dict)
        or not isinstance(configuration, dict)
        or not isinstance(files, dict)
    ):
        raise BM25Error("BM25_INDEX_MANIFEST_INVALID: Invalid manifest structure.")
    for field in ("version", "content_hash", "passage_count"):
        if field not in corpus:
            raise BM25Error(
                f"BM25_INDEX_MANIFEST_INVALID: Corpus section is missing {field!r}."
            )
    for key in ("index", "passage_ids"):
        if not isinstance(files.get(key), dict):
            raise BM25Error(
                f"BM25_INDEX_MANIFEST_INVALID: Files section is missing {key!r}."
            )
        if set(("path", "sha256")) - set(files[key]):
            raise BM25Error(
                f"BM25_INDEX_MANIFEST_INVALID: Invalid file entry for {key}."
            )
    expected_paths = {"index": "index.pkl", "passage_ids": "passage_ids.json"}
    for key, expected_path in expected_paths.items():
        if (
            files[key]["path"] != expected_path
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(files[key]["sha256"])) is None
        ):
            raise BM25Error(
                f"BM25_INDEX_MANIFEST_INVALID: Invalid path or checksum for {key}."
            )
    expected_configuration = {
        "indexed_fields": ["text"],
        "tokenizer": TOKENIZER_NAME,
        "unicode_normalization": "NFKC",
        "lowercase": True,
        "remove_stop_words": False,
        "stemming": False,
    }
    if any(
        configuration.get(key) != expected
        for key, expected in expected_configuration.items()
    ):
        raise BM25Error(
            "BM25_INDEX_MANIFEST_INVALID: Index tokenization configuration is "
            "not supported by this retriever."
        )
    library = manifest.get("library")
    if (
        not isinstance(library, dict)
        or library.get("name") != "rank-bm25"
        or library.get("implementation") != "BM25Okapi"
    ):
        raise BM25Error(
            "BM25_INDEX_MANIFEST_INVALID: Index library configuration is invalid."
        )
    return manifest


class BM25Retriever:
    """A validated BM25 index loaded once and reusable across searches."""

    def __init__(
        self,
        index_dir: Path,
        corpus_dir: Path,
        *,
        corpus_data: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
    ) -> None:
        if not index_dir.is_dir():
            raise BM25Error(
                f"BM25_INDEX_NOT_FOUND: Index directory does not exist: {index_dir}."
            )
        self.index_dir = index_dir
        self.corpus_dir = corpus_dir
        self.index_manifest = _validate_index_manifest(
            _load_json(index_dir / "manifest.json", "BM25_INDEX_MANIFEST_INVALID")
        )
        self.corpus_manifest, self.passages = corpus_data or _load_corpus(corpus_dir)
        self._validate_compatibility()
        self.passage_ids = self._load_mapping_and_validate_checksums()
        self.passages_by_id = {
            passage["passage_id"]: passage for passage in self.passages
        }
        corpus_ids = [passage["passage_id"] for passage in self.passages]
        if self.passage_ids != corpus_ids or len(set(self.passage_ids)) != len(
            corpus_ids
        ):
            raise BM25Error(
                "BM25_PASSAGE_MAPPING_MISMATCH: passage_ids.json does not align "
                "with the ordered corpus passages."
            )
        self.bm25 = self._load_index()
        if getattr(self.bm25, "corpus_size", None) != len(self.passage_ids):
            raise BM25Error(
                "BM25_PASSAGE_MAPPING_MISMATCH: Serialized index size does not "
                "match passage_ids.json."
            )

    def _validate_compatibility(self) -> None:
        indexed_corpus = self.index_manifest["corpus"]
        if (
            indexed_corpus["version"] != self.corpus_manifest["corpus_version"]
            or indexed_corpus["content_hash"] != self.corpus_manifest["content_hash"]
            or indexed_corpus["passage_count"] != len(self.passages)
        ):
            raise BM25Error(
                "BM25_INDEX_CORPUS_MISMATCH: Index and corpus version, content "
                "hash, or passage count do not match."
            )

    def _load_mapping_and_validate_checksums(self) -> list[str]:
        files = self.index_manifest["files"]
        index_path = self.index_dir / files["index"]["path"]
        mapping_path = self.index_dir / files["passage_ids"]["path"]
        for path, expected in (
            (index_path, files["index"]["sha256"]),
            (mapping_path, files["passage_ids"]["sha256"]),
        ):
            if _sha256_file(path) != expected:
                raise BM25Error(
                    "BM25_INDEX_CHECKSUM_MISMATCH: "
                    f"{path.name} does not match the checksum recorded in manifest.json."
                )
        mapping = _load_json(mapping_path, "BM25_PASSAGE_MAPPING_MISMATCH")
        if not isinstance(mapping, list) or any(
            not isinstance(passage_id, str) for passage_id in mapping
        ):
            raise BM25Error(
                "BM25_PASSAGE_MAPPING_MISMATCH: passage_ids.json must be a list of strings."
            )
        return mapping

    def _load_index(self) -> BM25Okapi:
        index_entry = self.index_manifest["files"]["index"]
        index_path = self.index_dir / index_entry["path"]
        try:
            with index_path.open("rb") as input_file:
                bm25 = pickle.load(input_file)
        except Exception as exc:
            raise BM25Error(
                f"BM25_INDEX_LOAD_FAILED: Could not load {index_path.name}: {exc}."
            ) from exc
        if not isinstance(bm25, BM25Okapi):
            raise BM25Error(
                "BM25_INDEX_LOAD_FAILED: Serialized object is not a BM25Okapi index."
            )
        return bm25

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        if not isinstance(query, str):
            raise BM25Error(
                "BM25_EMPTY_QUERY: The search query must be a string containing "
                "at least one indexable word or number."
            )
        trimmed_query = query.strip()
        query_tokens = tokenize_bm25(trimmed_query)
        if not trimmed_query or not query_tokens:
            raise BM25Error(
                "BM25_EMPTY_QUERY: The search query must contain at least one "
                "indexable word or number."
            )
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 100
        ):
            raise BM25Error("BM25_INVALID_TOP_K: top_k must be between 1 and 100.")

        started = time.perf_counter()
        raw_scores = self.bm25.get_scores(query_tokens)
        candidates: list[tuple[float, str]] = []
        for passage_id, raw_score in zip(self.passage_ids, raw_scores, strict=True):
            score = float(raw_score)
            if not math.isfinite(score):
                raise BM25Error(
                    "BM25_INDEX_LOAD_FAILED: Index produced a non-finite score."
                )
            candidates.append((score, passage_id))
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))[
            : min(top_k, len(candidates))
        ]
        results = []
        for rank, (score, passage_id) in enumerate(ranked, start=1):
            passage = self.passages_by_id[passage_id]
            results.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "document_id": passage["document_id"],
                    "dataset": passage["dataset"],
                    "text": passage["text"],
                    "bm25_score": score,
                }
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "query": trimmed_query,
            "normalized_query_tokens": query_tokens,
            "retrieval_mode": "bm25",
            "top_k": top_k,
            "returned_count": len(results),
            "latency_ms": latency_ms,
            "corpus_version": self.corpus_manifest["corpus_version"],
            "index_version": self.index_manifest["index_version"],
            "results": results,
        }
