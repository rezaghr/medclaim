"""Build, validate, and query immutable FAISS dense indexes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from medclaim.corpus.scifact_corpus import corpus_content_hash

from .embedding import (
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    EmbeddingError,
    SentenceTransformerEmbedder,
    normalize_claim_input,
    normalized_float32_matrix,
    resolve_device,
)

BUILDER_VERSION = "1.0.0"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
CHECKSUM_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class DenseError(Exception):
    """Raised for controlled dense artifact or retrieval failures."""


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise DenseError(
            "DENSE_DEPENDENCY_MISSING: Install faiss-cpu for dense retrieval."
        ) from exc
    return faiss


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric value {value}")


def _load_json(path: Path, error_code: str) -> Any:
    try:
        with path.open(encoding="utf-8") as input_file:
            return json.load(input_file, parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise DenseError(f"{error_code}: Required file does not exist: {path}.") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        reason = getattr(exc, "msg", str(exc))
        raise DenseError(f"{error_code}: Could not parse {path}: {reason}.") from exc
    except OSError as exc:
        raise DenseError(f"{error_code}: Could not read {path}: {exc}.") from exc


def _load_passages(path: Path) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    passage = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    reason = getattr(exc, "msg", str(exc))
                    raise DenseError(
                        "DENSE_CORPUS_MANIFEST_INVALID: "
                        f"Could not parse {path} at line {line_number}: {reason}."
                    ) from exc
                if not isinstance(passage, dict):
                    raise DenseError(
                        "DENSE_CORPUS_MANIFEST_INVALID: "
                        f"Expected an object in {path} at line {line_number}."
                    )
                passages.append(passage)
    except FileNotFoundError as exc:
        raise DenseError(
            f"DENSE_CORPUS_NOT_FOUND: Required file does not exist: {path}."
        ) from exc
    except OSError as exc:
        raise DenseError(f"DENSE_CORPUS_NOT_FOUND: Could not read {path}: {exc}.") from exc
    return passages


def _load_and_validate_corpus(
    corpus_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not corpus_dir.is_dir():
        raise DenseError(
            f"DENSE_CORPUS_NOT_FOUND: Corpus directory does not exist: {corpus_dir}."
        )
    manifest = _load_json(
        corpus_dir / "manifest.json", "DENSE_CORPUS_MANIFEST_INVALID"
    )
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != "medical_evidence_corpus":
        raise DenseError(
            "DENSE_CORPUS_MANIFEST_INVALID: Expected a medical evidence corpus."
        )
    corpus_version = manifest.get("corpus_version")
    passage_count = manifest.get("passage_count")
    content_hash = manifest.get("content_hash")
    if not isinstance(corpus_version, str) or not corpus_version:
        raise DenseError(
            "DENSE_CORPUS_MANIFEST_INVALID: corpus_version must be non-empty."
        )
    if (
        not isinstance(passage_count, int)
        or isinstance(passage_count, bool)
        or passage_count < 1
    ):
        raise DenseError(
            "DENSE_CORPUS_MANIFEST_INVALID: passage_count must be positive."
        )
    if not isinstance(content_hash, str) or CHECKSUM_PATTERN.fullmatch(content_hash) is None:
        raise DenseError(
            "DENSE_CORPUS_MANIFEST_INVALID: content_hash must be a SHA-256."
        )

    passages = _load_passages(corpus_dir / "passages.jsonl")
    if len(passages) != passage_count:
        raise DenseError(
            "DENSE_CORPUS_COUNT_MISMATCH: Corpus passage count does not match manifest."
        )
    seen_ids: set[str] = set()
    required_fields = ("passage_id", "document_id", "dataset", "text", "corpus_version")
    for passage in passages:
        passage_id = passage.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id:
            raise DenseError(
                "DENSE_CORPUS_MANIFEST_INVALID: Passage has no valid passage_id."
            )
        if passage_id in seen_ids:
            raise DenseError(
                f"DENSE_DUPLICATE_PASSAGE_ID: Passage ID {passage_id} is duplicated."
            )
        seen_ids.add(passage_id)
        if any(field not in passage or not isinstance(passage[field], str) for field in required_fields):
            raise DenseError(
                f"DENSE_CORPUS_MANIFEST_INVALID: Passage {passage_id} has invalid fields."
            )
        if not passage["text"].strip():
            raise DenseError(
                f"DENSE_EMPTY_PASSAGE: Passage {passage_id} contains empty text."
            )
        if passage["corpus_version"] != corpus_version:
            raise DenseError(
                "DENSE_INDEX_CORPUS_MISMATCH: Passage corpus version does not match manifest."
            )
    if corpus_content_hash(passages) != content_hash:
        raise DenseError(
            "DENSE_CORPUS_HASH_MISMATCH: passages.jsonl does not match its manifest."
        )
    return manifest, passages


def _validate_version(version: str) -> None:
    if (
        not isinstance(version, str)
        or not version
        or version in {".", ".."}
        or VERSION_PATTERN.fullmatch(version) is None
    ):
        raise DenseError(
            "DENSE_INVALID_VERSION: Version may contain only letters, numbers, "
            "dots, underscores, and hyphens."
        )


def _validate_build_options(batch_size: int, device: str) -> str:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise DenseError("DENSE_INVALID_BATCH_SIZE: batch_size must be positive.")
    try:
        return resolve_device(device)
    except EmbeddingError as exc:
        raise DenseError(str(exc)) from exc


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise DenseError(f"DENSE_INDEX_NOT_FOUND: Could not read {path}: {exc}.") from exc
    return f"sha256:{hasher.hexdigest()}"


def _write_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, allow_nan=False, indent=2)
            output_file.write("\n")
    except (OSError, ValueError) as exc:
        raise DenseError(
            f"DENSE_OUTPUT_WRITE_FAILED: Could not write {path}: {exc}."
        ) from exc


def build_dense_index(
    corpus_dir: Path,
    output_root: Path,
    version: str,
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str = "cpu",
    model_revision: str | None = None,
    query_prefix: str | None = None,
    *,
    embedder: Embedder | None = None,
    show_progress_bar: bool = True,
) -> Path:
    """Embed passage text and create one immutable IndexFlatIP artifact."""
    _validate_version(version)
    resolved_device = _validate_build_options(batch_size, device)
    index_dir = output_root / version
    if index_dir.exists():
        raise DenseError(
            "DENSE_INDEX_VERSION_EXISTS: "
            f"Index version {version!r} already exists. Use a new version."
        )

    corpus_manifest, passages = _load_and_validate_corpus(corpus_dir)
    if embedder is None:
        try:
            embedder = SentenceTransformerEmbedder(
                model_id, device=resolved_device, model_revision=model_revision
            )
        except EmbeddingError as exc:
            raise DenseError(str(exc)) from exc
    else:
        try:
            metadata_matches = (
                embedder.model_id == model_id
                and embedder.model_revision == model_revision
                and isinstance(embedder.dimension, int)
                and not isinstance(embedder.dimension, bool)
                and embedder.dimension > 0
            )
        except AttributeError as exc:
            raise DenseError(
                "DENSE_INVALID_MODEL: Supplied embedder is missing required metadata."
            ) from exc
        if not metadata_matches:
            raise DenseError(
                "DENSE_MODEL_MISMATCH: Supplied embedder metadata does not match "
                "build configuration."
            )

    texts = [passage["text"] for passage in passages]
    try:
        raw_embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        embeddings = normalized_float32_matrix(
            raw_embeddings,
            expected_count=len(passages),
            expected_dimension=embedder.dimension,
        )
    except EmbeddingError as exc:
        raise DenseError(str(exc)) from exc
    except Exception as exc:
        raise DenseError(
            f"DENSE_ENCODING_FAILED: Passage embedding generation failed: {exc}."
        ) from exc

    faiss = _import_faiss()
    dimension = int(embeddings.shape[1])
    try:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
    except Exception as exc:
        raise DenseError(
            f"DENSE_INDEX_BUILD_FAILED: Could not build FAISS index: {exc}."
        ) from exc
    if index.ntotal != len(passages):
        raise DenseError(
            "DENSE_INDEX_COUNT_MISMATCH: FAISS vector count does not match passages."
        )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root))
    except OSError as exc:
        raise DenseError(
            f"DENSE_OUTPUT_WRITE_FAILED: Could not create build directory: {exc}."
        ) from exc

    try:
        index_path = temporary_dir / "index.faiss"
        embeddings_path = temporary_dir / "embeddings.npy"
        mapping_path = temporary_dir / "passage_ids.json"
        try:
            faiss.write_index(index, str(index_path))
            with embeddings_path.open("wb") as output_file:
                np.save(output_file, embeddings, allow_pickle=False)
        except Exception as exc:
            raise DenseError(
                f"DENSE_OUTPUT_WRITE_FAILED: Could not serialize dense index: {exc}."
            ) from exc
        passage_ids = [passage["passage_id"] for passage in passages]
        _write_json(passage_ids, mapping_path)
        manifest = {
            "artifact_type": "dense_index",
            "retrieval_type": "dense",
            "index_version": version,
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "builder_version": BUILDER_VERSION,
            "corpus": {
                "version": corpus_manifest["corpus_version"],
                "content_hash": corpus_manifest["content_hash"],
                "passage_count": len(passages),
            },
            "embedding": {
                "model_id": model_id,
                "model_revision": model_revision,
                "dimension": dimension,
                "normalize_embeddings": True,
                "dtype": "float32",
                "batch_size": batch_size,
                "provider": "ollama" if hasattr(embedder, "input_prefix") else "sentence_transformers",
                "document_prefix": getattr(embedder, "input_prefix", None),
                "query_prefix": query_prefix,
            },
            "faiss": {
                "index_type": "IndexFlatIP",
                "metric": "cosine_similarity_via_normalized_inner_product",
            },
            "files": {
                "index": {"path": "index.faiss", "sha256": _sha256_file(index_path)},
                "embeddings": {
                    "path": "embeddings.npy",
                    "sha256": _sha256_file(embeddings_path),
                },
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
            raise DenseError(
                "DENSE_INDEX_VERSION_EXISTS: "
                f"Index version {version!r} already exists. Use a new version."
            ) from exc
        except OSError as exc:
            raise DenseError(
                f"DENSE_OUTPUT_WRITE_FAILED: Could not finalize {index_dir}: {exc}."
            ) from exc
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return index_dir


def _validate_index_manifest(manifest: Any) -> dict[str, Any]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_type") != "dense_index"
        or manifest.get("retrieval_type") != "dense"
    ):
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Expected a dense index manifest.")
    for field in ("index_version", "corpus", "embedding", "faiss", "files"):
        if field not in manifest:
            raise DenseError(
                f"DENSE_INDEX_MANIFEST_INVALID: Manifest is missing {field!r}."
            )
    corpus = manifest["corpus"]
    embedding = manifest["embedding"]
    faiss_config = manifest["faiss"]
    files = manifest["files"]
    if not all(isinstance(section, dict) for section in (corpus, embedding, faiss_config, files)):
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Invalid manifest structure.")
    if (
        not isinstance(manifest["index_version"], str)
        or not manifest["index_version"]
        or not isinstance(corpus.get("version"), str)
        or not corpus["version"]
        or not isinstance(corpus.get("passage_count"), int)
        or isinstance(corpus.get("passage_count"), bool)
        or corpus["passage_count"] < 1
        or not isinstance(corpus.get("content_hash"), str)
        or CHECKSUM_PATTERN.fullmatch(corpus["content_hash"]) is None
    ):
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Invalid corpus metadata.")
    if (
        not isinstance(embedding.get("model_id"), str)
        or not embedding["model_id"]
        or embedding.get("model_revision") is not None
        and (
            not isinstance(embedding["model_revision"], str)
            or not embedding["model_revision"]
        )
        or not isinstance(embedding.get("dimension"), int)
        or isinstance(embedding.get("dimension"), bool)
        or embedding["dimension"] < 1
        or embedding.get("normalize_embeddings") is not True
        or embedding.get("dtype") != "float32"
        or not isinstance(embedding.get("batch_size"), int)
        or isinstance(embedding.get("batch_size"), bool)
        or embedding["batch_size"] < 1
    ):
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Invalid embedding metadata.")
    provider = embedding.get("provider", "sentence_transformers")
    if provider not in {"sentence_transformers", "ollama"}:
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Invalid embedding provider.")
    for prefix_field in ("document_prefix", "query_prefix"):
        prefix = embedding.get(prefix_field)
        if prefix is not None and not isinstance(prefix, str):
            raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Invalid embedding prefix metadata.")
    if faiss_config != {
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity_via_normalized_inner_product",
    }:
        raise DenseError("DENSE_INDEX_MANIFEST_INVALID: Unsupported FAISS configuration.")
    expected_paths = {
        "index": "index.faiss",
        "embeddings": "embeddings.npy",
        "passage_ids": "passage_ids.json",
    }
    for key, expected_path in expected_paths.items():
        entry = files.get(key)
        if (
            not isinstance(entry, dict)
            or entry.get("path") != expected_path
            or not isinstance(entry.get("sha256"), str)
            or CHECKSUM_PATTERN.fullmatch(entry["sha256"]) is None
        ):
            raise DenseError(
                f"DENSE_INDEX_MANIFEST_INVALID: Invalid file metadata for {key}."
            )
    return manifest


class DenseRetriever:
    """Validated FAISS retriever that loads its model and artifacts once."""

    def __init__(
        self,
        index_dir: Path,
        corpus_dir: Path,
        device: str = "cpu",
        *,
        embedder: Embedder | None = None,
    ) -> None:
        if not index_dir.is_dir():
            raise DenseError(
                f"DENSE_INDEX_NOT_FOUND: Index directory does not exist: {index_dir}."
            )
        self.index_dir = index_dir
        self.corpus_dir = corpus_dir
        try:
            resolve_device(device)
        except EmbeddingError as exc:
            raise DenseError(str(exc)) from exc
        self.index_manifest = _validate_index_manifest(
            _load_json(index_dir / "manifest.json", "DENSE_INDEX_MANIFEST_INVALID")
        )
        self.corpus_manifest, self.passages = _load_and_validate_corpus(corpus_dir)
        indexed_corpus = self.index_manifest["corpus"]
        if (
            indexed_corpus["version"] != self.corpus_manifest["corpus_version"]
            or indexed_corpus["content_hash"] != self.corpus_manifest["content_hash"]
            or indexed_corpus["passage_count"] != len(self.passages)
        ):
            raise DenseError(
                "DENSE_INDEX_CORPUS_MISMATCH: Dense index and corpus are incompatible."
            )
        self._validate_checksums()
        self.passage_ids = self._load_mapping()
        corpus_ids = [passage["passage_id"] for passage in self.passages]
        if self.passage_ids != corpus_ids or len(set(self.passage_ids)) != len(corpus_ids):
            raise DenseError(
                "DENSE_PASSAGE_MAPPING_MISMATCH: Mapping does not match corpus order."
            )
        self.passages_by_id = {passage["passage_id"]: passage for passage in self.passages}
        self.embeddings = self._load_embeddings()
        self.index = self._load_faiss_index()

        embedding_config = self.index_manifest["embedding"]
        if embedder is None:
            try:
                embedder = SentenceTransformerEmbedder(
                    embedding_config["model_id"],
                    device=device,
                    model_revision=embedding_config["model_revision"],
                )
            except EmbeddingError as exc:
                raise DenseError(str(exc)) from exc
        try:
            model_matches = (
                embedder.model_id == embedding_config["model_id"]
                and embedder.model_revision == embedding_config["model_revision"]
            )
            embedder_dimension = embedder.dimension
        except AttributeError as exc:
            raise DenseError(
                "DENSE_INVALID_MODEL: Query embedder is missing required metadata."
            ) from exc
        if not model_matches:
            raise DenseError(
                "DENSE_MODEL_MISMATCH: Query embedder metadata does not match index."
            )
        expected_query_prefix = embedding_config.get("query_prefix")
        if expected_query_prefix is not None and getattr(embedder, "input_prefix", None) != expected_query_prefix:
            raise DenseError(
                "DENSE_MODEL_MISMATCH: Query embedding prefix does not match index metadata."
            )
        if embedder_dimension != embedding_config["dimension"]:
            raise DenseError(
                "DENSE_DIMENSION_MISMATCH: Model output dimension does not match index."
            )
        self.embedder = embedder

    def _validate_checksums(self) -> None:
        for entry in self.index_manifest["files"].values():
            path = self.index_dir / entry["path"]
            if _sha256_file(path) != entry["sha256"]:
                raise DenseError(
                    "DENSE_INDEX_CHECKSUM_MISMATCH: "
                    f"{path.name} does not match manifest.json."
                )

    def _load_mapping(self) -> list[str]:
        mapping = _load_json(
            self.index_dir / "passage_ids.json", "DENSE_PASSAGE_MAPPING_MISMATCH"
        )
        if not isinstance(mapping, list) or any(
            not isinstance(passage_id, str) or not passage_id for passage_id in mapping
        ):
            raise DenseError(
                "DENSE_PASSAGE_MAPPING_MISMATCH: passage_ids.json must contain strings."
            )
        return mapping

    def _load_embeddings(self) -> np.ndarray:
        try:
            embeddings = np.load(self.index_dir / "embeddings.npy", allow_pickle=False)
            validated = normalized_float32_matrix(
                embeddings,
                expected_count=len(self.passage_ids),
                expected_dimension=self.index_manifest["embedding"]["dimension"],
            )
        except EmbeddingError as exc:
            raise DenseError(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise DenseError(
                f"DENSE_INDEX_LOAD_FAILED: Could not load embeddings.npy: {exc}."
            ) from exc
        if embeddings.dtype != np.float32:
            raise DenseError(
                "DENSE_INDEX_LOAD_FAILED: embeddings.npy must use float32."
            )
        if not np.allclose(embeddings, validated, rtol=1e-5, atol=1e-6):
            raise DenseError(
                "DENSE_INDEX_LOAD_FAILED: Stored embeddings are not L2-normalized."
            )
        return embeddings

    def _load_faiss_index(self):
        faiss = _import_faiss()
        try:
            index = faiss.read_index(str(self.index_dir / "index.faiss"))
        except Exception as exc:
            raise DenseError(
                f"DENSE_INDEX_LOAD_FAILED: Could not load index.faiss: {exc}."
            ) from exc
        dimension = self.index_manifest["embedding"]["dimension"]
        if index.ntotal != len(self.passage_ids):
            raise DenseError(
                "DENSE_INDEX_COUNT_MISMATCH: FAISS vector count does not match mapping."
            )
        if index.d != dimension:
            raise DenseError(
                "DENSE_DIMENSION_MISMATCH: FAISS dimension does not match manifest."
            )
        if type(index).__name__ != "IndexFlatIP":
            raise DenseError(
                "DENSE_INDEX_LOAD_FAILED: FAISS index must be IndexFlatIP."
            )
        return index

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        try:
            normalized_query = normalize_claim_input(query)
        except EmbeddingError as exc:
            raise DenseError(str(exc)) from exc
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 100
        ):
            raise DenseError("DENSE_INVALID_TOP_K: top_k must be between 1 and 100.")

        started = time.perf_counter()
        try:
            raw_query_vector = self.embedder.encode(
                [normalized_query],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vector = normalized_float32_matrix(
                raw_query_vector,
                expected_count=1,
                expected_dimension=self.index_manifest["embedding"]["dimension"],
            )
        except EmbeddingError as exc:
            raise DenseError(str(exc)) from exc
        except Exception as exc:
            raise DenseError(
                f"DENSE_ENCODING_FAILED: Query embedding generation failed: {exc}."
            ) from exc
        requested_count = min(top_k, len(self.passage_ids))
        # Search the complete exact index so passage-ID tie-breaking also applies
        # when a score tie crosses the requested top-k boundary.
        try:
            scores, positions = self.index.search(query_vector, len(self.passage_ids))
        except Exception as exc:
            raise DenseError(
                f"DENSE_SEARCH_FAILED: FAISS search failed: {exc}."
            ) from exc
        candidates: list[tuple[float, str]] = []
        for score_value, position_value in zip(scores[0], positions[0], strict=True):
            position = int(position_value)
            if position == -1:
                continue
            if not 0 <= position < len(self.passage_ids):
                raise DenseError(
                    "DENSE_PASSAGE_MAPPING_MISMATCH: FAISS returned an invalid position."
                )
            score = float(score_value)
            if not math.isfinite(score):
                raise DenseError(
                    "DENSE_INDEX_LOAD_FAILED: FAISS returned a non-finite score."
                )
            candidates.append((score, self.passage_ids[position]))
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))[
            :requested_count
        ]
        results: list[dict[str, Any]] = []
        for rank, (score, passage_id) in enumerate(ranked, start=1):
            passage = self.passages_by_id[passage_id]
            results.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "document_id": passage["document_id"],
                    "dataset": passage["dataset"],
                    "text": passage["text"],
                    "dense_score": score,
                }
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "query": normalized_query,
            "retrieval_mode": "dense",
            "top_k": top_k,
            "returned_count": len(results),
            "latency_ms": latency_ms,
            "corpus_version": self.corpus_manifest["corpus_version"],
            "index_version": self.index_manifest["index_version"],
            "embedding_model": self.index_manifest["embedding"]["model_id"],
            "results": results,
        }
