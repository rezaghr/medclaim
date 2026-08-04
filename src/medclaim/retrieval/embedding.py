"""Embedding model adapters and vector validation for dense retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Protocol

import httpx
import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_QUERY_CHARACTERS = 5000
SUPPORTED_DEVICES = {"cpu", "cuda", "auto"}


class EmbeddingError(Exception):
    """Raised when an embedding model or vector output is invalid."""


class Embedder(Protocol):
    model_id: str
    model_revision: str | None
    dimension: int
    device: str

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


def resolve_device(device: str) -> str:
    if device not in SUPPORTED_DEVICES:
        raise EmbeddingError(
            "DENSE_INVALID_DEVICE: Device must be 'cpu', 'cuda', or 'auto'."
        )
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_claim_input(value: Any) -> str:
    if not isinstance(value, str):
        raise EmbeddingError("DENSE_EMPTY_QUERY: Query must be a non-empty string.")
    normalized = unicodedata.normalize("NFKC", value)
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in normalized
    )
    cleaned = re.sub(r"\s+", " ", without_controls).strip()
    if not cleaned:
        raise EmbeddingError("DENSE_EMPTY_QUERY: Query must be a non-empty string.")
    if len(cleaned) > MAX_QUERY_CHARACTERS:
        raise EmbeddingError(
            "DENSE_QUERY_TOO_LONG: Query exceeds the 5000-character limit."
        )
    return cleaned


def normalized_float32_matrix(
    vectors: Any,
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Validate, convert, and L2-normalize a two-dimensional vector matrix."""
    try:
        matrix = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"DENSE_INVALID_EMBEDDINGS: Embeddings are not a regular matrix: {exc}."
        ) from exc
    if matrix.ndim != 2:
        raise EmbeddingError(
            "DENSE_INVALID_EMBEDDINGS: Embeddings must be a two-dimensional matrix."
        )
    if matrix.shape[0] != expected_count:
        raise EmbeddingError(
            "DENSE_VECTOR_COUNT_MISMATCH: Embedding row count does not match passages."
        )
    if matrix.shape[1] < 1:
        raise EmbeddingError(
            "DENSE_INVALID_EMBEDDINGS: Embedding dimension must be positive."
        )
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise EmbeddingError(
            "DENSE_DIMENSION_MISMATCH: Embedding dimension does not match the index."
        )
    try:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError(
            f"DENSE_INVALID_EMBEDDINGS: Could not convert embeddings to float32: {exc}."
        ) from exc
    if not np.isfinite(matrix).all():
        raise EmbeddingError(
            "DENSE_NONFINITE_EMBEDDING: Embeddings contain NaN or infinite values."
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise EmbeddingError(
            "DENSE_ZERO_NORM_EMBEDDING: Embeddings must have non-zero finite norms."
        )
    normalized = np.ascontiguousarray(matrix / norms, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise EmbeddingError(
            "DENSE_NONFINITE_EMBEDDING: Normalized embeddings are not finite."
        )
    return normalized


class SentenceTransformerEmbedder:
    """Lazy adapter around SentenceTransformer for production embedding calls."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        model_revision: str | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise EmbeddingError(
                "DENSE_INVALID_MODEL: Model ID must be a non-empty string."
            )
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise EmbeddingError(
                "DENSE_INVALID_MODEL: Model revision must be null or non-empty."
            )
        self.model_id = model_id.strip()
        self.model_revision = model_revision.strip() if model_revision else None
        self.device = resolve_device(device)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "DENSE_DEPENDENCY_MISSING: Install sentence-transformers to load "
                "the embedding model."
            ) from exc
        arguments: dict[str, Any] = {"device": self.device}
        if self.model_revision is not None:
            arguments["revision"] = self.model_revision
        try:
            self.model = SentenceTransformer(self.model_id, **arguments)
            dimension = self.model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise EmbeddingError(
                f"DENSE_MODEL_LOAD_FAILED: Could not load {self.model_id}: {exc}."
            ) from exc
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise EmbeddingError(
                "DENSE_INVALID_MODEL: Model returned an invalid embedding dimension."
            )
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        try:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"DENSE_ENCODING_FAILED: Embedding generation failed: {exc}."
            ) from exc


class OllamaEmbedder:
    """Embedding adapter for a locally installed Ollama embedding model."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        dimension: int | None = None,
        input_prefix: str = "",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise EmbeddingError("DENSE_INVALID_MODEL: Model ID must be non-empty.")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise EmbeddingError("DENSE_INVALID_MODEL: Ollama base URL must be HTTP(S).")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise EmbeddingError("DENSE_INVALID_MODEL: Ollama timeout must be positive.")
        self.model_id = model_id.strip()
        self.model_revision = None
        self.device = "ollama"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if not isinstance(input_prefix, str):
            raise EmbeddingError("DENSE_INVALID_MODEL: Embedding prefix must be text.")
        self.input_prefix = input_prefix
        if dimension is None:
            probe = self._request(["embedding dimension probe"])
            if len(probe) != 1 or not probe[0]:
                raise EmbeddingError("DENSE_INVALID_MODEL: Ollama returned no probe embedding.")
            self.dimension = len(probe[0])
        elif isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
            self.dimension = dimension
        else:
            raise EmbeddingError("DENSE_INVALID_MODEL: Embedding dimension must be positive.")

    def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            with httpx.Client(trust_env=False) as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model_id,
                        "input": [f"{self.input_prefix}{text}" for text in texts],
                    },
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise EmbeddingError("DENSE_ENCODING_TIMEOUT: Ollama embedding timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(
                f"DENSE_ENCODING_FAILED: Ollama embedding request failed: {exc}."
            ) from exc
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                "DENSE_VECTOR_COUNT_MISMATCH: Ollama embedding count does not match inputs."
            )
        return embeddings

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise EmbeddingError("DENSE_INVALID_EMBEDDINGS: Text inputs must be a list of strings.")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise EmbeddingError("DENSE_INVALID_EMBEDDINGS: Batch size must be positive.")
        matrix = np.empty((len(texts), self.dimension), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._request(batch)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    "DENSE_VECTOR_COUNT_MISMATCH: Ollama embedding count does not match inputs."
                )
            try:
                matrix[start : start + len(batch)] = np.asarray(vectors, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EmbeddingError(
                    f"DENSE_INVALID_EMBEDDINGS: Ollama returned invalid vectors: {exc}."
                ) from exc
        if normalize_embeddings:
            return normalized_float32_matrix(
                matrix,
                expected_count=len(texts),
                expected_dimension=self.dimension,
            )
        if matrix.shape != (len(texts), self.dimension) or not np.isfinite(matrix).all():
            raise EmbeddingError("DENSE_INVALID_EMBEDDINGS: Ollama returned invalid vectors.")
        return matrix
