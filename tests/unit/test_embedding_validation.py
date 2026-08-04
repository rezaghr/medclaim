import numpy as np
import pytest

from medclaim.retrieval.embedding import (
    EmbeddingError,
    MAX_QUERY_CHARACTERS,
    normalize_claim_input,
    normalized_float32_matrix,
    resolve_device,
)


def test_embedding_conversion_and_vector_normalization():
    matrix = normalized_float32_matrix(
        [[3.0, 4.0], [0.0, 2.0]], expected_count=2
    )
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])


@pytest.mark.parametrize("value", [[[np.nan, 1.0]], [[np.inf, 1.0]]])
def test_nonfinite_vectors_are_rejected(value):
    with pytest.raises(EmbeddingError, match="DENSE_NONFINITE_EMBEDDING"):
        normalized_float32_matrix(value, expected_count=1)


def test_inconsistent_dimension_and_count_are_rejected():
    with pytest.raises(EmbeddingError, match="DENSE_DIMENSION_MISMATCH"):
        normalized_float32_matrix(
            [[1.0, 0.0]], expected_count=1, expected_dimension=3
        )
    with pytest.raises(EmbeddingError, match="DENSE_VECTOR_COUNT_MISMATCH"):
        normalized_float32_matrix([[1.0, 0.0]], expected_count=2)


def test_zero_norm_vector_is_rejected():
    with pytest.raises(EmbeddingError, match="DENSE_ZERO_NORM_EMBEDDING"):
        normalized_float32_matrix([[0.0, 0.0]], expected_count=1)


def test_claim_input_normalization_and_validation():
    assert normalize_claim_input("  Vitamin\tD\nclaim  ") == "Vitamin D claim"
    with pytest.raises(EmbeddingError, match="DENSE_EMPTY_QUERY"):
        normalize_claim_input("   ")
    with pytest.raises(EmbeddingError, match="DENSE_QUERY_TOO_LONG"):
        normalize_claim_input("x" * (MAX_QUERY_CHARACTERS + 1))


def test_device_validation_and_auto_cpu_without_torch(monkeypatch):
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(EmbeddingError, match="DENSE_INVALID_DEVICE"):
        resolve_device("tpu")
