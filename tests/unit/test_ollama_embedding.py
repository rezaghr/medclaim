import numpy as np
import pytest

from medclaim.retrieval.embedding import EmbeddingError, OllamaEmbedder


def test_ollama_embedder_batches_and_normalizes(monkeypatch):
    calls = []

    def request(self, texts):
        calls.append(list(texts))
        return [[3.0, 4.0] for _ in texts]

    monkeypatch.setattr(OllamaEmbedder, "_request", request)
    embedder = OllamaEmbedder("nomic", dimension=2)
    result = embedder.encode(
        ["one", "two", "three"],
        batch_size=2,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert calls == [["one", "two"], ["three"]]
    assert result.dtype == np.float32
    assert np.allclose(result, [[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]])


def test_ollama_embedder_rejects_wrong_vector_count(monkeypatch):
    monkeypatch.setattr(OllamaEmbedder, "_request", lambda self, texts: [])
    embedder = OllamaEmbedder("nomic", dimension=2)
    with pytest.raises(EmbeddingError, match="VECTOR_COUNT_MISMATCH"):
        embedder.encode(
            ["one"],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
