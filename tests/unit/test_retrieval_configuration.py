import json
from pathlib import Path

import pytest

from medclaim.retrieval.configuration import (
    RetrievalConfigurationError,
    RetrievalSettings,
    create_retriever,
    load_retrieval_settings,
)
from medclaim.retrieval.hybrid import HybridRetriever

CONFIG = Path(__file__).resolve().parents[2] / "configs/retrieval/hybrid_scifact_v1.json"


class ReadyRetriever:
    index_manifest = {
        "index_version": "v1",
        "corpus": {
            "version": "corpus-v1",
            "content_hash": "sha256:" + "a" * 64,
            "passage_count": 1,
        },
    }


def test_default_configuration_file():
    settings = load_retrieval_settings(CONFIG)
    assert settings == RetrievalSettings()


def test_unknown_and_missing_configuration_fields_fail(tmp_path):
    values = json.loads(CONFIG.read_text())
    values["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(values))
    with pytest.raises(RetrievalConfigurationError, match="UNKNOWN_FIELD"):
        load_retrieval_settings(path)
    values.pop("unknown")
    values.pop("rrf_k")
    path.write_text(json.dumps(values))
    with pytest.raises(RetrievalConfigurationError, match="MISSING_FIELD"):
        load_retrieval_settings(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "invalid"},
        {"sparse_top_k": 0},
        {"dense_top_k": 101},
        {"fusion_top_k": 100, "sparse_top_k": 1, "dense_top_k": 1},
        {"rrf_k": 0},
        {"final_evidence_k": 31},
    ],
)
def test_settings_validation(overrides):
    values = {
        "mode": "hybrid",
        "sparse_top_k": 50,
        "dense_top_k": 50,
        "fusion_top_k": 30,
        "rrf_k": 60,
        "final_evidence_k": 5,
    }
    values.update(overrides)
    with pytest.raises(RetrievalConfigurationError):
        RetrievalSettings.from_dict(values)


def test_mode_selector_supports_all_retrieval_modes():
    sparse = ReadyRetriever()
    dense = ReadyRetriever()
    assert create_retriever(
        "bm25", sparse_retriever=sparse, dense_retriever=dense
    ) is sparse
    assert create_retriever(
        "dense", sparse_retriever=sparse, dense_retriever=dense
    ) is dense
    assert isinstance(
        create_retriever(
            "hybrid", sparse_retriever=sparse, dense_retriever=dense
        ),
        HybridRetriever,
    )
    reranked = object()
    assert create_retriever(
        "hybrid_reranked",
        sparse_retriever=sparse,
        dense_retriever=dense,
        reranked_retriever=reranked,
    ) is reranked
    with pytest.raises(RetrievalConfigurationError, match="RETRIEVAL_NOT_READY"):
        create_retriever(
            "hybrid", sparse_retriever=sparse, dense_retriever=None
        )
