import json
from pathlib import Path

import pytest

from medclaim.reranking.models import (
    RerankingConfiguration,
    RerankingConfigurationError,
    load_reranking_configuration,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs/reranking/scifact_cross_encoder_v1.json"


def test_default_reranking_configuration_file():
    assert load_reranking_configuration(CONFIG) == RerankingConfiguration()


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_count": 0},
        {"candidate_count": 101},
        {"final_evidence_k": 21},
        {"batch_size": 0},
        {"device": "tpu"},
        {"maximum_input_length": 0},
    ],
)
def test_invalid_reranking_configuration(overrides):
    values = json.loads(CONFIG.read_text())
    values.update(overrides)
    with pytest.raises(RerankingConfigurationError, match="RERANKER_INVALID_CONFIGURATION"):
        RerankingConfiguration.from_dict(values)


def test_unknown_configuration_field(tmp_path):
    values = json.loads(CONFIG.read_text())
    values["unexpected"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    with pytest.raises(RerankingConfigurationError, match="unknown field"):
        load_reranking_configuration(path)
