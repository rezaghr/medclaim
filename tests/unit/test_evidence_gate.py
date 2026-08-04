import math

import pytest

from medclaim.evidence_gate.gate import (
    EvidenceGate,
    EvidenceGateConfiguration,
    EvidenceGateError,
)


def candidate(passage_id, score, document_id="d1"):
    return {
        "passage_id": passage_id,
        "document_id": document_id,
        "text": "Evidence text.",
        "reranker_score": score,
    }


def gate(**overrides):
    values = {
        "version": "gate-v1",
        "enabled": True,
        "minimum_score": 0.5,
        "minimum_relevant_passages": 1,
        "minimum_unique_documents": 1,
        "score_field": "reranker_score",
    }
    values.update(overrides)
    return EvidenceGate(EvidenceGateConfiguration(**values))


def test_no_candidates_abstains_with_version():
    decision, selected = gate().evaluate([])
    assert decision.status == "ABSTAIN"
    assert decision.reason == "NO_CANDIDATES"
    assert decision.gate_version == "gate-v1"
    assert selected == []


def test_score_below_threshold_abstains():
    decision = gate().decide([candidate("p1", 0.49)])
    assert decision.reason == "TOP_SCORE_BELOW_THRESHOLD"
    assert decision.top_score == 0.49
    assert decision.score_field == "reranker_score"


def test_relevant_candidate_proceeds_and_only_relevant_is_selected():
    decision, selected = gate().evaluate([candidate("p1", 0.8), candidate("p2", 0.2)])
    assert decision.status == "PROCEED"
    assert decision.evidence_passage_ids == ["p1"]
    assert [row["passage_id"] for row in selected] == ["p1"]


def test_relevant_passage_count_failure():
    decision = gate(minimum_relevant_passages=2).decide(
        [candidate("p1", 0.8), candidate("p2", 0.2)]
    )
    assert decision.reason == "TOO_FEW_RELEVANT_PASSAGES"


def test_unique_document_count_failure():
    decision = gate(minimum_unique_documents=2).decide(
        [candidate("p1", 0.8), candidate("p2", 0.7)]
    )
    assert decision.reason == "TOO_FEW_UNIQUE_DOCUMENTS"


def test_gate_disabled_does_not_require_scores():
    rows = [{"passage_id": "p1", "document_id": "d1", "text": "Evidence."}]
    decision, selected = gate(enabled=False).evaluate(rows)
    assert decision.status == "PROCEED"
    assert selected == rows


@pytest.mark.parametrize("score", [math.nan, math.inf, "0.8", True])
def test_invalid_score_is_rejected(score):
    with pytest.raises(EvidenceGateError, match="GATE_SCORE_FIELD_MISSING"):
        gate().decide([candidate("p1", score)])


def test_missing_score_field_is_rejected():
    row = candidate("p1", 0.8)
    del row["reranker_score"]
    with pytest.raises(EvidenceGateError, match="GATE_SCORE_FIELD_MISSING"):
        gate().decide([row])


def test_gate_result_is_deterministic():
    rows = [candidate("p1", 0.8), candidate("p2", 0.7, "d2")]
    assert gate().decide(rows) == gate().decide(rows)
