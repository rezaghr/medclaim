import pytest

from medclaim.evidence_gate.gate import EvidenceGateDecision
from medclaim.verification.aggregation import AggregationError, aggregate_component_results
from medclaim.verification.models import AtomicClaimResult


def component(index, verdict, confidence=0.8, evidence=None):
    return AtomicClaimResult(
        component_id=f"claim:component:{index}",
        claim=f"Claim {index}",
        verdict=verdict,
        confidence=confidence,
        explanation=f"Explanation {index}",
        evidence_used=evidence or [],
        gate_decision=EvidenceGateDecision(
            status="PROCEED",
            reason="RELEVANCE_REQUIREMENTS_MET",
            threshold=0.5,
            score_field="reranker_score",
            top_score=0.8,
            relevant_passage_count=1,
            unique_document_count=1,
            evidence_passage_ids=evidence or [],
            gate_version="gate-v1",
        ),
    )


@pytest.mark.parametrize("verdict", ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
def test_uniform_components_keep_verdict(verdict):
    result = aggregate_component_results([component(1, verdict), component(2, verdict)])
    assert result.verdict == verdict


@pytest.mark.parametrize(
    "verdicts",
    [
        ("SUPPORTS", "REFUTES"),
        ("SUPPORTS", "NOT_ENOUGH_INFO"),
        ("REFUTES", "NOT_ENOUGH_INFO"),
        ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"),
    ],
)
def test_different_component_verdicts_are_mixed(verdicts):
    result = aggregate_component_results([component(i, verdict) for i, verdict in enumerate(verdicts, 1)])
    assert result.verdict == "MIXED"
    assert result.explanation.endswith("The overall result is MIXED.")


def test_confidence_is_minimum_and_evidence_is_ordered_and_deduplicated():
    result = aggregate_component_results([
        component(1, "SUPPORTS", 0.9, ["p2", "p1"]),
        component(2, "REFUTES", 0.6, ["p1", "p3"]),
    ])
    assert result.confidence == 0.6
    assert result.evidence_used == ["p2", "p1", "p3"]
    assert [row.component_id for row in result.component_results] == ["claim:component:1", "claim:component:2"]


def test_one_component_preserves_explanation():
    result = aggregate_component_results([component(1, "SUPPORTS")])
    assert result.explanation == "Explanation 1"


def test_invalid_verdict_is_rejected():
    with pytest.raises(AggregationError, match="AGGREGATION_INVALID_VERDICT"):
        aggregate_component_results([component(1, "MIXED")])
