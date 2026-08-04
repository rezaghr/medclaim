import pytest

from medclaim.explanation.evaluation import evaluate_explanation
from medclaim.explanation.validation import (
    ExplanationValidationError,
    ExplanationValidator,
    validate_with_one_correction,
)


PASSAGES = [
    {"passage_id": "p1", "document_id": "d1", "text": "The evidence directly supports the measured association."},
    {"passage_id": "p2", "document_id": "d2", "text": "The trial evidence contradicts the proposed effect."},
]


@pytest.mark.parametrize(
    "result,gate_abstained",
    [
        ({"verdict": "SUPPORTS", "explanation": "The selected evidence directly supports the measured claim.", "evidence_used": ["p1"], "limitations": []}, False),
        ({"verdict": "REFUTES", "explanation": "The selected evidence contradicts the stated medical claim.", "evidence_used": ["p2"], "limitations": []}, False),
        ({"verdict": "NOT_ENOUGH_INFO", "explanation": "The indexed evidence is insufficient to establish this claim.", "evidence_used": [], "limitations": []}, True),
        ({"verdict": "MIXED", "explanation": "The components have different outcomes, so the overall result is mixed.", "evidence_used": ["p1", "p2"], "limitations": [], "component_results": [{"verdict": "SUPPORTS"}, {"verdict": "REFUTES"}]}, False),
    ],
)
def test_valid_explanations(result, gate_abstained):
    assert ExplanationValidator().validate(result, PASSAGES, gate_abstained=gate_abstained).valid


@pytest.mark.parametrize(
    "verdict,explanation,evidence_id",
    [
        ("SUPPORTS", "The observed result establishes that the claim is true.", "p1"),
        ("REFUTES", "The observed cases are counterexamples, so the claim is false.", "p2"),
        ("REFUTES", "The observed residual events disprove the universal claim.", "p2"),
        ("REFUTES", "The claim that the intervention prevents every event is false.", "p2"),
    ],
)
def test_equivalent_verdict_language_is_accepted(verdict, explanation, evidence_id):
    result = {
        "verdict": verdict,
        "explanation": explanation,
        "evidence_used": [evidence_id],
        "limitations": [],
    }
    assert ExplanationValidator().validate(result, PASSAGES).valid


@pytest.mark.parametrize(
    "result,error",
    [
        ({"verdict": "SUPPORTS", "explanation": "The selected evidence supports this claim clearly.", "evidence_used": ["unknown"]}, "CITATION_INVALID"),
        ({"verdict": "SUPPORTS", "explanation": "The selected evidence supports this claim clearly.", "evidence_used": ["p1", "p1"]}, "CITATION_INVALID"),
        ({"verdict": "SUPPORTS", "explanation": "", "evidence_used": ["p1"]}, "EMPTY"),
        ({"verdict": "SUPPORTS", "explanation": "Too short", "evidence_used": ["p1"]}, "LENGTH_INVALID"),
        ({"verdict": "SUPPORTS", "explanation": "The evidence contradicts the claim and rejects it.", "evidence_used": ["p1"]}, "VERDICT_CONFLICT"),
        ({"verdict": "REFUTES", "explanation": "The evidence directly supports the claim as written.", "evidence_used": ["p2"]}, "VERDICT_CONFLICT"),
        ({"verdict": "NOT_ENOUGH_INFO", "explanation": "The evidence discusses this medical topic in detail.", "evidence_used": ["p1"]}, "VERDICT_CONFLICT"),
        ({"verdict": "SUPPORTS", "explanation": "Based on this, you should take the medication daily.", "evidence_used": ["p1"]}, "MEDICAL_ADVICE"),
        ({"verdict": "SUPPORTS", "explanation": "The selected evidence supports that intervention causes recovery.", "evidence_used": ["p1"]}, "CAUSAL_OVERSTATEMENT"),
        ({"verdict": "SUPPORTS", "explanation": "Evidence 7 directly supports the selected medical claim.", "evidence_used": ["p1"]}, "CITATION_INVALID"),
    ],
)
def test_invalid_explanations(result, error):
    with pytest.raises(ExplanationValidationError, match=error):
        ExplanationValidator().validate(result, PASSAGES)


def test_explanation_too_long_is_not_silently_truncated():
    result = {"verdict": "SUPPORTS", "explanation": "word " * 121, "evidence_used": ["p1"]}
    with pytest.raises(ExplanationValidationError, match="LENGTH_INVALID"):
        ExplanationValidator().validate(result, PASSAGES)


def test_mixed_requires_component_disagreement():
    result = {
        "verdict": "MIXED", "explanation": "The components have different outcomes and a mixed result.",
        "evidence_used": ["p1"], "component_results": [{"verdict": "SUPPORTS"}, {"verdict": "SUPPORTS"}],
    }
    with pytest.raises(ExplanationValidationError, match="MIXED_INVALID"):
        ExplanationValidator().validate(result, PASSAGES)


def test_one_llm_correction_attempt():
    class Verifier:
        def __init__(self):
            self.corrections = 0

        def verify(self, claim, evidence):
            return {"verdict": "SUPPORTS", "explanation": "short", "evidence_used": ["p1"]}

        def correct(self, claim, evidence, result, error):
            self.corrections += 1
            return {"verdict": "SUPPORTS", "explanation": "The selected evidence directly supports the measured claim.", "evidence_used": ["p1"]}

    verifier = Verifier()
    result = validate_with_one_correction(verifier, "claim", PASSAGES, ExplanationValidator())
    assert result["verdict"] == "SUPPORTS"
    assert verifier.corrections == 1


def test_automated_evaluation_keeps_llm_judge_disabled():
    result = {"verdict": "SUPPORTS", "explanation": "The selected evidence directly supports the measured claim.", "evidence_used": ["p1"]}
    report = evaluate_explanation(result, PASSAGES)
    assert report["status"] == "valid"
    assert report["llm_judge"] == {"enabled": False, "status": "not_run"}
