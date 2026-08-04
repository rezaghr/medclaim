from medclaim.runtime.configuration import RuntimeSettings
from medclaim.runtime.service import VerificationService


class Pipeline:
    def __init__(self):
        self.calls = 0

    def verify(self, claim, request_id):
        self.calls += 1
        return {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.2,
            "explanation": "The limited corpus does not establish this medical claim.",
            "evidence_used": ["p1"],
            "component_results": [],
            "limitations": ["Limited corpus."],
        }


def test_limited_and_emergency_requests_never_call_pipeline():
    pipeline = Pipeline()
    service = VerificationService(RuntimeSettings(), pipeline)
    for claim in (
        "What illness do I have?",
        "What medication should I take?",
        "What dosage should I use?",
        "Should I stop my prescription?",
        "I have chest pain and can't breathe.",
    ):
        result = service.verify(claim)
        assert result["verification"] is None
        assert "not medical advice" in result["safety_disclaimer"]
    assert pipeline.calls == 0


def test_declarative_claim_is_verified_with_one_request_id():
    pipeline = Pipeline()
    result = VerificationService(RuntimeSettings(), pipeline).verify(
        "Aspirin affects heart health.", "req-17"
    )
    assert pipeline.calls == 1
    assert result["request_id"] == "req-17"
    assert result["scope_decision"]["action"] == "VERIFY"
    assert result["verification"]["verdict"] == "NOT_ENOUGH_INFO"
