"""Deterministic no-model runtime fixture for CI and deployment smoke tests."""

from __future__ import annotations


class FakeVerificationPipeline:
    """Safe smoke pipeline that never invents evidence or a substantive verdict."""

    def verify(self, claim: str, request_id: str) -> dict:
        return {
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.0,
            "explanation": (
                "The fake smoke-test provider has no indexed evidence and cannot establish this claim."
            ),
            "evidence_used": [],
            "component_results": [
                {
                    "component_id": f"{request_id}:component:1",
                    "verdict": "NOT_ENOUGH_INFO",
                    "confidence": 0.0,
                    "evidence_used": [],
                    "gate_decision": {
                        "status": "ABSTAIN",
                        "reason": "FAKE_PROVIDER_NO_EVIDENCE",
                    },
                }
            ],
            "limitations": [
                "Fake provider mode is for deterministic smoke tests only.",
                "No corpus evidence was evaluated.",
            ],
            "technical_metadata": {
                "provider": "fake",
                "confidence_note": "Confidence is not a clinical probability.",
            },
        }
