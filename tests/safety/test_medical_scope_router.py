import pytest

from medclaim.safety import MANDATORY_SAFETY_DISCLAIMER, route_scope


@pytest.mark.parametrize(
    ("text", "action", "category"),
    [
        (
            "Vitamin D supplementation reduces respiratory infection incidence.",
            "VERIFY",
            "PUBLIC_HEALTH_CLAIM",
        ),
        (
            "Expired boxes of cake and pancake mix are dangerously toxic.",
            "VERIFY",
            "PUBLIC_HEALTH_CLAIM",
        ),
        (
            "Mold-contaminated flour can trigger an allergic reaction.",
            "VERIFY",
            "PUBLIC_HEALTH_CLAIM",
        ),
        (
            "Lead in drinking water is a health hazard.",
            "VERIFY",
            "PUBLIC_HEALTH_CLAIM",
        ),
        (
            "Citrullinated proteins externalized in neutrophil extracellular traps "
            "act indirectly to disrupt the inflammatory cycle.",
            "VERIFY",
            "MEDICAL_CLAIM",
        ),
        (
            "Macrophages release cytokines that regulate the immune response.",
            "VERIFY",
            "MEDICAL_CLAIM",
        ),
        ("Aspirin prevents every heart attack.", "VERIFY", "MEDICAL_CLAIM"),
        ("What illness do I have?", "LIMIT_SCOPE", "PERSONAL_DIAGNOSIS"),
        ("What medication should I take?", "LIMIT_SCOPE", "TREATMENT_REQUEST"),
        ("What dosage should I use?", "LIMIT_SCOPE", "DOSAGE_REQUEST"),
        ("Should I stop my prescription?", "LIMIT_SCOPE", "MEDICATION_CHANGE_REQUEST"),
        (
            "I have chest pain and can't breathe.",
            "EMERGENCY_LIMITED_RESPONSE",
            "EMERGENCY_PERSONAL_REQUEST",
        ),
        ("Who won the football match?", "LIMIT_SCOPE", "NON_MEDICAL"),
        ("Expired concert tickets cannot be refunded.", "LIMIT_SCOPE", "NON_MEDICAL"),
    ],
)
def test_scope_categories(text, action, category):
    decision = route_scope(text)
    assert (decision.action, decision.category) == (action, category)


def test_disclaimer_states_complete_boundary():
    assert "limited indexed corpus" in MANDATORY_SAFETY_DISCLAIMER
    assert "not a doctor" in MANDATORY_SAFETY_DISCLAIMER
    assert "not medical advice" in MANDATORY_SAFETY_DISCLAIMER
