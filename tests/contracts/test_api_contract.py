from fastapi.testclient import TestClient

from medclaim.api.app import create_app
from medclaim.runtime.configuration import RuntimeSettings
from medclaim.runtime.service import VerificationService


class Pipeline:
    def verify(self, claim, request_id):
        return {
            "verdict": "SUPPORTS",
            "confidence": 0.7,
            "explanation": "The supplied indexed passage supports this textual claim.",
            "evidence_used": ["p1"],
            "component_results": [],
            "limitations": ["Limited corpus."],
        }


def client():
    settings = RuntimeSettings()
    return TestClient(
        create_app(settings=settings, service=VerificationService(settings, Pipeline()))
    )


def test_api_disclaimer_scope_and_request_id_contract():
    response = client().post(
        "/v1/verify",
        json={"claim": "What dosage should I use?"},
        headers={"X-Request-ID": "contract-1"},
    )
    assert response.status_code == 200
    value = response.json()
    assert value["request_id"] == "contract-1"
    assert value["scope_decision"]["category"] == "DOSAGE_REQUEST"
    assert value["verification"] is None
    assert "not medical advice" in value["safety_disclaimer"]


def test_api_verification_metrics_and_documentation_contract():
    api = client()
    response = api.post("/v1/verify", json={"claim": "Aspirin affects heart health."})
    assert response.status_code == 200
    assert response.json()["verification"]["verdict"] == "SUPPORTS"
    assert api.get("/health/live").json() == {"status": "alive"}
    assert api.get("/health/ready").status_code == 503
    assert "verification_requests_total" in api.get("/metrics").text
    schema = api.get("/openapi.json").json()
    assert "not a doctor" in schema["info"]["description"]


def test_api_accepts_indexed_food_safety_claim():
    response = client().post(
        "/v1/verify",
        json={
            "claim": "Expired boxes of cake and pancake mix are dangerously toxic."
        },
    )
    assert response.status_code == 200
    value = response.json()
    assert value["scope_decision"]["action"] == "VERIFY"
    assert value["scope_decision"]["category"] == "PUBLIC_HEALTH_CLAIM"
    assert value["verification"]["verdict"] == "SUPPORTS"


def test_api_accepts_indexed_biomedical_mechanism_claim():
    response = client().post(
        "/v1/verify",
        json={
            "claim": (
                "Citrullinated proteins externalized in neutrophil extracellular traps "
                "act indirectly to disrupt the inflammatory cycle."
            )
        },
    )
    assert response.status_code == 200
    value = response.json()
    assert value["scope_decision"]["action"] == "VERIFY"
    assert value["scope_decision"]["category"] == "MEDICAL_CLAIM"
    assert value["scope_decision"]["rule_version"] == "medical-scope-v3"


def test_api_rejects_unknown_request_fields_and_unsafe_request_id():
    api = client()
    assert (
        api.post(
            "/v1/verify",
            json={"claim": "Aspirin affects health.", "dataset": "unknown"},
        ).status_code
        == 422
    )
    assert (
        api.post(
            "/v1/verify",
            json={"claim": "Aspirin affects health."},
            headers={"X-Request-ID": "has spaces"},
        ).status_code
        == 400
    )


def test_default_fake_provider_is_safe_and_deterministic():
    api = TestClient(create_app(settings=RuntimeSettings()))
    value = api.post("/v1/verify", json={"claim": "Aspirin affects heart health."}).json()
    assert value["verification"]["verdict"] == "NOT_ENOUGH_INFO"
    assert value["verification"]["evidence_used"] == []
    assert value["verification"]["technical_metadata"]["provider"] == "fake"
