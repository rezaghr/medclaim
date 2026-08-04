from medclaim.decomposition.decomposer import ClaimDecomposer
from medclaim.evidence_gate.gate import EvidenceGate, EvidenceGateConfiguration
from medclaim.verification.pipeline import VerificationPipeline


class FakeDecomposer:
    def decompose(self, claim, prompt):
        left, right = claim.split(" and ")
        return {
            "is_compound": True,
            "atomic_claims": [
                {"index": 1, "text": left, "source_span": left},
                {"index": 2, "text": right, "source_span": right},
            ],
        }


class FakeRetriever:
    def __init__(self, weak=()):
        self.queries = []
        self.weak = set(weak)

    def search(self, query, top_k):
        self.queries.append(query)
        score = 0.1 if query in self.weak else 0.9
        return {"results": [{
            "passage_id": f"p:{len(self.queries)}",
            "document_id": f"d:{len(self.queries)}",
            "text": f"Evidence for {query}",
            "reranker_score": score,
        }]}


class FakeVerifier:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def verify(self, claim, evidence):
        self.calls.append((claim, evidence))
        verdict = self.verdicts[claim]
        relationship = {
            "SUPPORTS": "supports",
            "REFUTES": "contradicts",
            "NOT_ENOUGH_INFO": "is insufficient to establish",
        }[verdict]
        return {
            "verdict": verdict,
            "confidence": 0.8,
            "explanation": f"The selected evidence {relationship} this claim component.",
            "evidence_used": [row["passage_id"] for row in evidence],
        }


def gate():
    return EvidenceGate(EvidenceGateConfiguration("gate-v1", True, 0.5, 1, 1, "reranker_score"))


def test_atomic_supported_claim_runs_once_and_is_streamlit_ready():
    retriever = FakeRetriever()
    verifier = FakeVerifier({"Aspirin reduces pain.": "SUPPORTS"})
    result = VerificationPipeline(retriever, verifier, gate(), decomposition_mode="off").verify(
        "Aspirin reduces pain.", "claim:1"
    )
    value = result.to_dict()
    assert value["verdict"] == "SUPPORTS"
    assert value["component_results"][0]["component_id"] == "claim:1:component:1"
    assert value["component_results"][0]["gate_decision"]["status"] == "PROCEED"
    assert value["component_results"][0]["retrieved_candidates"] == [
        {
            "passage_id": "p:1",
            "document_id": "d:1",
            "text": "Evidence for Aspirin reduces pain.",
            "reranker_score": 0.9,
        }
    ]
    assert value["component_results"][0]["model_input_evidence"] == [
        {"passage_id": "p:1", "text": "Evidence for Aspirin reduces pain."}
    ]
    assert len(retriever.queries) == len(verifier.calls) == 1


def test_gate_abstention_avoids_verifier_and_returns_controlled_nei():
    claim = "Unknown intervention works."
    retriever = FakeRetriever(weak={claim})
    verifier = FakeVerifier({claim: "SUPPORTS"})
    result = VerificationPipeline(retriever, verifier, gate(), decomposition_mode="off").verify(claim, "req1")
    assert result.verdict == "NOT_ENOUGH_INFO"
    assert result.confidence == 0.0
    assert result.evidence_used == []
    assert result.component_results[0].retrieved_candidates[0]["passage_id"] == "p:1"
    assert result.component_results[0].model_input_evidence == []
    assert verifier.calls == []
    assert "not sufficiently relevant" in result.explanation


def test_supported_and_refuted_components_produce_mixed_with_independent_retrieval():
    claim = "A reduces pain and B causes harm"
    retriever = FakeRetriever()
    verifier = FakeVerifier({"A reduces pain": "SUPPORTS", "B causes harm": "REFUTES"})
    pipeline = VerificationPipeline(
        retriever, verifier, gate(), ClaimDecomposer(FakeDecomposer()), "always"
    )
    result = pipeline.verify(claim, "req2")
    assert result.verdict == "MIXED"
    assert retriever.queries == ["A reduces pain", "B causes harm"]
    assert [row.verdict for row in result.component_results] == ["SUPPORTS", "REFUTES"]


def test_supported_and_abstained_components_produce_mixed():
    claim = "A reduces pain and B causes harm"
    retriever = FakeRetriever(weak={"B causes harm"})
    verifier = FakeVerifier({"A reduces pain": "SUPPORTS", "B causes harm": "REFUTES"})
    pipeline = VerificationPipeline(
        retriever, verifier, gate(), ClaimDecomposer(FakeDecomposer()), "always"
    )
    result = pipeline.verify(claim, "req3")
    assert result.verdict == "MIXED"
    assert [row.verdict for row in result.component_results] == ["SUPPORTS", "NOT_ENOUGH_INFO"]
    assert [call[0] for call in verifier.calls] == ["A reduces pain"]


def test_auto_decomposition_failure_falls_back_to_original():
    class Failed:
        def decompose(self, claim, prompt):
            raise RuntimeError("offline")

    claim = "A is safe and B is effective."
    retriever = FakeRetriever()
    verifier = FakeVerifier({claim: "SUPPORTS"})
    result = VerificationPipeline(
        retriever, verifier, gate(), ClaimDecomposer(Failed()), "auto"
    ).verify(claim, "req4")
    assert result.verdict == "SUPPORTS"
    assert result.technical_metadata["decomposition_warnings"]
    assert retriever.queries == [claim]
