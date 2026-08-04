import pytest

from medclaim.decomposition.decomposer import ClaimDecomposer, DecompositionError


class UnsafeProvider:
    def decompose(self, claim, prompt):
        return {
            "is_compound": True,
            "atomic_claims": [
                {"index": 1, "text": "Aspirin cures cancer", "source_span": "Aspirin helps"},
                {"index": 2, "text": "Follow treatment advice", "source_span": "is affordable"},
            ],
        }


def test_decomposer_rejects_added_medical_facts_and_treatment_instruction():
    with pytest.raises(DecompositionError, match="MEANING_NOT_PRESERVED"):
        ClaimDecomposer(UnsafeProvider()).decompose("Aspirin helps and is affordable.", "always")
