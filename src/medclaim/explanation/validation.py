"""Deterministic explanation and evidence-use validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED"}
ADVICE_PATTERNS = (
    r"\byou should take\b",
    r"\bstop taking\b",
    r"\brecommended dosage\b",
    r"\byou have\b",
    r"\byou are suffering from\b",
)
BOILERPLATE = {"based on the evidence", "according to the evidence", "see evidence"}


class ExplanationValidationError(Exception):
    """Raised when an explanation cannot be rendered safely."""


@dataclass(frozen=True)
class ExplanationValidationResult:
    valid: bool
    checks: dict[str, bool]
    warnings: list[str]
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExplanationValidator:
    def __init__(self, minimum_words: int = 5, maximum_words: int = 120) -> None:
        if (
            not isinstance(minimum_words, int)
            or isinstance(minimum_words, bool)
            or not isinstance(maximum_words, int)
            or isinstance(maximum_words, bool)
            or minimum_words < 1
            or maximum_words < minimum_words
        ):
            raise ExplanationValidationError(
                "EXPLANATION_CONFIG_INVALID: Word limits are invalid."
            )
        self.minimum_words = minimum_words
        self.maximum_words = maximum_words

    def validate(
        self,
        result: dict[str, Any],
        supplied_passages: list[dict[str, Any]],
        *,
        gate_abstained: bool = False,
    ) -> ExplanationValidationResult:
        if not isinstance(result, dict) or result.get("verdict") not in VERDICTS:
            raise ExplanationValidationError(
                "EXPLANATION_INVALID_VERDICT: Result has no supported verdict."
            )
        explanation = result.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ExplanationValidationError(
                "EXPLANATION_EMPTY: Explanation must be non-empty."
            )
        explanation = explanation.strip()
        normalized = re.sub(r"\s+", " ", explanation.casefold()).strip(" .")
        if normalized in BOILERPLATE:
            raise ExplanationValidationError(
                "EXPLANATION_EMPTY: Boilerplate-only explanation is invalid."
            )
        words = explanation.split()
        if len(words) < self.minimum_words or len(words) > self.maximum_words:
            raise ExplanationValidationError(
                "EXPLANATION_LENGTH_INVALID: Explanation contains "
                f"{len(words)} words; expected {self.minimum_words} through "
                f"{self.maximum_words}."
            )
        supplied_ids: list[str] = []
        for passage in supplied_passages:
            passage_id = passage.get("passage_id") if isinstance(passage, dict) else None
            if not isinstance(passage_id, str) or not passage_id or passage_id in supplied_ids:
                raise ExplanationValidationError(
                    "EXPLANATION_SUPPLIED_EVIDENCE_INVALID: Supplied passage IDs must be unique."
                )
            supplied_ids.append(passage_id)
        evidence_used = result.get("evidence_used")
        if not isinstance(evidence_used, list) or any(
            not isinstance(value, str) for value in evidence_used
        ):
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_INVALID: evidence_used must be a list of IDs."
            )
        limitations = result.get("limitations", [])
        if not isinstance(limitations, list) or any(
            not isinstance(value, str) for value in limitations
        ):
            raise ExplanationValidationError(
                "EXPLANATION_LIMITATIONS_INVALID: limitations must be a list of strings."
            )
        if len(evidence_used) != len(set(evidence_used)):
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_INVALID: Duplicate evidence IDs are not allowed."
            )
        unknown = set(evidence_used) - set(supplied_ids)
        if unknown:
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_INVALID: Explanation cites unsupplied passage "
                f"{sorted(unknown)[0]!r}."
            )
        verdict = result["verdict"]
        if verdict != "NOT_ENOUGH_INFO" and not evidence_used:
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_COVERAGE: Non-NEI results must cite evidence."
            )
        if verdict == "NOT_ENOUGH_INFO" and not gate_abstained and not evidence_used:
            # Verifier-generated NEI may cite selected evidence; a gate abstention may not.
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_COVERAGE: Verifier-generated NEI must cite selected evidence."
            )
        referenced_numbers = [
            int(value) for value in re.findall(r"\bEvidence\s+(\d+)\b", explanation, re.IGNORECASE)
        ]
        if any(value < 1 or value > len(supplied_ids) for value in referenced_numbers):
            raise ExplanationValidationError(
                "EXPLANATION_CITATION_INVALID: Explanation references an unavailable evidence number."
            )
        lower = explanation.casefold()
        if any(re.search(pattern, lower) for pattern in ADVICE_PATTERNS):
            raise ExplanationValidationError(
                "EXPLANATION_MEDICAL_ADVICE: Personalized medical advice is prohibited."
            )
        self._validate_verdict_language(verdict, lower, result)
        if re.search(r"\bcaus(?:e|es|ed|ing)\b", lower):
            evidence_text = " ".join(str(item.get("text", "")) for item in supplied_passages).casefold()
            if not re.search(r"\bcaus(?:e|es|ed|ing|al)\b", evidence_text):
                raise ExplanationValidationError(
                    "EXPLANATION_CAUSAL_OVERSTATEMENT: Causal language is unsupported by supplied evidence."
                )
        checks = {
            "citation_validity": True,
            "citation_coverage": bool(evidence_used) or (verdict == "NOT_ENOUGH_INFO" and gate_abstained),
            "label_consistency": True,
            "length_valid": True,
            "medical_safety_compliance": True,
            "concise": len(words) <= self.maximum_words,
            "non_empty": True,
        }
        return ExplanationValidationResult(True, checks, [], len(words))

    @staticmethod
    def _validate_verdict_language(
        verdict: str, lower: str, result: dict[str, Any]
    ) -> None:
        if verdict == "SUPPORTS" and re.search(r"\b(contradicts?|refutes?)\b", lower):
            raise ExplanationValidationError(
                "EXPLANATION_VERDICT_CONFLICT: SUPPORTS explanation uses contradiction wording."
            )
        if verdict == "REFUTES" and re.search(r"\b(directly supports?|supports the claim)\b", lower):
            raise ExplanationValidationError(
                "EXPLANATION_VERDICT_CONFLICT: REFUTES explanation uses support wording."
            )
        if verdict == "NOT_ENOUGH_INFO" and not re.search(
            r"\b(insufficient|not enough|not sufficiently|cannot|could not|"
            r"does not establish|does not provide|does not determine|unable|"
            r"does not conclusively|not conclusive|not consistent in establishing|"
            r"unclear|uncertain|inconclusive|undetermined|no definitive)\b",
            lower,
        ):
            raise ExplanationValidationError(
                "EXPLANATION_VERDICT_CONFLICT: NEI explanation must communicate insufficiency."
            )
        if verdict == "MIXED":
            components = result.get("component_results")
            component_verdicts = {
                item.get("verdict") for item in components or [] if isinstance(item, dict)
            }
            if len(component_verdicts) < 2:
                raise ExplanationValidationError(
                    "EXPLANATION_MIXED_INVALID: MIXED requires disagreeing component verdicts."
                )
            if not re.search(r"\b(mixed|different|conflict|partly|components?)\b", lower):
                raise ExplanationValidationError(
                    "EXPLANATION_VERDICT_CONFLICT: MIXED explanation must mention differing outcomes."
                )


def validate_with_one_correction(
    verifier: Any,
    claim: str,
    supplied_passages: list[dict[str, Any]],
    validator: ExplanationValidator,
    *,
    implementation: str = "llm",
) -> dict[str, Any]:
    """Validate verifier output and allow one explicit LLM correction call."""
    result = verifier.verify(claim, supplied_passages)
    try:
        validator.validate(result, supplied_passages)
        return result
    except ExplanationValidationError as first_error:
        if implementation != "llm" or not callable(getattr(verifier, "correct", None)):
            raise
        corrected = verifier.correct(
            claim, supplied_passages, result, str(first_error)
        )
        validator.validate(corrected, supplied_passages)
        return corrected
