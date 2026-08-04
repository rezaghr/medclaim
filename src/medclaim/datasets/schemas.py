"""Canonical record shapes used by dataset adapters."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class EvidenceSet(TypedDict):
    evidence_set_id: str
    relationship: str
    document_id: str
    sentence_indices: list[int]


class ClaimMetadata(TypedDict):
    cited_document_ids: list[str]


class CanonicalClaim(TypedDict):
    claim_id: str
    dataset: str
    source_claim_id: str
    claim_text: str
    original_split: str
    original_label: str | None
    unified_label: str | None
    language: str
    evidence_sets: list[EvidenceSet]
    metadata: ClaimMetadata


class DocumentMetadata(TypedDict):
    structured: NotRequired[bool]


class CanonicalDocument(TypedDict):
    document_id: str
    dataset: str
    source_document_id: str
    title: str | None
    source_type: str
    abstract_sentences: list[str]
    text: str
    metadata: DocumentMetadata
