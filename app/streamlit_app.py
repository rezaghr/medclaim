"""Streamlit client that calls the API without receiving provider secrets."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st

from medclaim.ui import collect_used_evidence

API_TIMEOUT_SECONDS = float(os.environ.get("MEDCLAIM_API_TIMEOUT_SECONDS", "300"))


def open_internal_api(
    request: urllib.request.Request, timeout: float = API_TIMEOUT_SECONDS
):
    """Call the trusted MedClaim API without inheriting workstation proxy settings."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def render_used_evidence(verification: dict) -> None:
    """Show model-cited evidence as collapsed, individually copyable passages."""
    evidence = collect_used_evidence(verification)
    st.subheader("Evidence used for this verdict")
    if not evidence:
        st.info("The model did not cite any corpus passages for this verdict.")
        return

    st.caption(
        "Expand a passage to read its complete text. Use the copy icon in the "
        "top-right corner of the text box to copy it."
    )
    missing_text = False
    for index, record in enumerate(evidence, start=1):
        passage_id = str(record.get("passage_id", "unknown passage"))
        dataset = record.get("dataset")
        label = f"Evidence {index} · {passage_id}"
        if dataset:
            label += f" · {dataset}"
        with st.expander(label, expanded=False):
            document_id = record.get("document_id")
            rank = record.get("rank")
            details = []
            if document_id:
                details.append(f"Document: {document_id}")
            if rank is not None:
                details.append(f"Retrieval rank: {rank}")
            if details:
                st.caption(" · ".join(details))
            text = record.get("text")
            if isinstance(text, str) and text:
                st.code(text, language=None, wrap_lines=True)
            else:
                missing_text = True
                st.warning("Passage text was not included by the API process.")
    if missing_text:
        st.warning(
            "The API is returning the old ID-only response. Rebuild and restart both "
            "the api and streamlit services to expose passage text."
        )


def render_retrieval_trace(verification: dict) -> None:
    """Render the database retrieval and the exact evidence context sent to Ollama."""
    components = verification.get("component_results", [])
    if not isinstance(components, list) or not components:
        st.info("No retrieval trace is available for this response.")
        return

    st.subheader("Retrieved evidence sent to the AI")
    st.caption(
        "These are the exact corpus passages retrieved from the indexed database. "
        "‘Sent to Ollama’ identifies the passage text included in the model prompt; "
        "‘cited’ identifies passages named in the model response."
    )
    for component_index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            continue
        claim_text = str(component.get("claim", ""))
        label = (
            f"Component {component_index}: {claim_text}"
            if claim_text
            else f"Component {component_index}"
        )
        with st.container(border=True):
            st.markdown(f"#### {label}")
            retrieval_metadata = component.get("retrieval_metadata", {})
            if isinstance(retrieval_metadata, dict) and retrieval_metadata:
                columns = st.columns(5)
                columns[0].metric("Mode", retrieval_metadata.get("retrieval_mode", "—"))
                columns[1].metric("Retrieved", retrieval_metadata.get("returned_count", 0))
                latency = retrieval_metadata.get("latency_ms")
                total_latency = latency.get("total") if isinstance(latency, dict) else latency
                reranking_latency = latency.get("reranking") if isinstance(latency, dict) else None
                columns[2].metric(
                    "Retrieval latency",
                    f"{total_latency:.1f} ms" if isinstance(total_latency, (int, float)) else "—",
                )
                columns[3].metric(
                    "Reranking latency",
                    f"{reranking_latency:.1f} ms"
                    if isinstance(reranking_latency, (int, float))
                    else "—",
                )
                columns[4].metric(
                    "Corpus",
                    retrieval_metadata.get("corpus_version", "—"),
                )

            gate = component.get("gate_decision", {})
            if isinstance(gate, dict) and gate:
                top_score = gate.get("top_score")
                st.caption(
                    f"Evidence gate: {gate.get('status', 'unknown')} · "
                    f"score={gate.get('score_field', 'unknown')} · "
                    f"top={top_score:.3f} · threshold={gate.get('threshold')}"
                    if isinstance(top_score, (int, float))
                    else f"Evidence gate: {gate.get('status', 'unknown')}"
                )

            candidates = component.get("retrieved_candidates", [])
            model_input = component.get("model_input_evidence", [])
            sent_ids = {row.get("passage_id") for row in model_input if isinstance(row, dict)}
            cited_ids = set(component.get("evidence_used", []))
            if not isinstance(candidates, list) or not candidates:
                st.info("The retriever returned no candidates for this component.")
                continue

            for fallback_rank, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, dict):
                    continue
                passage_id = str(candidate.get("passage_id", "unknown passage"))
                rank = candidate.get("rank", fallback_rank)
                reranker_rank = candidate.get("reranker_rank")
                dataset = candidate.get("dataset", "unknown dataset")
                score_items = [
                    f"{key}={value:.4f}"
                    for key in ("bm25_score", "dense_score", "rrf_score", "reranker_score")
                    if isinstance((value := candidate.get(key)), (int, float))
                ]
                status = []
                if passage_id in sent_ids:
                    status.append("sent to Ollama")
                if passage_id in cited_ids:
                    status.append("cited by model")
                status_text = " · ".join(status) if status else "not sent"
                title = f"#{rank} · {passage_id} · {dataset} · {status_text}"
                with st.expander(title, expanded=fallback_rank == 1):
                    st.caption(
                        f"Document: {candidate.get('document_id', 'unknown')}"
                        + (
                            f" · Reranker rank: {reranker_rank}"
                            if isinstance(reranker_rank, int)
                            else ""
                        )
                        + (f" · {' · '.join(score_items)}" if score_items else "")
                    )
                    st.write(candidate.get("text", ""))

            with st.expander("Raw retrieval and exact model-input JSON"):
                st.json(
                    {
                        "retrieval_metadata": retrieval_metadata,
                        "retrieved_candidates": candidates,
                        "model_input_evidence": model_input,
                    }
                )


st.set_page_config(page_title="MedClaimRAG", page_icon="🔎")
st.title("MedClaimRAG")
claim = st.text_area("Textual medical or public-health claim", max_chars=5000)
if st.button("Verify claim", type="primary"):
    request = urllib.request.Request(
        os.environ.get("MEDCLAIM_API_URL", "http://api:8000").rstrip("/") + "/v1/verify",
        data=json.dumps({"claim": claim}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with st.spinner("Retrieving and verifying evidence..."):
            with open_internal_api(request) as response:
                result = json.load(response)
        decision = result.get("scope_decision", {})
        if decision.get("action") != "VERIFY":
            st.error(decision.get("message", "This request cannot be verified."))
        else:
            verification = result.get("verification", {})
            st.subheader(str(verification.get("verdict", "No verdict")))
            st.write(verification.get("explanation", ""))
            confidence = verification.get("confidence")
            if isinstance(confidence, (int, float)):
                st.metric("Model confidence", f"{confidence:.0%}")
            st.caption("Confidence is a model estimate, not a clinical probability.")
            render_used_evidence(verification)
            render_retrieval_trace(verification)
            with st.expander("Request and full verification response"):
                st.json(
                    {
                        "request_id": result.get("request_id"),
                        "verification": verification,
                    }
                )
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode())
            detail = error.get("detail", f"HTTP {exc.code}")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = f"HTTP {exc.code}"
        if detail == "PIPELINE_UNAVAILABLE":
            st.error(
                "The evidence corpus and retrieval indexes are not loaded. "
                "Ollama is connected, but verification cannot run without indexed evidence."
            )
        else:
            st.error(f"The verification API rejected the request: {detail}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.error(f"The verification API is unavailable: {exc}")
