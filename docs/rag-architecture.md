# MedClaimRAG Architecture and Algorithms

MedClaimRAG is a modular, file-backed hybrid RAG system. The default deployment
combines BM25 lexical retrieval with dense semantic retrieval, fuses both result
lists, reranks the candidates for direct claim relevance, and verifies the final
evidence with local Ollama models. It does not require a hosted model or vector
database service.

## Live request path

```text
Streamlit UI
    |
    v
FastAPI POST /v1/verify
    |
    v
Medical scope router
    |
    v
Claim decomposition check
    |
    v
Parallel retrieval over 126,855 corpus passages
    |-- BM25 lexical search
    `-- Nomic dense semantic search through Ollama + FAISS
    |
    v
Reciprocal Rank Fusion (30 candidates)
    |
    v
Ollama direct-relevance reranking
    |
    v
Relevance gate (score >= 0.75, at most 5 passages)
    |
    v
Evidence text inserted into a secure prompt
    |
    v
Local Ollama: dolphin-llama3:8b
    |
    v
Schema, citation, and explanation validation
    |
    v
Component aggregation
    |
    v
Verdict, confidence, cited evidence, and retrieval trace
```

The default runtime configuration is in
[`configs/deployment/default.yaml`](../configs/deployment/default.yaml).

## Validated default configuration

This document reflects the locally validated deployment as of 2026-08-04:

| Component | Active configuration |
| --- | --- |
| Corpus | `medical-corpus-v2` — 17,756 documents and 126,855 passages |
| Sparse index | `medical-bm25-v2` — BM25Okapi |
| Dense index | `medical-dense-v2` — local FAISS `IndexFlatIP` |
| Embedding model | Ollama `nomic-embed-text:latest`, 768 dimensions |
| Document/query prefixes | `search_document: ` / `search_query: ` |
| Retrieval mode | `hybrid_reranked` |
| RRF candidate count | `30` |
| Reranker | Ollama `dolphin-llama3:8b`, candidates scored independently |
| Reranker batch size | `1` |
| Gate | `reranker_score >= 0.75` |
| Maximum verifier evidence | `5` passages |
| Final verifier | Ollama `dolphin-llama3:8b`, temperature `0` |
| SQL persistence | Disabled |

## 1. User-interface layer

The Streamlit UI sends a claim to the API as JSON:

```http
POST /v1/verify
Content-Type: application/json

{"claim": "..."}
```

It displays:

- The verdict and explanation.
- Raw model confidence.
- Evidence cited by the model.
- Every retrieved passage.
- BM25, dense, RRF, and reranker scores when available.
- Pre-rerank and reranker positions.
- Hybrid-retrieval and reranking latency.
- Whether a passage was sent to Ollama.
- Whether Ollama cited a passage.
- Raw retrieval and model-input JSON.

Implementation: [`app/streamlit_app.py`](../app/streamlit_app.py).

## 2. API and service layer

FastAPI validates the request, generates a request ID, and calls the
`VerificationService`. The service records:

- A SHA-256 hash of the claim instead of logging its raw text.
- Request, retrieval, and verification timings.
- Verdict counts.
- Provider failures and timeouts.
- Evidence-gate abstentions.

Implementations:

- [`src/medclaim/api/app.py`](../src/medclaim/api/app.py)
- [`src/medclaim/runtime/service.py`](../src/medclaim/runtime/service.py)

## 3. Scope-routing layer

Before retrieval, a deterministic rule-based classifier examines the claim.
"Deterministic" means that this layer uses predefined regular expressions and
vocabulary rather than asking an LLM.

It recognizes:

- General medical terminology.
- Biomedical terminology such as proteins, neutrophils, citrullination, and
  inflammation.
- Public-health terminology.
- Food-toxicity and health-hazard claims.
- Personal diagnosis, treatment, dosage, and medication-change requests.
- Personal emergencies.

Medical and public-health claims continue to retrieval. Personal diagnosis,
treatment, dosage, medication-change, and emergency requests are limited.

Implementation: [`src/medclaim/safety/scope.py`](../src/medclaim/safety/scope.py).

## 4. Claim decomposition layer

The decomposition algorithm detects potentially compound claims using signals
such as:

- Semicolons separating clauses.
- `and` or `but` where both sides appear to contain predicates.

The architecture supports splitting a compound claim into at most four atomic
claims, independently verifying them, and aggregating their results.

However, the current live pipeline constructs `ClaimDecomposer(None)`. In
automatic mode, simple claims remain unchanged. A potential compound claim
causes an attempted decomposition, but because no decomposition provider is
configured, the complete original claim is used as a fallback and a warning is
recorded. The default deployment therefore does not currently perform real
LLM-based decomposition.

Implementation:
[`src/medclaim/decomposition/decomposer.py`](../src/medclaim/decomposition/decomposer.py).

## 5. Corpus and database layer

The RAG knowledge base is not PostgreSQL and does not depend on a vector-database
service. It uses immutable corpus files plus local BM25 and FAISS index files:

```text
artifacts/corpora/medical-corpus-v2/
|-- manifest.json
|-- documents.jsonl
|-- passages.jsonl
|-- gold_evidence.jsonl
`-- quality_report.json
```

Current corpus statistics:

| Property | Value |
| --- | ---: |
| Datasets | SciFact, HealthVer, PubHealth |
| Documents | 17,756 |
| Passages | 126,855 |
| Claims | 15,513 |
| Gold evidence sets | 19,592 |
| Maximum passage size | 120 words |

Documents are divided into smaller passages because focused sections are more
useful for retrieval than entire long documents.

Artifact description:
[`artifacts/corpora/medical-corpus-v2/manifest.json`](../artifacts/corpora/medical-corpus-v2/manifest.json).

### Important distinction: claims versus evidence

Dataset claims, labels, and gold-evidence mappings are available for evaluation,
but online retrieval indexes only the passage `text` field. The runtime does not
perform this exact lookup:

```text
claim text -> stored gold label
```

It performs this RAG process instead:

```text
claim text -> retrieve similar evidence passages -> ask Ollama to judge them
```

Consequently, a claim being present in a source dataset does not automatically
cause its stored dataset label to be returned.

## 6. BM25 retrieval layer

BM25 means **Best Matching 25**. It is a traditional sparse text-search ranking
algorithm. "Sparse" means it matches words and numbers directly; it does not
turn sentences into neural embedding vectors.

### Tokenization

Corpus passages and user claims are processed using:

1. Unicode NFKC normalization.
2. Lowercasing.
3. Extraction of ASCII letters and numbers.

For example:

```text
"Aspirin prevents Heart-Attacks!"
```

becomes approximately:

```text
["aspirin", "prevents", "heart", "attacks"]
```

There is currently no stemming, synonym expansion, or stop-word removal. For
example, `prevent`, `prevents`, and `prevented` remain different tokens.

Implementation:
[`src/medclaim/retrieval/tokenization.py`](../src/medclaim/retrieval/tokenization.py).

### BM25 scoring

For each query word, BM25 considers:

- How frequently it occurs in a passage.
- How rare it is across the corpus, called inverse document frequency or IDF.
- Passage length relative to the average passage length.
- Diminishing returns for repeated occurrences.

A simplified formula is:

```text
score(passage, query) =
    sum over query words of:

                 frequency * (k1 + 1)
    IDF(word) * ---------------------------
                 frequency + k1 * length-normalization
```

The current parameters are:

- `k1 = 1.5`: controls how quickly repeated occurrences stop adding value.
- `b = 0.75`: controls passage-length normalization.
- `epsilon = 0.25`: controls how the library handles problematic negative IDF
  values.

For example, a rare term such as `citrullinated` usually contributes more than a
common term such as `proteins`. A passage containing several relevant rare terms
ranks highly, while length normalization prevents long passages from winning
only because they contain more words.

BM25 scores all 126,855 passages and contributes its highest-ranked candidates
to the hybrid candidate pool. BM25 is no longer the only live retriever.

Implementation: [`src/medclaim/retrieval/bm25.py`](../src/medclaim/retrieval/bm25.py).

The serialized index is stored as:

```text
artifacts/indexes/medical-bm25-v2/
|-- index.pkl
|-- passage_ids.json
`-- manifest.json
```

The application loads this index into memory at startup. Manifests, corpus
hashes, passage mappings, and file checksums ensure that the index matches the
corpus from which it was built.

## 7. Dense semantic retrieval layer

Dense retrieval handles relationships that do not depend on exact word overlap.
The local Ollama model `nomic-embed-text:latest` converts each passage and claim
into a 768-dimensional numerical vector. Texts with similar meanings tend to
produce vectors pointing in similar directions, even when they use different
words.

Corpus passages use the model's `search_document: ` task prefix, while incoming
claims use `search_query: `. The prefixes tell the embedding model which side of
the retrieval task it is encoding and are recorded in the dense-index manifest.

All vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index. Because
the vectors have unit length, inner-product search is equivalent to cosine
similarity:

```text
cosine_similarity(query, passage) = query_vector dot passage_vector
```

The dense index is stored locally:

```text
artifacts/indexes/medical-dense-v2/
|-- index.faiss
|-- embeddings.npy
|-- passage_ids.json
`-- manifest.json
```

The corpus and query embeddings use the same locally installed Nomic model.
Index manifests and checksums prevent the runtime from combining an index with
the wrong corpus or embedding model.

Implementations:

- [`src/medclaim/retrieval/embedding.py`](../src/medclaim/retrieval/embedding.py)
- [`src/medclaim/retrieval/dense.py`](../src/medclaim/retrieval/dense.py)

## 8. Hybrid fusion and reranking

BM25 and dense retrieval run independently. Reciprocal Rank Fusion (RRF) merges
their rankings without trying to compare a BM25 score with a cosine-similarity
score directly:

```text
RRF score(passage) = sum over result lists of 1 / (60 + rank)
```

A passage appearing near the top of both lists receives a higher fused score.
The default runtime sends 30 fused candidates to the reranker.

The reranker is a separate relevance decision from retrieval. It sends the claim
and candidate texts to the local `dolphin-llama3:8b` model and requests one
normalized relevance score per passage:

- `0.00-0.20`: incidental overlap or unrelated.
- `0.21-0.49`: same broad topic but not the claimed relationship or outcome.
- `0.50-0.69`: partially relevant but missing a material part of the claim.
- `0.70-1.00`: directly capable of supporting or refuting the complete claim.

Generic mention of the same drug, disease, or treatment is explicitly scored as
insufficient. The reranker preserves BM25, dense, and RRF scores while adding a
separate `reranker_score` and `pre_rerank_rank`. Its latency is reported under
`retrieval_metadata.latency_ms.reranking`.

Candidates are scored independently (`reranker_batch_size: 1`). This avoids
cross-passage score contamination observed with the local 8B model and ensures
that every one of the 30 candidates receives a complete structured decision.

Implementations:

- [`src/medclaim/retrieval/hybrid.py`](../src/medclaim/retrieval/hybrid.py)
- [`src/medclaim/retrieval/reranked.py`](../src/medclaim/retrieval/reranked.py)
- [`src/medclaim/reranking/ollama.py`](../src/medclaim/reranking/ollama.py)

## 9. Evidence gate

The evidence gate decides whether retrieved passages are strong enough to send
to the verifier. It supports checking:

- The top retrieval score.
- The number of relevant passages.
- The number of unique source documents.
- The maximum number of passages sent to the model.

The current live configuration is:

| Setting | Value |
| --- | --- |
| Score field | `reranker_score` |
| Minimum score | `0.75` |
| Minimum relevant passages | `1` |
| Minimum unique documents | `1` |
| Hybrid candidates before reranking | `30` |
| Maximum evidence passages | `5` |

The gate now consumes the reranker's normalized relevance score rather than a
BM25 value. Its result contains both `score_field` and `top_score`; it no longer
labels a BM25 score as `top_reranker_score`. Passing the gate means only that the
configured relevance requirements were met, so the reason is
`RELEVANCE_REQUIREMENTS_MET` rather than `SUFFICIENT_EVIDENCE`.

The `0.75` threshold is a conservative policy threshold aligned with the
reranker's scoring rubric. It is not claimed to be a statistically calibrated
probability. If no candidate reaches it, the pipeline returns `NOT_ENOUGH_INFO`
without running the final verification call.

### Tamoxifen regression result

The end-to-end regression claim is:

```text
A breast cancer patient's capacity to metabolize tamoxifen influences treatment outcome.
```

Under the former BM25-only path, the highest score (`30.6668`) belonged to a
generic tamoxifen-treatment passage. In the corrected path:

1. Dense retrieval recovers the CYP2D6 outcome evidence.
2. Hybrid RRF includes it in the 30-passage candidate pool.
3. Independent reranking assigns the direct passage `0.80`.
4. The `0.75` gate excludes generic passages scored `0.70` or lower.
5. Only `scifact:document:24341590:p:13` is sent to the verifier.
6. The final verdict is `SUPPORTS`, citing that passage.

The selected text directly states that CYP2D6 functional variation is
associated with better or worse clinical outcomes among women treated with
tamoxifen. This regression verifies retrieval relevance, real reranker output,
gate score provenance, and final evidence attribution together.

Implementation:
[`src/medclaim/evidence_gate/gate.py`](../src/medclaim/evidence_gate/gate.py).

## 10. Ollama verification layer

Selected passages are reduced to the following fields:

```json
{
  "passage_id": "...",
  "text": "complete passage text"
}
```

They are inserted into XML-style evidence blocks and sent to Ollama using:

```text
Endpoint: http://127.0.0.1:11434/api/generate
Model: dolphin-llama3:8b
Temperature: 0
Streaming: false
Structured JSON output: enabled
```

The model must return:

```json
{
  "verdict": "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
  "confidence": 0.0,
  "explanation": "...",
  "evidence_used": ["passage-id"],
  "limitations": []
}
```

The secure verifier instructs the model to use only supplied evidence. The model
has no tools, database access, web access, or file access.

Implementations:

- [`src/medclaim/runtime/ollama.py`](../src/medclaim/runtime/ollama.py)
- [`src/medclaim/security/verifier.py`](../src/medclaim/security/verifier.py)

### Absolute-claim second pass

Claims containing terms such as `all`, `always`, `never`, or `guaranteed` receive
special handling. If Ollama initially returns `NOT_ENOUGH_INFO`, the verifier:

1. Scores retrieved passages for counterexample signals.
2. Selects at most three strong counterexample candidates.
3. Calls Ollama a second time.
4. Explicitly asks whether an observed exception refutes the absolute claim.

For example, a claim that aspirin prevents *all* heart attacks is refuted when
the evidence establishes even one residual heart attack among aspirin users.

## 11. Output-validation layer

The result is rejected or sent through one correction attempt if it:

- Is not valid JSON.
- Uses an unknown verdict.
- Has confidence outside the range `0` to `1`.
- Cites passage IDs that were not supplied to Ollama.
- Contains an empty or invalid-length explanation.
- Contains prohibited medical-advice language.
- Requests tools or exposes prompt- or secret-like content.
- Makes causal statements unsupported by the evidence text.

## 12. Aggregation layer

Each atomic component is independently verified. Aggregation is deterministic:

- If every component has the same verdict, that becomes the final verdict.
- If component verdicts differ, the final verdict is `MIXED`.
- Overall confidence is the minimum component confidence.
- Evidence IDs are combined and deduplicated.
- Compound-claim explanations are constructed from component outcomes.

Implementation:
[`src/medclaim/verification/aggregation.py`](../src/medclaim/verification/aggregation.py).

## Active and alternative algorithms

| Algorithm | Meaning and use | Default state |
| --- | --- | --- |
| BM25 retrieval | Finds passages with strong exact-term overlap. | Active. |
| Dense retrieval | Converts claims and passages into Nomic vectors and searches by semantic similarity. | Active. |
| Hybrid retrieval | Runs BM25 and dense retrieval and combines their rankings. | Active. |
| Reciprocal Rank Fusion (RRF) | Combines rankings using `1 / (60 + rank)` without mixing incomparable raw scores. | Active. |
| Ollama relevance reranking | Reads the claim and each candidate and assigns a separate direct-relevance score. | Active. |
| Cross-encoder reranking | A transformer reads each claim-passage pair together and predicts relevance. | Implemented as an alternative, but inactive by default. |
| Confidence calibration | Learns how raw confidence relates to observed correctness. | A version name is configured, but live output is not calibrated. |

The confidence displayed by the UI is currently raw confidence generated by
`dolphin-llama3:8b`; it is not a clinical probability or a statistically
calibrated correctness probability.

## PostgreSQL status

[`docker-compose.yml`](../docker-compose.yml) defines an optional PostgreSQL 17
container under the `persistence` profile. However:

- Persistence is disabled in the default configuration.
- Claim-text persistence is disabled.
- Explanation persistence is disabled.
- No PostgreSQL repository or SQL read/write implementation is connected to the
  verification request path.
- PostgreSQL is not used for RAG retrieval.

The storage architecture can therefore be summarized as:

```text
RAG knowledge storage       = JSONL corpus + BM25 index + local FAISS dense index
Application-result database = currently none
Optional PostgreSQL         = declared but not implemented into verification
LLM/embedding runtime       = local Ollama
```

## Validation and operational characteristics

The corrected runtime has been checked with:

- `375` automated tests passing.
- Ruff static checks passing.
- Deployment readiness reporting `ready` for corpus, BM25 index, dense index,
  Ollama models, evidence gate, and persistence configuration.
- An end-to-end tamoxifen regression returning `SUPPORTS` and citing only
  `scifact:document:24341590:p:13` after the `0.75` gate.
- Dense-index checksum and corpus-content-hash validation.
- Markdown link validation for this document.

The local dense artifact occupies approximately 748 MB. Independent Ollama
reranking improves relevance consistency but is the main latency cost: the
tamoxifen regression spent approximately 45 seconds reranking 30 candidates on
the tested laptop. Hybrid retrieval itself took approximately 0.5 seconds. This
is an explicit correctness/latency tradeoff; the runtime does not silently skip
reranking or fall back to BM25-only retrieval when reranking fails.

After changing the deployment configuration or rebuilding an index, restart the
API and Streamlit processes so the runtime loads the new immutable artifacts.
