# MedClaimRAG: A Codebase-Specific Technical Study Guide

This guide is a course in the engineering ideas behind the current MedClaimRAG
repository. It is deliberately not API documentation. The goal is to close the
gap between being able to run the project and understanding why it behaves as it
does, what guarantees it actually provides, and where those guarantees stop.

The repository described here is the compact Docker-only version. It has no
PostgreSQL database, ORM, message broker, background worker, or separate vector
database. Its durable runtime data consists of files in a Docker named volume.

## 1. The system in one mental model

MedClaimRAG is a synchronous, evidence-bounded claim-verification pipeline:

```text
Browser
  -> Streamlit UI
  -> FastAPI POST /v1/verify
  -> safety routing
  -> optional claim decomposition
  -> BM25 retrieval ---------+
  -> dense FAISS retrieval --+-> reciprocal-rank fusion
                                -> Ollama reranking
                                -> evidence-sufficiency gate
                                -> Ollama verification
                                -> deterministic validation and aggregation
  <- structured JSON result
```

The most important distinction is between retrieval and verification:

- Retrieval asks, "Which indexed passages may be relevant?"
- Verification asks, "Given only these passages, do they support, refute, or
  fail to establish the claim?"

Retrieval relevance is not truth. A passage can be highly relevant because it
directly contradicts a claim. That is exactly what should happen for the claim
"Antibiotics effectively treat influenza": the antibiotic passage should rank
highly, and the verifier should classify the relationship as `REFUTES`.

The system is a form of retrieval-augmented generation, or RAG. The "retrieval"
part supplies a bounded context. The "generation" part is the local Ollama model
producing a structured verdict and explanation. RAG reduces dependence on model
memory, but it does not make the output automatically correct. Every boundary
still needs validation.

### The exact default pipeline and its cardinalities

The current Docker configuration is not an abstract “hybrid RAG” label. It is
this concrete sequence:

```text
1 input claim
  -> safety rules
  -> normally 1 atomic component
  -> BM25 scans 101,491 passages and returns 50
  -> FAISS searches 101,491 vectors and returns 50
  -> RRF fuses the two lists and retains 30
  -> Ollama reranker scores 30 in batches of 9
  -> reranker retains the best 5
  -> evidence gate keeps only passages scoring >= 0.70
  -> Ollama verifier returns verdict + confidence + citations
  -> application validators accept, correct once, or fail
  -> component aggregation produces the API result
```

The word “score” is overloaded, so keep these four quantities separate:

| Quantity | Produced by | Meaning | Used as probability? |
|---|---|---|---|
| `bm25_score` | BM25 | Lexical term-match strength | No |
| `dense_score` | FAISS | Cosine similarity between embeddings | No |
| `rrf_score` | RRF | Agreement of rank positions | No |
| `reranker_score` | Ollama reranker | Uncalibrated direct-relevance judgment | No |
| `confidence` | Ollama verifier | Uncalibrated confidence in its evidence-bound verdict | No |

Only `reranker_score` is compared with the current evidence threshold. Verdict
`confidence` is displayed to the user but is not used to decide whether the gate
opens. A high dense similarity cannot directly become high verdict confidence,
and a high relevance score does not imply that the claim is true: relevant
evidence may refute it.

### Current parameters at a glance

| Stage | Parameter | Current value | Effect |
|---|---|---:|---|
| Chunking | maximum words | 120 | Upper bound based on whitespace-separated words |
| Chunking | overlap | 0 | Adjacent passages do not repeat text intentionally |
| BM25 | `k1` | 1.5 | Controls term-frequency saturation |
| BM25 | `b` | 0.75 | Controls passage-length normalization |
| BM25 | `epsilon` | 0.25 | Floors negative IDF values in `rank-bm25` |
| Hybrid | sparse candidates | 50 | BM25 results entering fusion |
| Hybrid | dense candidates | 50 | FAISS results entering fusion |
| RRF | `k` | 60 | Dampens rank-position differences |
| Reranking | candidate count | 30 | Fused passages scored by the LLM |
| Reranking | batch size | 9 | Maximum passages per generation call |
| Final evidence | `top_k` | 5 | Maximum reranked passages offered to the gate/verifier |
| Gate | minimum score | 0.70 | Minimum reranker score considered directly relevant |
| Gate | minimum relevant passages | 1 | Required count at or above threshold |
| Gate | minimum unique documents | 1 | Required provenance diversity |
| Embedding build | batch size | 64 | Passages per Ollama embedding request |
| API input | maximum claim characters | 5,000 | Pydantic and embedding-input bound |
| Decomposition | maximum components | 4 | Hard limit if a provider is later connected |
| Explanation | word range | 5–120 | Accepted explanation length |
| Absolute-claim pass | focused passages | 3 | Counterexample candidates in the second verifier pass |
| Ollama | per-call timeout | 120 s runtime, 600 s index build | Bounds one model HTTP call, not the whole pipeline |
| Streamlit | whole-request timeout | 300 s | Bounds waiting for the complete API request |

## 2. Repository map and responsibility boundaries

| Area | Main files | Responsibility |
|---|---|---|
| Container startup | `docker-compose.yml`, `Dockerfile` | Order services, mount artifacts, connect containers to host Ollama |
| Artifact initialization | `scripts/initialize_runtime_artifacts.py` | Download and normalize all three releases, then build BM25/FAISS indexes |
| Dataset normalization | `src/medclaim/corpus/full.py` | Map three source schemas into canonical documents, claims, passages, and evidence sets |
| HTTP API | `src/medclaim/api/app.py` | Validate requests, expose health/metrics/verification endpoints |
| Application service | `src/medclaim/runtime/service.py` | Safety routing, request correlation, metrics, error translation |
| Pipeline assembly | `src/medclaim/runtime/pipeline.py` | Construct and connect retrievers, reranker, gate, and verifier |
| Sparse retrieval | `src/medclaim/retrieval/bm25.py` | Build and query the lexical BM25 index |
| Dense retrieval | `src/medclaim/retrieval/dense.py`, `embedding.py` | Embed text and search the FAISS vector index |
| Hybrid retrieval | `src/medclaim/retrieval/hybrid.py` | Fuse BM25 and dense rankings with RRF |
| Reranking | `src/medclaim/reranking/ollama.py` | Ask Ollama for pointwise relevance scores |
| Evidence gate | `src/medclaim/evidence_gate/gate.py` | Proceed or abstain based on evidence sufficiency |
| Verification security | `src/medclaim/security/verifier.py` | Construct evidence-only prompts and reject invalid output |
| Result validation | `src/medclaim/explanation/validation.py` | Enforce citation, explanation, and safety rules |
| Aggregation | `src/medclaim/verification/aggregation.py` | Combine atomic-claim results deterministically |
| Observability | `src/medclaim/observability/` | Privacy-aware logs and in-memory metrics |
| UI | `app/streamlit_app.py`, `src/medclaim/ui.py` | Call the API and display verdicts and retrieval traces |

This is a modular monolith: the modules have different responsibilities, but
they run in one API process and call one another in memory. The Streamlit UI is a
separate process, but it is only a client of the API.

## 3. Docker startup as a dependency graph

### Problem

The API cannot verify claims before the corpus and both indexes exist. Starting
all containers simultaneously would create a race: the API might load files
while the initialization container is still writing them.

### Intuition

Startup is a graph of prerequisites, not just a list of processes. An edge
`A -> B` means B must reach a required condition before A may start.

### Concept: dependency ordering and readiness

Docker Compose expresses two distinct conditions:

- `service_completed_successfully`: a one-shot job must exit with status zero.
- `service_healthy`: a long-running service must pass its health check.

The current order is:

```text
artifacts-init exits 0
        -> API becomes healthy
                -> Streamlit starts
```

`artifacts-init` mounts `medclaim-artifacts` read-write. The API mounts the same
volume read-only. This is an ownership boundary: initialization may publish
artifacts; online request handling may only read them.

### Liveness versus readiness

Liveness answers, "Is this process running?" The API's `/health/live` always
returns `alive` if FastAPI can answer.

Readiness answers, "Can this process serve a correct request now?"
`/health/ready` checks corpus integrity, index compatibility, embedding metadata,
and installed Ollama models. Compose waits for readiness, not mere liveness.

This distinction matters during partial failure. An API process can be alive but
unable to verify claims because Ollama is stopped or an index checksum fails.

### Docker networking to host Ollama

The API and initializer use:

```text
http://host.docker.internal:11434
```

On Linux, Compose adds `host.docker.internal:host-gateway`, mapping that hostname
inside the container to the Docker host. The application then calls Ollama's
built-in HTTP endpoints directly:

- `GET /api/tags` for model availability
- `POST /api/embed` for embeddings
- `POST /api/generate` for reranking and verification

There is no API gateway or Ollama container. Ollama must listen on an interface
reachable from Docker, commonly `OLLAMA_HOST=0.0.0.0:11434`.

### Container hardening

The image runs as the non-root `medclaim` user. The API and Streamlit root
filesystems are read-only, use temporary in-memory filesystems for `/tmp`, and
set `no-new-privileges`. These controls reduce what a compromised process can
modify. They do not protect the host Ollama endpoint from every local process;
network exposure and host firewall configuration remain separate concerns.

## 4. Artifacts, idempotency, transactions, and race conditions

These concepts are related but not interchangeable.

### Idempotency from first principles

An operation is idempotent when performing it more than once has the same
externally relevant effect as performing it once. Setting a flag to `true` is
idempotent; incrementing a counter is not.

The initializer is operationally idempotent for an existing volume:

- If the versioned corpus directory exists, it reuses it.
- If a BM25 index directory exists, it skips rebuilding it.
- If a dense index directory exists, it skips rebuilding it.
- It then loads and validates both retrievers.

Therefore repeated `docker compose up` calls normally reuse the same artifacts.

The named volume is `medclaim_medclaim-artifacts` under the default Compose
project name. Inside containers, its important layout is:

```text
/app/artifacts/downloads/                         pinned raw releases
/app/artifacts/corpora/medical-corpus-full-v1/   normalized JSONL corpus
/app/artifacts/indexes/medical-bm25-full-v1/     sparse index
/app/artifacts/indexes/medical-dense-full-v1/    vectors and FAISS index
```

The downloader publishes each source through a `.part` file, retries download
up to three times with short exponential delays, verifies a hard-coded SHA-256,
and only then replaces the cached source filename. Existing cached downloads are
also hashed before reuse. This prevents an interrupted or changed upstream file
from silently entering normalization.

The current generated corpus directory contains `documents.jsonl`,
`passages.jsonl`, `claims.jsonl`, `gold_evidence.jsonl`, `manifest.json`, and
`quality_report.json`. These are combined multi-dataset files; each record has a
`dataset` field rather than maintaining three duplicated runtime layouts.

This is not the same as a bit-for-bit reproducible rebuild. If the volume is
deleted, timestamps change and Ollama embedding output may vary with model or
runtime versions. The logical operation is "ensure usable versioned artifacts
exist," not "always reproduce identical bytes on every machine."

### Transaction intuition

A transaction groups changes so observers see either the old state or the new
state, not a half-written mixture. Databases provide this with ACID semantics.
This repository has no database transaction.

It does use a transaction-like filesystem publication pattern:

1. Create a temporary directory beside the target.
2. Write every file into the temporary directory.
3. Compute manifests and checksums.
4. Rename the completed directory to its final versioned name.

On the same filesystem, directory rename is atomic: an observer sees either no
final directory or the complete final directory. This prevents readers from
opening a half-written `index.faiss` under the final path.

### What this pattern does not guarantee

The corpus, BM25 index, and dense index are three separately published
artifacts. There is no transaction covering all three. If dense index creation
fails, the corpus and BM25 index may remain. This is acceptable because the next
initializer run detects existing pieces, builds the missing piece, and validates
compatibility at the end.

### Locking and races

A lock serializes access to shared state. The artifact builder does not use a
file lock. It relies on Compose normally running one initializer.

If two initializer containers run concurrently, both can observe that a target
does not exist and begin building it. This is a time-of-check/time-of-use race:

```text
Process A: target absent -------- build -------- rename
Process B: target absent -------- build -------- rename fails
```

The atomic rename prevents silent replacement, and builders clean temporary
directories after failure, but duplicate work still occurs. A production system
with multiple initializers should add a volume-level lock or designate exactly
one artifact-building job.

## 5. Corpus structure, hashing, and indexing

### Three source schemas, one canonical contract

The datasets do not share a schema, so `corpus/full.py` acts as an adapter
boundary. It downloads immutable source files whose SHA-256 checksums are fixed
in code, cleans Unicode and whitespace, namespaces every ID by dataset, and
emits `documents.jsonl`, `passages.jsonl`, `claims.jsonl`,
`gold_evidence.jsonl`, a manifest, and a quality report.

| Dataset | Source unit | Normalization result |
|---|---|---|
| SciFact | 5,183 abstracts plus 1,409 claims | Abstract sentences become 11,933 grouped passages; sentence-level gold evidence is mapped to every passage containing that sentence |
| HealthVer | 14,330 claim/evidence CSV rows | Every row remains a claim; identical evidence text is content-hash deduplicated into 734 documents and 736 passages |
| PUBHEALTH | Fact-check claims and long article text | 12,255 usable rows become documents and claims; long articles become 88,822 sentence-aware passages |

PUBHEALTH's official files contain 26 rows without usable claim/article text
and 11 structurally malformed TSV rows. The raw ZIP is retained, while those
rows are excluded from retrieval and counted in `quality_report.json`. This is
data-quality handling, not synthetic repair: inventing missing column boundaries
or evidence would make the evaluation corpus less trustworthy.

The unified labels are `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, and `MIXED`.
Original labels and splits remain on each claim, so normalization does not erase
dataset-specific provenance. SciFact test claims and unknown source labels stay
unlabeled instead of receiving fabricated ground truth.

### Canonical records: what normalization actually produces

Normalization does not force each source row into one giant lossy table. It
creates four related record types, each designed for one responsibility:

```text
document  = preserved source text and provenance
passage   = bounded retrieval unit derived from a document
claim     = proposition and label used for evaluation
gold set  = mapping from a labeled claim to its evidence passages
```

A canonical document contains `document_id`, `dataset`, original source ID,
title, source type, complete cleaned text, optional source sentences, metadata,
content hash, and corpus version. A passage contains its own opaque ID, parent
document ID, dataset, passage index, exact text, character offsets, word count,
content hash, corpus version, and source-sentence metadata. A claim retains both
the original and unified labels, original split, explanation when available,
and evidence-set references.

IDs are namespaced because the same integer may appear in multiple datasets:

```text
scifact:document:123
healthver:claim:123
pubhealth:document:train:123
```

Passage IDs are short ordered IDs such as `p0001234`. They are deliberately
independent of source IDs: source identity remains in `document_id`, while the
short passage ID is easier for a small local model to reproduce exactly in its
`evidence_used` array.

### Common text normalization

Every textual field passes through the same conservative cleanup:

1. Reject non-string values as empty input for that field.
2. Apply Unicode NFKC normalization.
3. Replace Unicode control characters with spaces.
4. Collapse every whitespace run to one ordinary space.
5. Strip leading and trailing whitespace.

NFKC makes compatibility forms comparable—for example, full-width Latin
characters normalize to their ordinary equivalents. This helps hashing and
search consistency, but it is not medical-language rewriting. The builder does
not paraphrase claims, correct scientific statements, translate text, or invent
missing evidence.

### Dataset-specific adapters

SciFact supplies one abstract corpus plus train/dev/test claim files. An
abstract becomes a document whose supplied abstract list remains the sentence
boundary source. Claim evidence contains document IDs and exact sentence
indexes. `SUPPORT` maps to `SUPPORTS`; `CONTRADICT` maps to `REFUTES`; claims
without evidence in train/dev map to `NOT_ENOUGH_INFO`; unannotated test claims
stay unlabeled. After chunking, every gold sentence index is resolved to the
passage or passages containing that sentence.

HealthVer supplies rows shaped approximately as `id, evidence, claim, label,
topic_ip, question`. Every one of its 14,330 rows remains a claim. Repeated
evidence is normalized, SHA-256 hashed, and stored once as a document; multiple
claims may reference it. This is evidence deduplication, not row deletion.
`Supports`, `Refutes`, and `Neutral` map to `SUPPORTS`, `REFUTES`, and
`NOT_ENOUGH_INFO`.

PUBHEALTH supplies claim metadata and a long `main_text` fact-check article.
Each usable row becomes both a claim and a document. Split is included in its
ID to prevent collisions. `true`, `false`, `unproven`, and `mixture` map to
`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, and `MIXED`. The source explanation,
fact checkers, date, subjects, and evidence-source URLs remain as provenance.

### The sentence-aware chunking algorithm

“Sentence-aware” is a repository-specific strategy, not the name of one
standard library algorithm. Its objective is to keep complete neighboring
sentences together without allowing an evidence prompt to receive an entire
long article.

The exact algorithm is:

```text
for each normalized document in deterministic dataset/source order:
    obtain sentence spans
    split any span longer than 120 words into <=120-word pieces
    greedily append consecutive pieces while combined words <= 120
    otherwise publish the current chunk and start the next one
    discard a chunk only if it contains no alphanumeric character
    store exact start/end character offsets and source sentence indexes
```

Sentence boundaries come from different places:

- SciFact uses the abstract sentence list supplied by the dataset.
- HealthVer treats one evidence statement as one source sentence.
- PUBHEALTH approximates boundaries after `.`, `!`, or `?` followed by
  whitespace because it supplies article text rather than sentence arrays.

The word limit uses `text.split()`, so it counts whitespace-separated words,
not embedding-model tokens. A 120-word chunk can therefore have more or fewer
than 120 model tokens. There is no intentional overlap. If one source sentence
is 250 words, it becomes pieces of at most 120, 120, and 10 words; all pieces
retain the source sentence index where one exists.

For example, sentence lengths `50, 40, 60` produce chunks `90, 60`, not one
150-word chunk. Greedy grouping improves context continuity and avoids cutting
ordinary sentences, while the hard fallback prevents a malformed or unusually
long sentence from defeating the limit.

No overlap reduces passage count, storage, embedding time, and duplicate search
results. Its cost is boundary sensitivity: evidence requiring the last sentence
of one chunk and first sentence of the next may be split. The current design
relies on retrieving up to five passages rather than duplicating overlap into
every chunk.

### Gold-evidence resolution after chunking

Gold annotations refer to source evidence, while runtime retrieval operates on
passage IDs. The builder constructs two mappings:

```text
document_id -> all derived passages
(document_id, sentence_index) -> passages containing that source sentence
```

SciFact and HealthVer use sentence-index mapping. PUBHEALTH evidence refers to
the article document, so all passages derived from that article are associated
with its gold evidence set. This can create broad PUBHEALTH gold sets; it does
not mean every article passage independently proves the claim.

### Data-quality policy

The builder fails on corrupt SciFact/HealthVer identities, unknown required
labels, invalid sentence indexes, duplicate canonical IDs, or unresolved gold
evidence. PUBHEALTH's official release itself contains 26 empty rows and 11
structurally malformed TSV rows; those raw rows remain in the cached ZIP and are
counted in `quality_report.json`, but cannot safely become invented documents.
One punctuation-only SciFact source fragment is retained in its document but
not emitted as a searchable passage.

The online `/v1/verify` path searches `passages.jsonl` through the indexes. It
does not look up the user's text in `claims.jsonl`, copy a gold label, or send
`gold_evidence.jsonl` to the verifier. Claims and gold mappings are preserved for
evaluation and analysis. Keeping labels outside runtime inference prevents a
benchmark claim from being “verified” merely by reading its answer key.

### Corpus versus index

The corpus is the source collection of passages. An index is a derived data
structure that makes searching the corpus efficient. If corpus content changes,
the old index is no longer valid even if filenames happen to match.

The full corpus currently contains 18,172 normalized documents and 101,491
retrieval passages. Each passage has:

- a stable `passage_id`
- a `document_id`
- a dataset label
- text and offsets
- a content hash
- a corpus version
- provenance metadata

The code computes a corpus hash by canonicalizing every passage as sorted JSON,
adding a newline after each record, and feeding the byte stream into SHA-256.
Both index manifests record the corpus version, content hash, and passage count.

This gives compatibility checking:

```text
index.corpus.identity == corpus.manifest.identity
```

### Integrity is not authenticity

A checksum detects accidental corruption or mismatched files. It does not prove
who created the files. An attacker who can modify both an artifact and its
manifest can recompute the hash. Authenticity would require a trusted signature,
restricted artifact publication, or another trust mechanism.

### Why indexing exists

Without an index, each query would require scanning and interpreting every
passage from scratch. An index precomputes structures that make later queries
fast. This repository builds two different indexes because lexical and semantic
matching fail in complementary ways.

## 6. Sparse retrieval with BM25

### Problem

Exact or near-exact medical terms are strong signals. A query containing
"antibiotics" should quickly find a passage containing "antibiotics."

### Concept

BM25 is a lexical ranking function. It rewards query terms that occur in a
document, especially terms that are rare across the corpus, while controlling
for repeated terms and document length.

A simplified term contribution is:

```text
IDF(term) * [f(term, passage) * (k1 + 1)]
            --------------------------------------------
            f(term, passage) + k1 * (1 - b + b * len/avglen)
```

`f` is term frequency. `k1` controls frequency saturation: the tenth occurrence
does not add ten times the value of the first. `b` controls length normalization.
This project uses `k1=1.5`, `b=0.75`, and `epsilon=0.25`. In
`rank-bm25`, epsilon supplies a small positive floor derived from average IDF
for terms whose raw IDF would be negative because they occur in more than half
the corpus. It prevents extremely common terms from contributing a large
negative value; it does not turn common words into strong evidence.

### Tokenization in this repository

Text is normalized with Unicode NFKC, lowercased, and split into ASCII
alphanumeric tokens. There is no stemming or stop-word removal. Consequently:

- `Antibiotics` and `antibiotics` match.
- `influenza` and `flu` do not match lexically.
- Morphological variants are not automatically collapsed.
- Non-ASCII medical terms may tokenize poorly because the regex is ASCII-only.

### Build and query behavior

The initializer tokenizes every passage and serializes a `rank_bm25.BM25Okapi`
object with Python pickle. At query time, the retriever computes a score for
every passage, sorts descending, and uses passage ID as a deterministic tie-break.

The standalone retriever accepts `top_k` from 1 through 100. In the default
hybrid pipeline it returns 50. BM25 score magnitude is query- and corpus-
dependent: `12` is not “twice as relevant” as `6`, and there is no global BM25
threshold in this application. Only ordering enters RRF. A passage with a zero
BM25 score may still be retrieved semantically by FAISS.

The corpus has 101,491 passages. `rank-bm25` keeps token statistics in memory
and scores the complete collection for each query. This remains practical at
the current scale, but a much larger corpus would benefit from a disk-backed
inverted index such as Lucene/OpenSearch rather than scoring every document in
Python.

### Pickle tradeoff

Pickle is convenient but unsafe for untrusted input because loading a malicious
pickle can execute code. Checksums detect accidental changes but do not make
untrusted pickle safe. The artifact volume must be treated as trusted input.

## 7. Embeddings, vector similarity, and FAISS

### Problem

Lexical matching misses semantic equivalence. "Influenza" and "flu" describe
the same disease but have different tokens. Dense retrieval is intended to find
such relationships.

### Intuition

An embedding model maps text to a vector: a fixed-length list of numbers. The
model is trained so semantically related texts tend to point in similar
directions in vector space.

With the current default model, the live artifact has 768 dimensions; the code
discovers that dimension dynamically with a probe rather than hard-coding it.
Individual dimensions do not normally have human-readable meanings. Meaning is
distributed across the whole vector.

### Vector similarity

Cosine similarity measures the angle between two vectors:

```text
cosine(q, d) = (q dot d) / (||q|| * ||d||)
```

The code L2-normalizes every vector so its norm is one. After normalization:

```text
cosine(q, d) = q dot d
```

That is why the FAISS index is `IndexFlatIP`: inner product becomes cosine
similarity for normalized vectors.

### Exact index, not a vector database

FAISS is an in-process vector-search library. `IndexFlatIP` performs exact
exhaustive search over all stored vectors. The project does not run Pinecone,
Milvus, Qdrant, Weaviate, MongoDB vector search, or PostgreSQL/pgvector.

Calling the current system a "vector database" would be inaccurate. It is a
file-backed FAISS index loaded into API memory. Exact search is simple and
correct for roughly one hundred thousand passages. At millions of passages, an
approximate index such as HNSW or IVF would trade some recall for much lower
latency and memory pressure.

### Document and query prefixes

The initializer embeds documents with `search_document: ` and queries with
`search_query: `. Some embedding models use prefixes to distinguish the role of
the text. The prefixes are part of the model contract, so the dense manifest
records them and the retriever refuses to load a query embedder with a different
prefix.

### Build-time and query-time execution

The initializer first probes `nomic-embed-text:latest` to discover its output
dimension. It sends document text to Ollama's `/api/embed` endpoint in batches
of 64 with the document prefix, validates and L2-normalizes the matrix, then
adds all 101,491 rows to `faiss.IndexFlatIP`. The current artifact therefore
contains 101,491 vectors of 768 `float32` values.

Three files serve different purposes:

```text
index.faiss       searchable FAISS structure
embeddings.npy    inspectable normalized source matrix
passage_ids.json  vector-row position -> passage ID
```

For a request, the claim is normalized, limited to 5,000 characters, prefixed
with `search_query: `, and embedded once. FAISS returns the 50 highest inner
products in the default hybrid mode. Ties use passage ID deterministically after
the FAISS result is converted into application records.

`IndexFlatIP` is exact: it compares the query with every stored vector. Its
approximate arithmetic work grows as `number_of_passages × dimension`; here that
is roughly `101,491 × 768` multiply-add comparisons per query. It avoids recall
loss and index-training parameters, at the cost of linear query work.

Dense score is normally near the cosine range `[-1, 1]`, but the application
does not interpret fixed bands as truth or direct evidence. It uses dense rank,
not raw dense magnitude, during fusion.

### Validation

The code rejects vectors with incorrect shape, dimension, zero norm, NaN, or
infinity. It converts to contiguous `float32`, normalizes them, stores both
`embeddings.npy` and `index.faiss`, and verifies their checksums when loading.

Storing both vectors and the FAISS index is redundant for serving, but it makes
validation and inspection easier. The cost is additional disk space.
In the validated build, `embeddings.npy` and `index.faiss` are about 312 MB
each, and the complete downloads/corpus/BM25/dense artifact set is about 886 MB.
The first full embedding build took roughly nine minutes on the development
laptop; that is an observation, not a portable performance guarantee.

## 8. Hybrid retrieval and Reciprocal Rank Fusion

### Problem

BM25 scores and cosine similarities are not directly comparable. A BM25 score
of 8 and a cosine score of 0.8 do not imply a tenfold difference in relevance.

### Concept

Reciprocal Rank Fusion, or RRF, combines rankings rather than raw scores:

```text
RRF(passage) = sum over retrievers [1 / (k + rank)]
```

The project uses `k=60`. A passage ranked first by both systems receives:

```text
1/61 + 1/61 = 0.032786...
```

A passage found by only one retriever receives only one contribution. This
rewards agreement without assuming score calibration.

### Current execution

`HybridRetriever.search` executes BM25 first and dense retrieval second. It does
not run them concurrently. It then validates that both indexes reference the
same corpus identity and merges results by `passage_id`.

Its constructor defaults are:

```text
sparse_top_k = 50
dense_top_k  = 50
fusion_top_k = 30
rrf_k        = 60
```

The reranked pipeline explicitly asks hybrid retrieval for 30 candidates, so
the fused list passed to Ollama has at most 30 records. A passage appearing in
both input lists is one output record with two contributions; a passage
appearing in only one list still remains eligible.

For example, a passage ranked BM25 #2 and dense #10 receives:

```text
1/(60 + 2) + 1/(60 + 10) = 0.030414...
```

Another passage ranked #1 only in BM25 receives `1/61 = 0.016393...`, so cross-
retriever agreement can outweigh one excellent isolated rank. `rrf_k=60`
dampens the difference between nearby ranks. A smaller value would emphasize
top positions more aggressively; a larger value would flatten contributions.

Tie-breaking is deterministic: descending RRF score, then best component rank,
then passage ID.

### Tradeoffs

RRF is robust and easy to reason about, but it discards the magnitude of raw
scores. The difference between dense similarities 0.90 and 0.50 matters only
through their ranks. A learned fusion model could exploit more information but
would require training data and calibration.

## 9. Reranking: what the score really means

### Problem

First-stage retrieval optimizes recall: avoid missing useful evidence. It often
returns broadly related passages. The verifier needs a smaller set of passages
that directly addresses the complete claim.

### Concept

Reranking applies a more expensive model to the candidate set and produces a
new ordering. Traditional rerankers are often trained cross-encoders. This
repository instead uses the local generative Ollama model to assign a
claim-passage relevance score. The judgment is pointwise in the sense that each
output score belongs to one passage, although several passages are presented in
the same generation batch.

### Current prompt contract

For each batch, the reranker sends only the claim, passage IDs, and passage text.
It does not send BM25, dense, or RRF scores. The model is instructed to emit a
number using this scale:

- `0.00-0.20`: incidental overlap or unrelated
- `0.21-0.49`: same broad topic
- `0.50-0.69`: partially addresses the claim
- `0.70-1.00`: direct supporting or contradicting evidence

The number is an LLM judgment, not a probability and not a mathematically
calibrated relevance score. Temperature zero reduces sampling variation but
does not create a deterministic or calibrated function across model versions,
hardware, or runtime implementations.

### Pointwise scoring and batch size

The current Compose settings retrieve 30 candidates and use
`MEDCLAIM_RERANKER_BATCH_SIZE=9`. The reranker therefore divides a normal
candidate set into four Ollama generation calls: three batches of nine and one
batch of three.

Each call returns an object shaped like:

```json
{
  "scores": [
    {"passage_id": "p0001234", "relevance_score": 0.82}
  ]
}
```

Four sequential generation calls are a major part of request latency. Batching
reduces the number of calls and places some competing passages in the same
context, but passages in different batches are not jointly compared. Therefore
score comparability across all 30 passages is imperfect. Larger batches reduce
round trips but ask a small model for larger, more failure-prone structured
arrays; smaller batches increase latency.

### Validation and recovery

The code requires exactly one valid score per supplied ID and rejects unknown,
duplicate, missing, non-finite, or out-of-range scores. If a multi-item batch
returns the wrong IDs or count, `_score_resilient` recursively splits the batch
and tries smaller groups.

This is a retry-like recovery strategy, but it is not a general network retry.
Timeouts and HTTP failures are not retried.

Temperature is zero, but the score is still not guaranteed to be identical on
every run. Model blobs, Ollama versions, floating-point kernels, prompt context,
and batch composition can change generation. Even an identical decimal should
be treated as a policy input, not a measured probability.

### Sorting

Results are sorted by:

1. descending LLM relevance score
2. previous hybrid rank
3. passage ID

The first five become gate candidates. The full scored ordering is retained for
the UI trace.

### The aligned direct-evidence boundary

The prompt defines `0.70` as direct evidence and the evidence gate uses the same
`0.70` threshold. A passage at the lower edge of the direct-evidence range can
therefore proceed to verification instead of being rejected by a contradictory
downstream policy.

This illustrates a general lesson: thresholds create decision discontinuities.
A change from `0.69` to `0.70` can change the product verdict even though the
underlying semantic judgment barely changed. An uncalibrated LLM decimal is a
fragile sole gate signal.

### Provider selection

`build_runtime_pipeline` reuses one `OllamaProvider` when the reranker and
verifier model names match. If they differ, it constructs a provider for each
model. This keeps configuration metadata aligned with the model actually called
without paying for a redundant provider object in the default configuration.

An earlier `maximum_input_length` setting was removed because it was validated
and reported but never enforced. A real input limit must define whether it is
measured in characters or model tokens and must specify safe passage truncation;
ceremonial configuration creates false confidence.

## 10. The evidence gate as a state machine

### State-machine concept

A state machine models a system as a finite set of states and explicit
transitions based on events or conditions. It is useful when certain actions are
allowed only after prerequisites are satisfied.

The evidence gate has two states:

```text
candidate passages
  -> no candidates -----------------------> ABSTAIN
  -> top score below threshold ----------> ABSTAIN
  -> too few relevant passages ----------> ABSTAIN
  -> too few unique documents -----------> ABSTAIN
  -> all requirements satisfied ---------> PROCEED
```

When it abstains, the verifier is not called. The pipeline deterministically
returns `NOT_ENOUGH_INFO`, confidence `0.0`, and no evidence citations.

### Why a gate exists

Without a gate, an LLM may manufacture a confident verdict from weak topical
context. The gate enforces the policy that weak evidence must become abstention,
not hallucinated support or refutation.

### Score-field polymorphism

The pipeline chooses the gate score according to retrieval mode:

- BM25 mode: `bm25_score`
- dense mode: `dense_score`
- hybrid mode: `rrf_score`
- hybrid-reranked mode: `reranker_score`

These score families have different scales. A threshold meaningful for a
reranker score is not meaningful for RRF or BM25. Configuration must therefore
be coupled to retrieval mode; the current single threshold environment variable
does not encode that relationship explicitly.

For example, with two retrievers and `rrf_k=60`, the largest possible RRF score
is `2/61 ≈ 0.0328`. Switching only `MEDCLAIM_RETRIEVAL_MODE=hybrid` while
leaving the gate threshold at `0.70` would make every request abstain. Mode and
threshold must be tuned as one policy. In pure BM25 or dense mode, the runtime
also bypasses RRF/reranking and sends that retriever's top five directly to the
gate.

### Exact gate algorithm and parameters

In default `hybrid_reranked` mode, the gate receives the five highest reranked
passages. It validates provenance and finite scores, then computes:

```text
top_score = maximum reranker_score
relevant  = candidates where reranker_score >= 0.70
selected  = first 5 relevant candidates
```

It proceeds only when all three conditions hold:

```text
top_score >= gate_minimum_score                    # 0.70
len(relevant) >= gate_minimum_relevant_passages   # 1
unique document IDs in relevant >= minimum        # 1
```

The first failing condition determines the reason code. With current defaults,
the last two conditions are effectively satisfied whenever one candidate
crosses the threshold. They become meaningful if increased—for example,
requiring two unique documents can reduce reliance on one source but may reject
claims whose only authoritative evidence is one paper.

The comparison is inclusive: exactly `0.70` passes. The gate does not average
scores, convert them into confidence, inspect verdict direction, or consider
BM25/dense/RRF scores in the default mode. It knows only evidence sufficiency,
not whether the evidence supports or refutes. Direction is the verifier's job.

If the gate abstains, its deterministic output is:

```text
verdict         = NOT_ENOUGH_INFO
confidence      = 0.0
evidence_used   = []
verifier calls  = 0
```

This `0.0` is not a claim that `NOT_ENOUGH_INFO` is certainly correct. It means
no model verdict was accepted because the relevance policy did not authorize a
verification call.

## 11. Evidence-bound verification and prompt security

### Trust-boundary problem

Claims and corpus passages are untrusted text. They can contain sentences such
as "ignore previous instructions" or strings that resemble tool commands. The
application must treat them as data, not authority.

### Defense strategy

`SecureVerifier` applies several layers:

1. Claim and passage text are HTML-escaped.
2. Text is placed inside explicit `<claim>` and `<evidence>` blocks.
3. The system prompt states that block contents are untrusted.
4. The provider receives no tool interface.
5. Ollama is asked for output matching a JSON schema.
6. Application code validates the returned object again.

Escaping alone does not solve prompt injection; models interpret natural
language, not an XML security type system. The stronger controls are evidence
restriction, absence of tools, strict output validation, and rejection of
unknown citations.

### Verdict semantics

- `SUPPORTS`: evidence establishes every material part.
- `REFUTES`: evidence establishes that any material part is false.
- `NOT_ENOUGH_INFO`: evidence neither establishes nor contradicts the claim.

Missing evidence is not negative evidence. The absence of a treatment claim
from the corpus cannot justify `REFUTES`; it justifies `NOT_ENOUGH_INFO`.

### Structured output and validation

The verifier must return exactly five fields: verdict, confidence, explanation,
evidence IDs, and limitations. The code rejects:

- unsupported verdicts
- confidence outside `[0,1]`
- unknown passage IDs
- tool-call-like extra fields
- secret-like output
- hidden-prompt disclosure patterns
- empty explanations or malformed limitations

This is defense in depth: Ollama's schema-guided generation improves the chance
of valid output, but the application never assumes the model complied.

### How verdict confidence is produced and checked

The verifier model itself generates `confidence` together with its verdict. No
formula derives it from BM25 score, cosine similarity, RRF, reranker score,
number of citations, or dataset label. The prompt asks for a JSON-schema number
between zero and one; application code then checks that the value is numeric,
finite, not a Boolean, and inside `[0, 1]`.

That is range validation, not calibration. Calibration would require evaluating
many labeled claims and showing, for example, that results reported near `0.8`
are correct about 80% of the time under a fixed corpus/model/prompt. This
repository has no such fitted mapping, so the UI correctly describes confidence
as a model estimate rather than a clinical probability.

There are three common cases:

```text
gate abstains
    -> confidence is deterministically 0.0; verifier did not run

gate proceeds, one component
    -> confidence is the verifier's validated number

multiple components
    -> overall confidence is min(component confidences)
```

The minimum aggregation rule is conservative but not statistical. Component
outputs are neither independent probabilities nor guaranteed calibrated
measurements. `min` simply prevents a compound result from appearing stronger
than its weakest component.

Confidence does not override evidence rules. A model response with confidence
`0.99` is still rejected if it cites an unknown passage, lacks required
citations, gives medical advice, contradicts its own verdict wording, or fails
schema validation. Conversely, a relevance score of `0.95` only says a passage
directly addresses the claim; it does not force `SUPPORTS` and does not become
verdict confidence.

### Explanation and citation validation

For an accepted non-NEI verdict, at least one supplied passage must be cited.
A verifier-generated `NOT_ENOUGH_INFO` must also cite the selected evidence that
proved insufficient; only gate-level abstention is allowed to have no citations.
Duplicate or unknown IDs are rejected.

Explanations must contain 5 through 120 whitespace-separated words, must not be
boilerplate-only, must use language consistent with the verdict, and must not
contain configured personalized-treatment phrases. Causal wording is rejected
unless the supplied passages themselves contain causal language. These checks
are heuristics: they reduce obvious inconsistency but do not prove scientific
quality.

### One correction attempt

After schema validation, `ExplanationValidator` enforces citation coverage,
length, verdict-language consistency, lack of personalized advice, and causal
language support. If validation fails, the LLM gets exactly one correction call.
The corrected result must pass or the request fails.

This is a bounded retry. Bounded retries prevent infinite loops and make worst-
case latency easier to estimate. The tradeoff is that a second correctable
failure still becomes a service error.

### Absolute claims

If an absolute claim initially receives `NOT_ENOUGH_INFO`, the verifier performs
one additional pass focused on counterexamples. This captures logic such as:

```text
Claim: Treatment X prevents all events.
Evidence: One treated patient still had the event.
Conclusion: REFUTES.
```

A single counterexample refutes a universal statement. This is deductive logic,
not a statistical confidence rule.

## 12. Claim decomposition and aggregation

### Problem

"Vitamin D supports bone health and antibiotics prevent influenza" contains two
independent propositions. A single verdict may hide that one is supported and
the other refuted.

### Concept

Claim decomposition converts a compound claim into atomic claims, verifies each
independently, then aggregates the results. The code detects likely compounds by
looking for `and`, `but`, or semicolons between predicate-bearing clauses.

### Actual current behavior

`VerificationPipeline` creates `ClaimDecomposer(None)`. No decomposition
provider is wired. In `auto` mode, a likely compound claim triggers an attempted
decomposition, encounters "No decomposition provider is configured," and falls
back to treating the original text as one atomic claim with a warning.

Therefore the code contains a sophisticated decomposition validator, but the
running application does not actually split compound claims. This is an example
of the difference between implemented capability and assembled runtime behavior.

### Aggregation rules

If decomposition were active:

- identical component verdicts become that overall verdict
- differing component verdicts become `MIXED`
- overall confidence is the minimum component confidence
- evidence IDs are de-duplicated in component order

Using the minimum confidence is conservative: the overall result is no stronger
than its weakest part. It is simple but not statistically justified because the
component confidences are uncalibrated LLM estimates.

## 13. Dependency injection and structural interfaces

### Problem

Hard-coding every dependency inside every function makes components difficult
to replace, test, or reason about independently.

### Concept

Dependency injection means a component receives its dependencies from outside
rather than constructing all of them internally. Constructor injection makes
the dependency graph explicit.

Examples in this repository:

- `VerificationPipeline(retriever, verifier, evidence_gate, ...)`
- `RerankedRetriever(hybrid_retriever, reranker, configuration)`
- `SecureVerifier(provider)`
- `create_app(settings=..., service=...)`

Python `Protocol` types define structural contracts. An object satisfies the
`Embedder` or `EvidenceReranker` protocol by having the required attributes and
methods; it need not inherit from a base class. This is static duck typing.

### Current limitations

The top-level `build_runtime_pipeline` is still a manual composition root and
constructs concrete Ollama implementations. That is appropriate for a small
application. A dependency-injection framework would add indirection without
much benefit here.

`create_app` supports injecting a service, but the repository no longer has a
test suite exercising that seam. Dependency injection improves testability only
when tests actually use it.

## 14. Synchronous execution, event loops, and concurrency

### Async from first principles

Synchronous code waits for each operation before continuing. Asynchronous code
can suspend while waiting for I/O so an event loop can advance other tasks on
the same thread.

An event loop repeatedly runs ready tasks and resumes tasks whose awaited I/O
completed. Async improves I/O concurrency; it does not make CPU-bound work
parallel by itself.

### Actual execution model

The `/v1/verify` route is declared with ordinary `def`, not `async def`. Ollama
calls use a process-wide synchronous `httpx.Client`, which safely reuses TCP
connections across worker threads. BM25, dense search, reranking batches, the
gate, and verification execute sequentially for one request.

FastAPI/Starlette normally runs synchronous endpoint functions in a worker
threadpool so the server event loop is not directly blocked. This permits
multiple requests, but each request occupies a thread during long Ollama waits.

### No queue and no backpressure

There is no job queue controlling access to Ollama. Concurrent API requests can
each issue several sequential operations. With 30 candidates in batches of
nine, one verification normally makes roughly:

```text
1 query embedding + 4 reranker generations + 1 verifier generation
```

Correction or absolute-claim passes can add calls. Multiple users multiply this
load. A production design could use a semaphore, bounded work queue, or Ollama-
aware request scheduler to prevent overload.

### Request-local reranking results

The pipeline and reranker are constructed once and shared by requests, so
request-specific data must not be stored on either object. An earlier design
wrote the full trace to a shared attribute:

```text
self.last_scored_candidates = ...
```

`RerankedRetriever` then read that attribute for tracing. Under concurrent
requests, request B could overwrite it between request A's write and read, and A
could receive B's candidate trace.

That was a race condition: correctness depended on timing between threads. The
current implementation avoids the shared mutation by returning both selected
and fully scored candidates from `rerank`. Each request keeps its trace in local
variables, so no lock is required for this data.

### Locking in metrics

`MetricsRegistry` uses `threading.Lock` around updates and rendering. Without the
lock, read-modify-write operations on counters could interleave and lose updates
or render inconsistent snapshots. The lock protects a small critical section.

The lock is process-local. If Uvicorn ran multiple worker processes, each worker
would have independent metrics, and `/metrics` would expose only the process
answering that request unless an external aggregation strategy were added.

## 15. Timeouts, retries, idempotency, and delivery guarantees

### Timeout layers

The runtime Ollama client has a 120-second timeout for each individual embed or
generation request. The artifact initializer uses 600 seconds per embedding
call because first-build batches may be slower. Streamlit waits up to 300
seconds for the entire API request.

These limits cover different scopes:

```text
LLM_TIMEOUT_SECONDS=120
    applies separately to each Ollama call

MEDCLAIM_API_TIMEOUT_SECONDS=300
    applies once to Streamlit waiting for the complete pipeline
```

A normal single-component request makes one query-embedding call, four reranker
generation calls for 30 candidates in batches of nine, and—if the gate
proceeds—one verifier generation. Validation correction or absolute-claim
counterexample handling can add another generation. Therefore total pipeline
time can exceed one per-call timeout without any individual call timing out.

Timeouts do not automatically cancel work across service boundaries. Closing the
Streamlit HTTP connection does not guarantee the API stops its pipeline or that
Ollama stops generation.

### Retry inventory

The repository has three different mechanisms often casually called retries:

- Compose health checks retry failed probes five times.
- Reranking recursively splits structurally invalid batches.
- Explanation validation allows one LLM correction.

There is no general HTTP retry with exponential backoff for Ollama timeouts or
connection failures. This avoids duplicating expensive generations but reduces
resilience to transient failures.

### Is POST `/v1/verify` idempotent?

Domain-wise, verification does not write a claim record, so repeating a request
does not duplicate stored business data. Operationally, it is not fully
idempotent:

- metrics counters increment again
- logs receive another trace
- Ollama may generate a different score or verdict
- latency and request ID differ

The `X-Request-ID` header correlates a request but does not deduplicate it. There
is no idempotency-key store.

### Message delivery guarantees

There is no message broker, so at-most-once, at-least-once, and exactly-once
queue delivery semantics do not directly apply. HTTP has an ambiguity instead:
if the client times out, it may not know whether the server completed. Retrying
can execute verification again. Because there is no persisted side effect, this
is mostly a cost and consistency concern rather than duplicate-data corruption.

## 16. Serialization formats and compatibility

Serialization converts in-memory structures to bytes that can cross process or
time boundaries.

This repository uses:

- JSON for manifests and API objects
- JSONL for passage/document records
- pickle for the BM25 Python object
- NumPy `.npy` for dense vectors
- FAISS's binary format for the vector index

JSON is portable and inspectable but has limited types. JSONL permits streaming
one record per line. NumPy and FAISS formats are efficient but library-specific.
Pickle is Python-specific and unsafe for untrusted artifacts.

Manifests are compatibility contracts. They record versions, model IDs,
dimensions, prefixes, file paths, and checksums. Runtime validation fails closed
when these contracts disagree rather than silently searching incompatible data.

## 17. Configuration precedence and reproducibility

Settings begin in `configs/deployment/default.yaml`. `load_runtime_settings`
then applies environment variables, and Pydantic validates and freezes the
result.

The precedence is:

```text
RuntimeSettings class defaults
  <- YAML values
  <- environment-variable overrides
```

Compose supplies the artifact paths and most production values, so the paths in
the YAML file are not the paths used by the Docker API.

The main runtime controls are:

| Environment variable | Settings field | Validation/current default |
|---|---|---|
| `LLM_MODEL` | verifier model | non-empty; `dolphin-llama3:8b` |
| `OLLAMA_BASE_URL` | Ollama origin | HTTP(S) origin; host gateway in Compose |
| `LLM_TIMEOUT_SECONDS` | per-call timeout | `>0`, at most 600; current 120 |
| `MEDCLAIM_CORPUS_DIR` | corpus artifact | required in strict Docker startup |
| `MEDCLAIM_BM25_INDEX_DIR` | BM25 artifact | required when sparse retrieval is used |
| `MEDCLAIM_DENSE_INDEX_DIR` | FAISS artifact | required when dense retrieval is used |
| `MEDCLAIM_EMBEDDING_MODEL` | embedding model | current `nomic-embed-text:latest` |
| `MEDCLAIM_EMBEDDING_QUERY_PREFIX` | query role prefix | current `search_query: ` |
| `MEDCLAIM_RERANKER_MODEL` | reranker model | required in reranked mode |
| `MEDCLAIM_RERANKER_BATCH_SIZE` | passages per call | integer 1–30; current 9 |
| `MEDCLAIM_RETRIEVAL_MODE` | pipeline variant | BM25, dense, hybrid, or hybrid-reranked |
| `MEDCLAIM_RETRIEVAL_CANDIDATE_COUNT` | reranker pool | integer 1–100; current 30 |
| `MEDCLAIM_TOP_K` | final evidence maximum | integer 1–100, no larger than candidate count; current 5 |
| `MEDCLAIM_GATE_MINIMUM_SCORE` | gate threshold | finite; in reranked mode at most 1; current 0.70 |
| `MEDCLAIM_API_TIMEOUT_SECONDS` | Streamlit wait | UI-only setting; current 300 |

`gate_minimum_relevant_passages` and `gate_minimum_unique_documents` currently
come from YAML and both default to one; they are not exposed through the
environment map. Hybrid's 50/50 component depths and `rrf_k=60` are constructor
defaults, not current YAML settings. Dense build batch size 64 and maximum chunk
words 120 are initializer arguments. Knowing where a parameter lives matters:
changing an environment variable that no loader reads has no effect.

The frozen settings object prevents accidental mutation after startup. Its
canonical JSON representation is hashed into `configuration_hash`, which is
included in request logs. This helps correlate behavior with configuration.

It does not capture everything affecting reproducibility: Ollama version, model
blob digest, Docker image digest, host hardware, and artifact hashes are not all
inside that configuration hash.

## 18. API boundaries and error semantics

### Safety routing before retrieval

The service first runs deterministic regular-expression rules. Personal
emergency language returns an emergency-limited response; personalized dosage,
medication-change, diagnosis, or treatment requests are scope-limited. General
text proceeds to retrieval even when it contains unfamiliar terminology. The
system deliberately does not use a medical-keyword allowlist: corpus retrieval
and the evidence gate decide coverage.

This router is a policy filter, not an intent classifier trained on broad data.
Its strengths are auditability and deterministic behavior; its weakness is that
natural-language paraphrases outside the patterns may be missed. It neither
retrieves evidence nor contributes to confidence.

Pydantic rejects empty claims, claims longer than 5000 characters, and unknown
request fields. The API validates optional request IDs to prevent whitespace and
unbounded header values.

`create_app` catches pipeline-construction errors and still starts FastAPI with
`pipeline=None`. This supports liveness and diagnostic readiness instead of
crashing immediately. Verification then returns a controlled 503.

Readiness independently checks artifacts and models and also publishes the
captured pipeline-construction error as a failed `pipeline` check. Therefore a
live process with an unavailable pipeline returns HTTP 503 from readiness and
Compose does not start Streamlit as though verification were ready.

All `VerificationServiceError` failures become HTTP 503. That is simple but
coarse. A provider timeout, malformed model output, and internal programming bug
can have different operational meanings and retry policies.

## 19. Observability, privacy, and cardinality

### Structured logging

Request logs are JSON objects with stage names, timestamps, request IDs,
configuration identity, claim length, and SHA-256 claim hash. Raw claims and
evidence are deliberately excluded.

A deterministic claim hash permits correlation of identical text without
logging the text. It is pseudonymization, not guaranteed anonymization. An
attacker can hash a dictionary of likely claims and compare results. A keyed HMAC
would resist offline guessing better.

### Metrics and cardinality

Prometheus-style labels create one time series per unique label combination.
Putting request IDs or claim hashes in labels would create unbounded cardinality
and exhaust memory. `MetricsRegistry` therefore permits only bounded labels such
as verdict, mode, category, and error code.

Metrics live only in API memory and disappear on restart. There is no external
Prometheus server in Compose.

Some metric names currently overstate precision. For example,
`retrieved_candidates_count` is updated with the number of final `evidence_used`
IDs, not the first-stage retrieved candidate count. Reading observability code
critically is part of understanding what a metric actually measures.

## 20. Caching: what is and is not cached

Caching stores a previously computed value so later requests can reuse it.

This system has several forms of reuse:

- Docker's named volume preserves corpus and index artifacts.
- The API loads BM25 and FAISS indexes once at process startup.
- BM25 and dense retrievers share the same parsed corpus objects instead of
  independently duplicating 101,491 passage dictionaries in memory.
- The query embedder object is constructed once.
- Successful immutable corpus validation is cached inside readiness, so the
  ten-second health probe does not repeatedly parse and hash a 106 MB JSONL
  passage file.
- Ollama may keep model weights in memory internally.

It does not cache query embeddings, retrieval results, reranker scores, or final
verdicts. Repeating the same claim repeats the full online pipeline and can
produce a different LLM score.

Caching verdicts would reduce latency but introduces invalidation questions:
which corpus version, model version, prompt version, threshold, and configuration
must be part of the cache key? Returning an answer produced under an old evidence
corpus would be a correctness bug.

## 21. No database, ORM, or isolation level

The current repository has no database connection. Therefore:

- there are no SQL transactions
- there is no ORM identity map or lazy loading
- there are no database locks or isolation levels
- there are no schema migrations
- there is no persisted request history

The named Docker volume is storage, not a database. It provides a filesystem,
not queries, transactions, indexes over arbitrary fields, or concurrent update
isolation.

If request history or user data is later added, database isolation becomes
relevant. For example, two concurrent workers updating the same aggregate under
`READ COMMITTED` can lose updates unless using atomic SQL, row locks, or optimistic
version checks. None of that should be assumed to exist today.

## 22. Worked trace: "Antibiotics effectively treat influenza"

1. The API validates the request body.
2. Safety routing classifies it as a general verifiable claim.
3. Compound detection finds one proposition, so no decomposition is attempted.
4. BM25 scores every passage and returns its best 50 lexical matches.
5. Ollama embeds the claim; exact FAISS search returns 50 semantic matches.
6. RRF merges those ranks and retains 30 candidates.
7. The reranker makes four sequential calls and retains five candidates.
8. The gate compares those candidates with the inclusive `0.70` threshold.

At this point two legitimate execution paths exist:

```text
top reranker score < 0.70
    -> gate ABSTAIN
    -> NOT_ENOUGH_INFO, confidence 0.0
    -> verifier never runs

top reranker score >= 0.70
    -> gate PROCEED
    -> selected passages go to verifier
    -> sufficiently direct contradiction should yield REFUTES
```

During full-corpus validation with the current local model, this example once
received a top reranker score of `0.56`, so the correct behavior under the
configured policy was abstention—even though a human may know the claim is
false. That is not evidence that the final verdict should be hard-coded. It
shows that corpus coverage, first-stage recall, reranker calibration, and gate
policy are separate failure points.

A second end-to-end test used a HealthVer claim about bats or pangolins as
possible original hosts of COVID-19. It retrieved direct evidence, crossed the
gate at `0.70`, and the verifier returned `SUPPORTS` with model confidence
`0.90`. The relevance score authorized verification; the separate confidence
came from the verifier and was validated only for structure/range.

This stage-by-stage conditional trace is the right way to diagnose RAG. Starting
from the final label and changing a prompt blindly hides which contract failed.

## 23. Failure analysis by stage

When debugging, classify the failure before changing prompts:

| Symptom | Likely stage | What to inspect |
|---|---|---|
| Correct passage absent | corpus/index/retrieval | Corpus coverage, tokenization, embedding model, top-k |
| Correct passage retrieved but ranked low | fusion/reranking | BM25/dense ranks, RRF, reranker prompt and batch behavior |
| Correct passage scored direct but result is NEI | evidence gate | Score field, threshold, minimum counts |
| Gate proceeds but wrong verdict | verifier | Exact model-input evidence, verifier prompt, model output |
| Good verdict rejected with 503 | validation | Schema, citations, explanation rules, correction attempt |
| UI times out while API logs continue | timeout layering | Streamlit 300-second whole-request limit, per-call Ollama timeout, number of generations |
| Wrong trace under concurrent load | request-state leak | Verify reranking returns request-local results |
| Startup remains unready | artifact/model checks | Manifests, hashes, volume, `/api/tags`, model names |

## 24. Current design strengths

- Clear evidence provenance is carried through every stage.
- Sparse and semantic retrieval complement one another.
- Index/corpus compatibility is validated aggressively.
- Artifacts are published atomically by versioned directory.
- Weak evidence causes abstention instead of forced classification.
- Provider output is treated as untrusted and validated twice.
- Online containers are read-only and non-root.
- Logs avoid raw claim and evidence text.
- Deterministic tie-breakers make retrieval ordering reproducible when scores tie.

## 25. Current design limitations and priorities

1. Calibrate the now-aligned reranker and gate boundary against a labeled
   evaluation set. Consider categorical relevance instead of an arbitrary LLM
   decimal.
2. Replace long synchronous UI waits with cancellable background jobs or
   streaming progress if compound verification becomes active.
3. Add bounded concurrency or queuing around Ollama.
4. Either wire a decomposition provider or remove the impression that runtime
   decomposition is active.
5. Restore focused tests for artifact idempotency, retrieval correctness,
   threshold boundaries, prompt validation, and concurrent trace isolation.
6. Replace or harden pickle loading if artifacts can ever come from an untrusted
   source.
7. Correct observability names and stage timing so metrics reflect what they
   claim to measure.
8. Pin or record model blob digests and image digests for stronger
   reproducibility.
9. Make gate thresholds mode-specific so changing retrieval mode cannot silently
   apply a reranker-scale threshold to BM25, cosine, or RRF scores.

## 26. Study exercises

1. Calculate BM25 and RRF ranks manually for a three-passage toy corpus.
2. Show why normalized inner product equals cosine similarity.
3. Change only the gate threshold in a controlled environment and trace which
   stage changes for the influenza claim.
4. Write a concurrency test that forces two reranker calls to interleave and
   proves their returned traces remain isolated.
5. Replace the reranker's tuple return with a named result object containing
   both selected and traced candidates; explain the readability tradeoff.
6. Design a cache key that is safe across corpus, model, prompt, and
   configuration changes.
7. Model the verification pipeline as an explicit state machine and identify
   terminal versus retryable failures.
8. Design a file-locking protocol for multiple artifact initializers and explain
   stale-lock recovery.
9. Replace the pointwise decimal reranker with `DIRECT`, `PARTIAL`, and
   `IRRELEVANT`; define validation and gate behavior.
10. Estimate worst-case Ollama calls for a compound absolute claim whose first
    explanation fails validation.

## 27. Compact glossary tied to this project

**Abstention:** Choosing `NOT_ENOUGH_INFO` because evidence is insufficient,
rather than forcing support or refutation.

**Atomic publication:** Making a completed directory visible with one rename so
readers do not observe partially written files.

**Caching:** Reusing prior computation. This project caches loaded artifacts but
not query results.

**Calibration:** Empirically aligning reported scores with observed outcomes.
The verifier and reranker numbers are range-validated but not calibrated.

**Chunking:** Converting long documents into bounded retrieval units. This
project greedily groups sentence-aware pieces up to 120 words without overlap.

**Confidence:** The verifier model's bounded self-estimate for its verdict. It
is not derived from retrieval scores and is not a clinical probability.

**Concurrency:** Multiple requests making progress during overlapping time. The
API can process sync routes in multiple threads.

**Dependency injection:** Supplying retrievers, providers, and services to
constructors instead of hiding their construction inside business logic.

**Embedding:** A numeric vector representing text semantics. Ollama produces the
vectors used by FAISS.

**Idempotency:** Repetition without additional relevant effect. Startup reuses
existing artifacts, while verification still increments metrics and reruns the
LLM.

**Index:** A derived search structure. BM25 and FAISS indexes accelerate two
different notions of relevance.

**Provenance:** Information connecting a passage to its dataset and source
document. It makes evidence traceable and supports unique-document gate rules.

**Reciprocal Rank Fusion:** Rank-based combination of BM25 and dense results
using `1/(k+rank)` contributions rather than incomparable raw scores.

**Lock:** A mechanism ensuring only one thread or process enters a critical
section. Metrics use a thread lock; artifact creation has no file lock.

**Race condition:** Behavior that changes with execution timing. The reranker
previously exposed this problem through shared trace state; it now returns
request-local trace data.

**Reranking:** Applying an expensive relevance judgment after broad retrieval.
Here it is a local LLM's uncalibrated score.

**Retry:** Reattempting failed work. This code has bounded structural recovery,
not general provider retry.

**Serialization:** Encoding state into JSON, JSONL, pickle, NPY, or FAISS binary
files so it can cross process or time boundaries.

**State machine:** Explicit states and transition conditions. The evidence gate
moves to `PROCEED` or `ABSTAIN`.

**Transaction:** A group of changes exposed atomically. Filesystem publication
is transaction-like for one artifact but there is no cross-artifact or database
transaction.

**Vector similarity:** Comparing embedding direction. Normalized inner product
implements cosine similarity in the current FAISS index.

## Closing perspective

The project is not "an LLM that knows medicine." It is a chain of contracts:

```text
trusted artifact identity
  -> valid retrieval candidates
  -> explicit relevance policy
  -> sufficient evidence
  -> constrained model judgment
  -> deterministic validation
  -> observable API result
```

Understanding MedClaimRAG means understanding where each contract is enforced,
where probabilistic model behavior enters, and where shared state or configuration
can invalidate an otherwise reasonable design. That is the difference between
knowing how to run the application and understanding how and why it works.
