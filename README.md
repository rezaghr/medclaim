# MedClaim

MedClaim is a claim-verification system that retrieves evidence and uses an LLM to classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Medical safety boundary

MedClaimRAG verifies textual claims against a limited indexed corpus. It is an
educational research prototype. It is not a doctor, diagnostic system,
treatment recommender, or emergency service. Its results are not medical
advice.

The runtime supports exactly HealthVer, SciFact, and PUBHEALTH. A deterministic
scope router prevents personal diagnosis, treatment, dosage, medication-change,
emergency-like, and non-medical requests from entering ordinary verification.
Claims and evidence are escaped, explicitly delimited as untrusted content, and
sent through a tool-free evidence-only provider interface. Unknown citations,
tool requests, unsupported metadata, prompt leakage, and secret-like output are
rejected.

## Hardened API and Streamlit demo

Copy `.env.example` to `.env`, set artifact paths, and keep provider credentials
only in the environment. Raw claim text and explanations are neither logged nor
persisted by default.

Start the API and UI locally:

```bash
PYTHONPATH=src uvicorn medclaim.api.app:app --host 0.0.0.0 --port 8000
streamlit run app/streamlit_app.py
```

The API publishes `/health/live`, `/health/ready`, `/metrics`, `/about`, and
`POST /v1/verify`. Readiness returns a controlled 503 until the configured
corpus, BM25 index, dense index, models, gate/calibrator, verifier, and optional
persistence dependencies are compatible.

Docker commands:

```bash
docker compose up --build api streamlit
docker compose --profile local-llm up --build
docker compose --profile persistence up --build
docker compose run --rm tools python scripts/audit_spec_alignment.py --repository-root . --strict
```

Artifact and configuration mounts are read-only in online services, containers
run as a non-root user, and index construction never occurs during startup. See
[`docs/deployment.md`](docs/deployment.md) for deployment and observability
details.

Run the mandatory alignment gate and warm fake-component profile:

```bash
python scripts/audit_spec_alignment.py --repository-root . --strict
python scripts/profile_demo_pipeline.py
```

## Run tests

```bash
pytest
```

## Prepare SciFact

Place the SciFact source files in `data/raw/scifact/`:

- `corpus.jsonl`
- `claims_train.jsonl`
- `claims_dev.jsonl`
- `claims_test.jsonl` (optional; public test claims are unlabeled)

SciFact rationale sentence indices are zero-based and are preserved without
conversion because they refer directly to the original abstract sentence list.

Run:

```bash
python scripts/prepare_scifact.py \
  --input-dir data/raw/scifact \
  --output-dir data/processed/scifact
```

Generated files:

- `claims.jsonl`
- `documents.jsonl`
- `label_mapping.json`
- `quality_report.json`
- `manifest.json`

Raw and processed datasets are not committed to Git. The adapter performs no
network access and does not modify the raw files.

## Build the SciFact Evidence Corpus

First prepare the normalized SciFact dataset:

```bash
python scripts/prepare_scifact.py
```

Build the default sentence-level corpus:

```bash
python scripts/build_scifact_corpus.py \
  --input-dir data/processed/scifact \
  --output-root artifacts/corpora \
  --version scifact-v1
```

Generated artifacts:

- `documents.jsonl`
- `passages.jsonl`
- `gold_evidence.jsonl`
- `quality_report.json`
- `manifest.json`

SciFact source sentences are used directly as the default retrieval passages.
Passage offsets are Python-style half-open character offsets, and `token_count`
is an approximate whitespace-based count rather than a model tokenizer count.
No retrieval index is created by this command.

The current official SciFact release contains source sentences as long as 218
whitespace-delimited words. The default safety limit is 120 and deliberately
fails instead of splitting such sentences. To build that release while keeping
its sentence boundaries intact, pass `--max-passage-words 218` (or a larger
explicit limit).

Corpus versions are immutable. Use a new version name whenever the input data or
chunking configuration changes. Generated corpus artifacts under
`artifacts/corpora/` are not committed to ordinary Git history.

## Build the SciFact BM25 Index

First build the SciFact corpus as described above. Then create an immutable BM25
index:

```bash
python scripts/build_bm25_index.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --output-root artifacts/indexes/bm25 \
  --version scifact-v1-bm25-v1
```

Search the index:

```bash
python scripts/search_bm25.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --query "Vitamin D reduces respiratory infections." \
  --top-k 10
```

The BM25 baseline indexes passage text only. It uses NFKC Unicode normalization
and lowercase alphanumeric tokens, without stemming, lemmatization, stop-word
removal, embeddings, or medical synonym expansion. Index versions are immutable
and must match the corpus version and content hash from which they were built.

The builder deliberately rejects passages that contain no alphanumeric tokens,
because silently omitting them would break index-to-corpus alignment. The
current official SciFact release produces 865 punctuation-only sentence
passages with this corpus configuration. Before indexing that full release,
publish a new corpus version that addresses those source records explicitly;
the BM25 builder will not silently remove them.

BM25 scores are lexical ranking scores. They are not confidence values, medical
probabilities, or clinical conclusions.

## Evaluate SciFact BM25 Retrieval

Evaluate one labeled SciFact split against the corpus gold-evidence mappings:

```bash
python scripts/evaluate_bm25.py \
  --claims data/processed/scifact/claims.jsonl \
  --gold-evidence artifacts/corpora/scifact-v1/gold_evidence.jsonl \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --split dev \
  --output-dir artifacts/experiments/scifact-bm25-dev-v1
```

The immutable experiment directory contains `predictions.jsonl`, `metrics.json`,
`retrieval_errors.jsonl`, and `manifest.json`. The primary metric is complete
evidence-set Recall@K: every passage from at least one valid evidence set must
appear in the top K. Any-gold-passage Recall@K is reported separately as a
diagnostic, while MRR uses the first retrieved gold passage. Claims without
usable gold evidence are excluded and summarized by reason rather than counted
as retrieval failures.

## Build and Evaluate Dense Retrieval

Build a normalized `float32` embedding matrix and exact FAISS `IndexFlatIP`
index. The default model is `sentence-transformers/all-MiniLM-L6-v2`:

```bash
python scripts/build_dense_index.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --output-root artifacts/indexes/dense \
  --version scifact-v1-minilm-v1 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 64 \
  --device cpu
```

Search the dense index:

```bash
python scripts/search_dense.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --query "Vitamin D lowers the likelihood of respiratory illness." \
  --top-k 10
```

Evaluate dense retrieval using the same evidence-set metrics as BM25:

```bash
python scripts/evaluate_dense.py \
  --claims data/processed/scifact/claims.jsonl \
  --gold-evidence artifacts/corpora/scifact-v1/gold_evidence.jsonl \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --split dev \
  --ks 5 10 20 \
  --output-dir artifacts/experiments/scifact-dense-dev-v1
```

Dense index versions are immutable and contain `index.faiss`, `embeddings.npy`,
`passage_ids.json`, and `manifest.json`. The manifest records the exact corpus,
model identifier and optional revision, resolved dimension, normalization,
dtype, batch size, and artifact checksums. Dense cosine-similarity scores are
ranking values, not probabilities or medical confidence.

## Search and Compare Hybrid Retrieval

Hybrid retrieval runs BM25 and dense search independently and combines their
ranks with Reciprocal Rank Fusion. It never adds their incompatible raw scores.
Defaults are stored in `configs/retrieval/hybrid_scifact_v1.json`: 50 sparse
candidates, 50 dense candidates, 30 fused candidates, `rrf_k=60`, and five
final evidence passages for a future verifier pipeline.

Search with RRF:

```bash
python scripts/search_hybrid.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --bm25-index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --dense-index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --query "Vitamin D lowers respiratory infection risk." \
  --sparse-top-k 50 \
  --dense-top-k 50 \
  --fusion-top-k 30 \
  --rrf-k 60
```

Compare all three retrieval modes on exactly the same eligible claims:

```bash
python scripts/compare_retrieval.py \
  --claims data/processed/scifact/claims.jsonl \
  --gold-evidence artifacts/corpora/scifact-v1/gold_evidence.jsonl \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --bm25-index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --dense-index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --split dev \
  --ks 5 10 20 \
  --output-dir artifacts/experiments/scifact-retrieval-comparison-v1
```

The immutable comparison directory contains method-specific predictions,
`metrics.json`, `comparison.csv`, `retrieval_errors.jsonl`, and `manifest.json`.
RRF scores are ranking values rather than probabilities or confidence. Hybrid
retrieval does not assume or guarantee an improvement over either component.

## Rerank Hybrid Evidence

The optional cross-encoder stage scores only the claim and each candidate's
passage text. It preserves BM25, dense, RRF, and pre-rerank ranks while reducing
the default 20-candidate pool to five final evidence passages. Defaults live in
`configs/reranking/scifact_cross_encoder_v1.json`.

Search with reranking:

```bash
python scripts/search_reranked.py \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --bm25-index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --dense-index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --query "Vitamin D lowers respiratory infection risk." \
  --candidate-count 20 \
  --top-k 5 \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2
```

Pass `--disable-reranking` to use the first RRF-ranked passages without loading
the cross-encoder.

Run the retrieval ablation:

```bash
python scripts/compare_reranking.py \
  --claims data/processed/scifact/claims.jsonl \
  --gold-evidence artifacts/corpora/scifact-v1/gold_evidence.jsonl \
  --corpus-dir artifacts/corpora/scifact-v1 \
  --bm25-index-dir artifacts/indexes/bm25/scifact-v1-bm25-v1 \
  --dense-index-dir artifacts/indexes/dense/scifact-v1-minilm-v1 \
  --split dev \
  --candidate-count 20 \
  --final-evidence-k 5 \
  --output-dir artifacts/experiments/scifact-reranking-comparison-v1
```

The immutable experiment contains hybrid and reranked predictions, retrieval
and classification metric files, a comparison CSV, rank-change records, errors,
and a manifest. `--max-claims 20` creates a manifest-marked sample experiment.
Because this repository does not yet include the verifier application,
the CLI records classification metrics as unavailable. The Python evaluator
accepts an evidence-only verifier implementation and computes accuracy,
macro/per-label metrics, a confusion matrix, and verification latency; this path
is covered by offline tests. Cross-encoder scores are ranking values, not
probabilities or confidence.

## Build the unified medical dataset and corpus

After the SciFact, HealthVer, and PUBHEALTH adapters have produced normalized
artifacts, validate and merge them without changing source IDs, labels, splits,
explanations, or evidence groups:

```bash
python scripts/build_unified_dataset.py \
  --scifact-dir data/processed/scifact \
  --healthver-dir data/processed/healthver \
  --pubhealth-dir data/processed/pubhealth \
  --output-root artifacts/datasets \
  --version medical-dataset-v1
```

The immutable dataset version contains `claims.jsonl`, `documents.jsonl`,
`evidence_relations.jsonl`, `label_schema.json`, `quality_report.json`, and
`manifest.json`. Its internal labels are `SUPPORTS`, `REFUTES`,
`NOT_ENOUGH_INFO`, and `MIXED`; mappings are loaded from each adapter's
`label_mapping.json`.

Build one retrieval corpus across all three datasets:

```bash
python scripts/build_combined_corpus.py \
  --dataset-dir artifacts/datasets/medical-dataset-v1 \
  --output-root artifacts/corpora \
  --version medical-corpus-v1
```

Gold explanations remain on claim records and are never made into passages.
SciFact uses abstract sentences, HealthVer uses supplied evidence-item or
sentence boundaries, and PUBHEALTH uses supplied evidence sentences or a
deterministic rule-based fallback. The existing `build_bm25_index.py` and
`build_dense_index.py` commands accept the resulting corpus unchanged.

## Evidence sufficiency and compound claims

US-013 adds a configurable gate before verification. Only reranked passages at
or above the calibrated score threshold proceed to a verifier; no candidates or
weak evidence produce a normal `NOT_ENOUGH_INFO` result without a verifier call.
The checked-in [default configuration](configs/evidence_gate/default_v1.json) is
a schema-valid bootstrap configuration. Calibrate the deployed threshold from a
project development split:

```bash
python scripts/calibrate_evidence_gate.py \
  --claims artifacts/datasets/medical-dataset-v1/claims.jsonl \
  --gold-evidence artifacts/corpora/medical-corpus-v1/gold_evidence.jsonl \
  --split-manifest artifacts/splits/medical-splits-v1/split_manifest.json \
  --predictions artifacts/experiments/reranked-dev-v1/predictions.jsonl \
  --split dev \
  --output-root artifacts/evidence-gates \
  --version evidence-gate-v1
```

Calibration rejects the test split and writes an immutable `config.json`,
`threshold_results.csv`, `metrics.json`, and `manifest.json`. Selection uses
binary sufficiency Macro-F1, then insufficient-evidence recall, then the higher
threshold.

The verification pipeline supports decomposition modes `off`, `auto`, and
`always`, at most four stable components, independent retrieval and gating per
component, and deterministic aggregation. Uniform component verdicts retain
their label; different verdicts produce `MIXED`. Aggregate confidence is the
minimum component estimate and is not a clinical probability.

Evaluate structured predictions without provider or model downloads:

```bash
python scripts/evaluate_gate_and_decomposition.py \
  --predictions artifacts/experiments/gate-decomposition-test-v1-input.jsonl \
  --output-dir artifacts/experiments/gate-decomposition-test-v1
```

The hardened application layer wraps these injectable contracts with scope
routing, privacy-safe request tracing, bounded operational metrics, readiness
checks, and an evidence-only tool-free provider boundary.

## Explanation attribution, confidence calibration, and experiments

Verifier explanations are validated before a successful component result is
returned. Validation enforces authoritative supplied passage IDs, evidence
coverage, 5–120 words by default, verdict-consistent wording, explicit
insufficiency for `NOT_ENOUGH_INFO`, component disagreement for `MIXED`, and
medical-advice and unsupported-causation checks. LLM verifiers may make one
explicit correction attempt; classifier templates fail directly if invalid.
Corpus titles, URLs, publication years, text, and source types are always
resolved from the versioned corpus rather than verifier-generated metadata.

Export a deterministic, stratified manual-review worksheet:

```bash
python scripts/export_explanation_review.py \
  --predictions artifacts/experiments/final-matrix-v1/predictions.jsonl \
  --output artifacts/reviews/explanation-review-v1.csv \
  --sample-size 100 \
  --seed 42 \
  --stratify-by dataset label correctness
```

Fit a probability-of-correctness calibrator using project-development records
only:

```bash
python scripts/fit_confidence_calibrator.py \
  --predictions artifacts/experiments/final-matrix-dev-v1/predictions.jsonl \
  --split-manifest artifacts/splits/medical-splits-v1/split_manifest.json \
  --method logistic \
  --output-root artifacts/calibration \
  --version confidence-calibrator-v1
```

Supported methods are `none`, `logistic`, and `isotonic`. The default logistic
features are raw verifier confidence, reranker score summaries, gate state,
selected passage/document counts, verifier kind, predicted-label indicators,
and decomposition state. Claim IDs, gold labels, and dataset test outcomes are
never features. The calibrated number estimates validation-observed prediction
correctness—not whether a medical claim is true and not clinical certainty.

Evaluate a fixed calibrator without refitting it:

```bash
python scripts/evaluate_calibration.py \
  --predictions artifacts/experiments/final-matrix-test-v1/predictions.jsonl \
  --calibrator artifacts/calibration/confidence-calibrator-v1 \
  --output-dir artifacts/experiments/calibration-test-v1
```

This produces raw/calibrated Brier score, ECE, MCE, log loss, reliability bins,
accuracy/coverage and risk/coverage tables, calibrated predictions, and PNG
figures. Calibrator pickles are loaded only after manifest and checksum checks.

Run the complete versioned experiment matrix sequentially:

```bash
python scripts/run_experiments.py \
  --config-dir configs/experiments \
  --output-root artifacts/experiments \
  --run-group final-matrix-v1 \
  --continue-on-error
```

The 12 configuration files expand to 17 concrete runs. They cover LLM-only,
BM25, dense, hybrid, reranking, classifier, oracle evidence, dataset-specific
and combined classifiers, three/four labels, gate ablation, and decomposition
ablation. Oracle runs remain explicitly marked. Run groups are immutable and
reuse requires matching configuration hashes and every recorded artifact/model
version. Set `MEDCLAIM_CODE_REVISION` in packaged runs to include a source
revision in reuse validation.

Because this repository does not yet contain the US-014 classifier or an LLM
provider, the checked-in matrix uses validated precomputed experiment-input
artifacts under `artifacts/experiment-inputs/`. Missing inputs become controlled,
isolated failed runs; the Python runner also accepts injected executors for the
existing pipeline when those implementations are available.
