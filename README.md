# MedClaimRAG

MedClaimRAG verifies a textual medical or public-health claim against indexed
SciFact, HealthVer, and PUBHEALTH evidence. It combines BM25 and FAISS
retrieval, reranks the evidence, and asks a locally installed Ollama model for
a structured verdict.

For a detailed course on the architecture and the engineering concepts behind
this codebase, read [STUDY_GUIDE.md](STUDY_GUIDE.md).

The application is intended to run only through Docker Compose. Docker does not
install Ollama or download models.

## Requirements

- Docker with Compose
- Ollama running on the host and reachable by Docker
- `dolphin-llama3:8b`
- `nomic-embed-text:latest`

Confirm the models are available:

```bash
ollama list
```

If your model names differ, copy `.env.example` to `.env` and change them there.

## Run

```bash
docker compose up --build
```

Open Streamlit at <http://localhost:8501>. The API is available at
<http://localhost:8000>, with readiness at `/health/ready` and verification at
`POST /v1/verify`.

On the first start, the `artifacts-init` container downloads checksum-pinned
official releases, normalizes all usable records, chunks long documents, and
builds the BM25/FAISS indexes in a persistent Docker volume. This first build is
large and can take considerable time because your host Ollama must embed about
100,000 passages. Later starts validate and reuse the completed artifacts.

Useful commands:

```bash
docker compose ps
docker compose logs -f artifacts-init api streamlit
docker compose down
```

`docker compose down -v` also deletes the downloaded datasets and generated
indexes, so the next startup must download and build them again.

## Configuration

The main settings are documented in `.env.example`. The defaults expect Ollama
at `http://host.docker.internal:11434` and expose ports 8000 and 8501. The
Streamlit client allows up to 300 seconds for a complete multi-stage retrieval
and verification request; override `MEDCLAIM_API_TIMEOUT_SECONDS` if needed.

The normalized corpus contains 18,172 documents, 101,491 passages, and 27,994
claims from the three releases. The original downloads are retained in the
artifact volume. PUBHEALTH contains 26 empty and 11 structurally malformed
source rows that cannot become evidence passages; they remain in the original
ZIP and are counted in `quality_report.json` rather than being silently
invented or repaired.

A verdict of `NOT_ENOUGH_INFO` means the indexed passages did not provide
enough direct evidence for the claim.
