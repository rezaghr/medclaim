# MedClaimRAG

MedClaimRAG verifies a textual medical or public-health claim against a small,
indexed evidence corpus. It combines BM25 and FAISS retrieval, reranks the
evidence, and asks a locally installed Ollama model for a structured verdict.

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

On the first start, the `artifacts-init` container creates the bundled
nine-passage evidence corpus and its BM25/FAISS indexes in a persistent Docker
volume. Later starts validate and reuse that volume.

Useful commands:

```bash
docker compose ps
docker compose logs -f artifacts-init api streamlit
docker compose down
```

`docker compose down -v` also deletes the generated evidence/index volume, so
the next startup must build it again.

## Configuration

The main settings are documented in `.env.example`. The defaults expect Ollama
at `http://host.docker.internal:11434` and expose ports 8000 and 8501.

The bundled corpus is intentionally small. A verdict of `NOT_ENOUGH_INFO` means
the indexed passages did not provide enough direct evidence for the claim.
