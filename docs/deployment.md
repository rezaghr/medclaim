# Deployment and operations

MedClaimRAG verifies textual claims against a limited indexed corpus. It is an educational research prototype. It is not a doctor, diagnostic system, treatment recommender, or emergency service. Its results are not medical advice.

Copy `.env.example` to `.env`, set artifact directories and a provider model, and keep API keys only in `.env` or the deployment secret mechanism. Raw claims and explanations are not persisted by default. The development configuration uses the local Ollama model `dolphin-llama3:8b`; when the API runs directly from `.venv`, `OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`. Docker Compose routes the API container to the host Ollama endpoint through `host.docker.internal`.

Start the external-provider deployment with `docker compose up --build api streamlit`. Start the optional local provider with `docker compose --profile local-llm up --build`; it is otherwise absent. Set `POSTGRES_PASSWORD` and start PostgreSQL with `docker compose --profile persistence up --build`. Run offline tooling with `docker compose run --rm tools <command>`.

The API exposes `/health/live`, `/health/ready`, and `/metrics`. Liveness only tests the process. Readiness validates the corpus manifest and passage count, BM25 and dense corpus references, dense dimensions, reranker and verifier configuration, gate/calibrator versions, and persistence configuration. Missing or incompatible artifacts return HTTP 503 without rebuilding anything.

Corpus, index, model, and configuration mounts are read-only. Experiment outputs are writable. API and Streamlit containers run as the unprivileged `medclaim` user with a read-only root filesystem.

Structured logs correlate stages by `request_id` and include only a SHA-256 claim hash and length. They exclude claim/evidence text, prompts, raw provider responses, secrets, network identifiers, and personal identifiers. Metrics use bounded operational labels only.

The demo profiler uses fake in-process components unless a future provider-specific profiler is supplied. Its report explicitly identifies that measurement mode; do not present it as production-model latency.
