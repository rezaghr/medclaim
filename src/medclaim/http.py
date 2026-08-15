"""Process-wide HTTP connection pool for local service calls."""

from functools import cache

import httpx


@cache
def local_http_client() -> httpx.Client:
    """Reuse TCP connections across Ollama requests and readiness checks."""
    return httpx.Client(trust_env=False)
