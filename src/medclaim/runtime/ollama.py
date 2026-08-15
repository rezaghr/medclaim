"""Tool-free local Ollama provider for evidence-bound verification."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from medclaim.http import local_http_client


class OllamaProviderError(Exception):
    """Raised when the local Ollama endpoint cannot satisfy the provider contract."""


class OllamaProvider:
    """Minimal Ollama `/api/generate` client with structured output enabled."""

    def __init__(self, model: str, base_url: str, timeout_seconds: float = 120.0) -> None:
        if not isinstance(model, str) or not model.strip():
            raise OllamaProviderError("OLLAMA_CONFIG_INVALID: Model must be non-empty.")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise OllamaProviderError(
                "OLLAMA_CONFIG_INVALID: Base URL must be an HTTP(S) origin without credentials."
            )
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(
        self, *, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any] | str:
        try:
            response = local_http_client().post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": response_schema,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError("OLLAMA_TIMEOUT: Local model request timed out.") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaProviderError(f"OLLAMA_REQUEST_FAILED: {exc}.") from exc
        generated = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(generated, str) or not generated.strip():
            raise OllamaProviderError(
                "OLLAMA_RESPONSE_INVALID: Ollama returned no generated response."
            )
        return generated

    def available_models(self) -> set[str]:
        """Return the model names installed on the configured endpoint."""
        try:
            response = local_http_client().get(
                f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 5.0)
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaProviderError(f"OLLAMA_UNAVAILABLE: {exc}.") from exc
        return {
            item.get("name")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        } if isinstance(payload, dict) else set()

    def check(self, available_models: set[str] | None = None) -> str:
        """Confirm that the configured model is installed on the endpoint."""
        names = self.available_models() if available_models is None else available_models
        if self.model not in names:
            raise OllamaProviderError(
                f"OLLAMA_MODEL_NOT_FOUND: Model {self.model!r} is not installed."
            )
        return self.model
