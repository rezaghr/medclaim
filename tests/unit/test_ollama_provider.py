import httpx
import pytest

from medclaim.runtime.ollama import OllamaProvider, OllamaProviderError


def test_ollama_provider_requests_structured_non_streaming_output(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"trust_env": False}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"response": '{"verdict":"NOT_ENOUGH_INFO"}'},
            )

    monkeypatch.setattr(httpx, "Client", Client)
    schema = {"type": "object"}
    result = OllamaProvider("dolphin-llama3:8b", "http://localhost:11434").complete(
        prompt="evidence only", response_schema=schema
    )

    assert result == '{"verdict":"NOT_ENOUGH_INFO"}'
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "dolphin-llama3:8b"
    assert captured["json"]["format"] == schema
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["temperature"] == 0


def test_ollama_provider_rejects_credentialed_base_url():
    with pytest.raises(OllamaProviderError, match="CONFIG_INVALID"):
        OllamaProvider("model", "http://user:password@localhost:11434")
