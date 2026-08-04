from pathlib import Path

import pytest

from medclaim.runtime.configuration import (
    RuntimeConfigurationError,
    load_runtime_settings,
)


BASE = """configuration_id: test-v1
environment: test
llm_provider: fake
llm_model: fake-v1
reranker_model: reranker-v1
persistence_enabled: false
persist_claim_text: false
persist_explanation: false
"""


def write(tmp_path: Path, suffix: str = "") -> Path:
    path = tmp_path / "deployment.yaml"
    path.write_text(BASE + suffix, encoding="utf-8")
    return path


def test_unknown_yaml_key_fails(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="Extra inputs"):
        load_runtime_settings(write(tmp_path, "unknown_setting: true\n"), environ={})


def test_privacy_defaults_and_hash_do_not_contain_secrets(tmp_path):
    settings = load_runtime_settings(
        write(tmp_path),
        environ={"LLM_API_KEY": "top-secret", "DATABASE_URL": "postgres://secret"},
    )
    assert not settings.persistence_enabled
    assert not settings.persist_claim_text
    assert "top-secret" not in settings.configuration_hash
    assert "postgres" not in settings.model_dump_json()


def test_strict_mode_rejects_missing_artifact_values(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="MEDCLAIM_CORPUS_DIR"):
        load_runtime_settings(write(tmp_path), environ={}, strict=True)


def test_content_persistence_requires_persistence(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="CONFIG_PRIVACY_INVALID"):
        load_runtime_settings(write(tmp_path, "persist_claim_text: true\n"), environ={})


def test_ollama_environment_configuration(tmp_path):
    configured = load_runtime_settings(
        write(tmp_path),
        environ={
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "dolphin-llama3:8b",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "LLM_TIMEOUT_SECONDS": "180",
        },
    )
    assert configured.llm_provider == "ollama"
    assert configured.llm_model == "dolphin-llama3:8b"
    assert configured.ollama_base_url == "http://127.0.0.1:11434"
    assert configured.llm_timeout_seconds == 180


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="llm_provider"):
        load_runtime_settings(
            write(tmp_path), environ={"LLM_PROVIDER": "unknown"}
        )
