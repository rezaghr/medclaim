"""Structured request tracing that excludes raw user and secret content."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

_PROHIBITED_KEYS = {
    "claim",
    "claim_text",
    "evidence",
    "evidence_text",
    "system_prompt",
    "provider_response",
    "api_key",
    "environment_values",
    "ip_address",
    "user_agent",
}


def safe_claim_hash(claim: str) -> str:
    return f"sha256:{hashlib.sha256(claim.encode('utf-8')).hexdigest()}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def configure_json_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("medclaim.request")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class RequestTracer:
    def __init__(self, logger: logging.Logger, base_fields: dict[str, Any]) -> None:
        self.logger = logger
        self.base_fields = dict(base_fields)

    def emit(self, stage: str, **fields: Any) -> None:
        prohibited = _PROHIBITED_KEYS & set(fields)
        if prohibited:
            raise ValueError(
                f"TRACE_PRIVACY_VIOLATION: Prohibited field {sorted(prohibited)[0]!r}."
            )
        payload = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **self.base_fields,
            **fields,
        }
        self.logger.info(payload)
