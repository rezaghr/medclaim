"""Safe API for the MedClaimRAG modular monolith."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from medclaim.runtime import RuntimeSettings, VerificationService, load_runtime_settings
from medclaim.runtime.pipeline import build_runtime_pipeline
from medclaim.runtime.readiness import readiness_snapshot
from medclaim.runtime.service import VerificationServiceError
from medclaim.safety import MANDATORY_SAFETY_DISCLAIMER


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, max_length=5000)


def _default_settings() -> RuntimeSettings:
    path = Path(os.environ.get("MEDCLAIM_CONFIG", "configs/deployment/default.yaml"))
    strict = os.environ.get("MEDCLAIM_STRICT_STARTUP", "false").casefold() == "true"
    return load_runtime_settings(path, strict=strict)


def create_app(
    *,
    settings: RuntimeSettings | None = None,
    service: VerificationService | None = None,
) -> FastAPI:
    resolved_settings = settings or _default_settings()
    pipeline_error: str | None = None
    if service is None:
        try:
            pipeline = build_runtime_pipeline(resolved_settings)
        except Exception as exc:
            pipeline = None
            pipeline_error = str(exc)
        resolved_service = VerificationService(resolved_settings, pipeline)
    else:
        resolved_service = service
    app = FastAPI(
        title="MedClaimRAG API",
        version="1.0.0",
        description=MANDATORY_SAFETY_DISCLAIMER,
    )
    app.state.settings = resolved_settings
    app.state.service = resolved_service
    app.state.pipeline_error = pipeline_error

    @app.get("/health/live", tags=["operations"])
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["operations"])
    def readiness(response: Response) -> dict:
        snapshot = readiness_snapshot(resolved_settings)
        if pipeline_error is not None:
            snapshot["checks"]["pipeline"] = {
                "ready": False,
                "error": pipeline_error,
            }
            snapshot["failed_checks"].append("pipeline")
            snapshot["status"] = "not_ready"
        else:
            snapshot["checks"]["pipeline"] = {"ready": True, "detail": "loaded"}
        if snapshot["status"] != "ready":
            response.status_code = 503
        return snapshot

    @app.get("/metrics", tags=["operations"], response_class=Response)
    def metrics() -> Response:
        return Response(resolved_service.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/about", tags=["safety"])
    def about() -> dict[str, str]:
        return {
            "project": "MedClaimRAG",
            "safety_disclaimer": MANDATORY_SAFETY_DISCLAIMER,
        }

    @app.post("/v1/verify", tags=["verification"])
    def verify(
        request: VerificationRequest,
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> dict:
        if x_request_id is not None and (
            len(x_request_id) > 128 or any(char.isspace() for char in x_request_id)
        ):
            raise HTTPException(status_code=400, detail="INVALID_REQUEST_ID")
        try:
            return resolved_service.verify(request.claim, x_request_id)
        except VerificationServiceError as exc:
            code = str(exc).split(":", 1)[0]
            raise HTTPException(status_code=503, detail=code) from exc

    return app


app = create_app()
