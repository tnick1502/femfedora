"""HTTP API. Run: uvicorn src.ring_health_v3.api:app --port 8790. No request bodies or health data are logged."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .engine import RingHealthV3
from .schema import AnalyzeRequest


@asynccontextmanager
async def lifespan(app):
    if os.getenv("RING_HEALTH_DEPLOYMENT") == "production" and not os.getenv("RING_HEALTH_API_KEY"):
        raise RuntimeError("Set RING_HEALTH_API_KEY in production")
    app.state.engine = RingHealthV3()
    yield


app = FastAPI(title="Ring Health v3 (research/demo)", version="3.0.0", lifespan=lifespan,
              description="Cycle (fertile window, ovulation, next period) and menopausal-transition analysis from smart-ring nights. Synthetic-trained; clinical validity not established.")


@app.exception_handler(RequestValidationError)
async def _invalid(request, exc):
    return JSONResponse(status_code=422, content={"error": "invalid_input", "details": [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]})


@app.middleware("http")
async def _limit(request: Request, call_next):
    if request.method == "POST":
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 3_000_000:
                return JSONResponse(status_code=413, content={"error": "request_too_large"})
        request._body = bytes(body)
    return await call_next(request)


def _auth(authorization: str | None = Header(default=None)):
    token = os.getenv("RING_HEALTH_API_KEY")
    if token and not hmac.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "Unauthorized")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    e = getattr(app.state, "engine", None)
    if e is None:
        raise HTTPException(503, "not ready")
    return {"status": "ready", "version": e.registry["version"], "registry_sha256": e.digest}


@app.post("/v3/analyze", dependencies=[Depends(_auth)])
def analyze(payload: AnalyzeRequest):
    try:
        return app.state.engine.analyze(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
