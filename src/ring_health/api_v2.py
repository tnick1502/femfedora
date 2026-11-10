"""ASGI deployment entry point. No health data or request bodies are logged."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends,FastAPI,Header,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .engine_v2 import Engine
from .schema_v2 import AnalyzeRequest


@asynccontextmanager
async def lifespan(app):
    if os.getenv("RING_HEALTH_DEPLOYMENT")=="production" and not os.getenv("RING_HEALTH_API_KEY"):
        raise RuntimeError("Set RING_HEALTH_API_KEY for production deployment")
    app.state.engine=Engine()
    yield


app=FastAPI(title="Ring Health Research API",version="2.0.0",lifespan=lifespan,description="Missing-aware reproductive research models. Clinical performance on Oura is not established.")


@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    # Avoid echoing raw health input or NaN into error payloads/logs.
    return JSONResponse(status_code=422,content={"error":"invalid_input","details":[{"location":list(e["loc"]),"message":e["msg"],"type":e["type"]} for e in exc.errors()]})


@app.middleware("http")
async def size_limit(request:Request,call_next):
    if request.method=="POST":
        try:
            n=int(request.headers.get("content-length","0"))
        except ValueError:return JSONResponse(status_code=400,content={"error":"invalid_content_length"})
        if n>2_000_000:return JSONResponse(status_code=413,content={"error":"request_too_large"})
        # Enforce the same bound for chunked bodies without Content-Length.
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>2_000_000:return JSONResponse(status_code=413,content={"error":"request_too_large"})
        request._body=bytes(body)
    return await call_next(request)


def authorize(authorization:str|None=Header(default=None)):
    token=os.getenv("RING_HEALTH_API_KEY")
    if token and not hmac.compare_digest(authorization or "","Bearer "+token):raise HTTPException(401,"Unauthorized")


@app.get("/health/live")
def live():return {"status":"alive"}


@app.get("/health/ready")
def ready():
    engine=getattr(app.state,"engine",None)
    if engine is None:raise HTTPException(503,"Artifacts not loaded")
    return {"status":"ready","version":"ring-health-v2","artifact_manifest_sha256":engine.model_digest}


@app.post("/v2/analyze",dependencies=[Depends(authorize)])
def analyze(payload:AnalyzeRequest):
    try:return app.state.engine.analyze(payload)
    except ValueError as e:raise HTTPException(422,str(e)) from e
