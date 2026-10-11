"""TerraMind AI - FastAPI application.  Run: uvicorn src.main:app --reload"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .config import settings
from .models.yield_model import ModelUnavailable
from .rag.retriever import RetrievalError
from .rag.store import DatasetUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
MAX_BODY = 64 * 1024

app = FastAPI(
    title="TerraMind AI",
    description="AI-powered agriculture crop advisory: hybrid RAG over historical cases, yield prediction, "
                "rule-based suitability, multi-agent orchestration and LLM-as-judge validation. Decision support only.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def limit_body(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY:
        return JSONResponse(status_code=413, content={"detail": f"Payload too large (max {MAX_BODY} bytes)"})
    return await call_next(request)


@app.exception_handler(DatasetUnavailable)
async def _ds(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "dataset_unavailable"})


@app.exception_handler(ModelUnavailable)
async def _mu(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "model_unavailable"})


@app.exception_handler(RetrievalError)
async def _re(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "retrieval_unavailable"})


@app.exception_handler(ValueError)
async def _ve(_, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc), "code": "invalid_request"})


app.include_router(router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
