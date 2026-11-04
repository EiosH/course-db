"""HTTP service: /embed + /rerank (+ /health, /ready)."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import embed_texts, is_ready, load_error, rerank_pairs, warmup


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    is_query: bool = False


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(default_factory=list)


class RerankResponse(BaseModel):
    scores: list[float]


def _require_ready() -> None:
    if is_ready():
        return
    detail = load_error() or "models still loading"
    raise HTTPException(status_code=503, detail=detail)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Bind HTTP first; load weights in background so /health stays reachable.
    threading.Thread(target=warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="embedding", lifespan=lifespan)


@app.get("/health")
def health():
    """Process is up (does not mean models are loaded)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Models loaded and ready to serve /embed + /rerank."""
    if is_ready():
        return {"status": "ready"}
    raise HTTPException(
        status_code=503,
        detail=load_error() or "models still loading",
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    _require_ready()
    return EmbedResponse(vectors=embed_texts(req.texts, is_query=req.is_query))


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    _require_ready()
    return RerankResponse(scores=rerank_pairs(req.query, req.documents))
