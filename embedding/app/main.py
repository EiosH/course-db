"""HTTP service: /embed + /rerank (+ /health)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from models import embed_texts, rerank_pairs, warmup


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup()
    yield


app = FastAPI(title="embedding", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    return EmbedResponse(vectors=embed_texts(req.texts, is_query=req.is_query))


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    return RerankResponse(scores=rerank_pairs(req.query, req.documents))
