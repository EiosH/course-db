"""Lazy-loaded embed + rerank models."""

from __future__ import annotations

import threading

from config import (
    EMBED_BATCH,
    EMBED_DEVICE,
    EMBED_DIM,
    EMBED_MODEL,
    RERANK_DEVICE,
    RERANK_MODEL,
)

_embedder = None
_reranker = None
_ready = False
_load_error: str | None = None
_lock = threading.Lock()


def is_ready() -> bool:
    return _ready


def load_error() -> str | None:
    return _load_error


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        where = EMBED_DEVICE or "cpu"
        print(f"loading embed model {EMBED_MODEL} on {where}...")
        kwargs = {"processor_kwargs": {"padding_side": "left"}}
        if EMBED_DEVICE:
            kwargs["device"] = EMBED_DEVICE
        _embedder = SentenceTransformer(EMBED_MODEL, **kwargs)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        where = RERANK_DEVICE or "auto"
        print(f"loading rerank model {RERANK_MODEL} on {where}...")
        kwargs = {"trust_remote_code": True}
        if RERANK_DEVICE:
            kwargs["device"] = RERANK_DEVICE
        _reranker = CrossEncoder(RERANK_MODEL, **kwargs)
    return _reranker


def warmup() -> None:
    """Load both models; safe to call from a background thread."""
    global _ready, _load_error
    with _lock:
        if _ready:
            return
        try:
            get_embedder()
            get_reranker()
            _ready = True
            _load_error = None
            print("embedding service ready (dense + rerank)")
        except Exception as e:
            _ready = False
            _load_error = str(e)
            print(f"embedding warmup failed: {e}")
            raise


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    out: list[list[float]] = []
    total = len(texts)
    encode_kwargs = {
        "normalize_embeddings": True,
        "show_progress_bar": False,
        "truncate_dim": EMBED_DIM,
    }
    if is_query:
        encode_kwargs["prompt_name"] = "query"
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vecs = model.encode(batch, **encode_kwargs)
        out.extend(vecs.tolist())
        print(f"embed {min(i + EMBED_BATCH, total)}/{total}")
    return out


def rerank_pairs(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    scores = get_reranker().predict([(query, doc) for doc in documents])
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(s) for s in scores]
