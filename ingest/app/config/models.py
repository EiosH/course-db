"""Ingest infra settings (embed, Qdrant)."""

import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda").strip()
EMBED_BATCH = 32
BM25_MODEL = "Qdrant/bm25"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
COLLECTION_NAME = "docs"
UPSERT_BATCH = 64
