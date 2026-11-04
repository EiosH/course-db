"""Ingest infra settings (embed service, Qdrant)."""

import os

EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8080").rstrip("/")
BM25_MODEL = "Qdrant/bm25"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
COLLECTION_NAME = "docs"
UPSERT_BATCH = 64
