"""Embed service settings (dense embed + rerank)."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda").strip()
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cuda").strip()
HOST = os.getenv("EMBED_HOST", "0.0.0.0")
PORT = int(os.getenv("EMBED_PORT", "8080"))
