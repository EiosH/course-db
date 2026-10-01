"""Model / infra settings (Ollama, embed, rerank, Qdrant, Langfuse)."""

import os

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3.8:27b"
EMBED_MODEL = "bge-m3"
BM25_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_CHAT_RETRIES = 3
# Plan / resolve-extract / rewrite stay greedy for stable JSON and retrieval.
OLLAMA_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2  # answer() only: slightly warmer tone
OLLAMA_SEED = 42
# LLM 上下文窗口（输入 + 输出 token），与具体推理后端无关；上下文预算据此计算。
# Ollama 下传为 options.num_ctx —— 不设时用默认值（约 2048~4096），
# 超出部分会从消息开头被静默截掉（system prompt 最先丢）。
LLM_CONTEXT_WINDOW = 16384
CONTEXT_WARN_RATIO = 0.9  # 输入+输出超过上下文窗口的这个比例就打印警告
EMBED_BATCH = 32
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"
UPSERT_BATCH = 64

# 选课路由用小模型（未设则跟主模型）；例如 COURSE_ROUTE_MODEL=qwen2.5:3b
COURSE_ROUTE_MODEL = os.getenv("COURSE_ROUTE_MODEL") or OLLAMA_MODEL

# Langfuse: keys / base URL from .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# LANGFUSE_BASE_URL). Set LANGFUSE_ENABLED=true to turn tracing on.
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "").lower() in ("1", "true", "yes")
