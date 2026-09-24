"""Ollama embed/chat and local reranker helpers."""

import time

import requests

from config import (
    EMBED_BATCH,
    EMBED_MODEL,
    OLLAMA_CHAT_RETRIES,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    RERANK_MODEL,
)
from tracing import observation

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"loading rerank model {RERANK_MODEL}...")
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker


def embed(texts, *, retries: int = 5):
    """Embed texts via Ollama in batches.

    Retries transient failures (Windows socket buffer / ephemeral-port
    exhaustion after long runs is a common cause of Ollama 400s).
    """
    out = []
    total = len(texts)
    session = requests.Session()
    try:
        for i in range(0, total, EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    r = session.post(
                        f"{OLLAMA_URL}/api/embed",
                        json={"model": EMBED_MODEL, "input": batch},
                        timeout=300,
                    )
                    if not r.ok:
                        # Socket exhaustion often surfaces as HTTP 400 from Ollama.
                        transient = r.status_code in (400, 429, 500, 502, 503)
                        msg = f"Ollama embed {r.status_code}: {r.text}"
                        if transient and attempt < retries:
                            wait = min(30, 2 ** attempt)
                            print(
                                f"embed error (attempt {attempt}/{retries}), "
                                f"retry in {wait}s: {msg[:200]}"
                            )
                            time.sleep(wait)
                            last_err = RuntimeError(msg)
                            continue
                        raise RuntimeError(msg)
                    out.extend(r.json()["embeddings"])
                    last_err = None
                    break
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    last_err = e
                    if attempt < retries:
                        wait = min(30, 2 ** attempt)
                        print(
                            f"embed timeout/error "
                            f"(attempt {attempt}/{retries}), "
                            f"retry in {wait}s..."
                        )
                        time.sleep(wait)
            if last_err is not None:
                raise last_err
            done = min(i + EMBED_BATCH, total)
            print(f"embed {done}/{total}")
            # Brief pause every ~1k chunks to let Windows reclaim sockets.
            if done % 1024 < EMBED_BATCH and done < total:
                time.sleep(0.5)
    finally:
        session.close()
    return out


def ollama_chat(
    messages,
    *,
    format=None,
    timeout=OLLAMA_CHAT_TIMEOUT,
    temperature=OLLAMA_TEMPERATURE,
    model: str | None = None,
    name: str = "llm",
):
    """Call Ollama /api/chat with retries; think=False avoids qwen thinking overhead."""
    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "seed": OLLAMA_SEED,
        },
    }
    if format is not None:
        payload["format"] = format

    # require_parent=True: never open a standalone root trace for an LLM call.
    with observation(
        name=name,
        as_type="generation",
        require_parent=True,
        model=model,
        input=messages,
        model_parameters={
            "temperature": temperature,
            "seed": OLLAMA_SEED,
            "format": format,
        },
    ) as gen:
        last_err = None
        for attempt in range(1, OLLAMA_CHAT_RETRIES + 1):
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                content = r.json()["message"]["content"].strip()
                if gen is not None:
                    gen.update(output=content)
                return content
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as e:
                last_err = e
                if attempt < OLLAMA_CHAT_RETRIES:
                    wait = 5 * attempt
                    print(
                        f"ollama chat timeout/error "
                        f"(attempt {attempt}/{OLLAMA_CHAT_RETRIES}), "
                        f"retry in {wait}s..."
                    )
                    time.sleep(wait)
        if gen is not None:
            gen.update(
                level="ERROR",
                status_message=str(last_err),
            )
        raise last_err
