"""Embed, chat (Ollama or vLLM), and local reranker helpers."""

import re
import time

import requests

from config import (
    CHAT_MODEL,
    CONTEXT_WARN_RATIO,
    EMBED_BATCH,
    EMBED_DEVICE,
    EMBED_DIM,
    EMBED_MODEL,
    LLM_BACKEND,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    OLLAMA_CHAT_RETRIES,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    RERANK_DEVICE,
    RERANK_MODEL,
    VLLM_BASE_URL,
)
from tracing import observation

_reranker = None
_embedder = None


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


def embed(texts, *, retries: int = 5, is_query: bool = False):
    """Embed texts locally with Qwen3-Embedding-4B. `retries` unused; kept for callers.

    Queries should pass is_query=True so the model's query instruction prompt is applied.
    """
    del retries
    if not texts:
        return []
    model = get_embedder()
    out = []
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
        done = min(i + EMBED_BATCH, total)
        print(f"embed {done}/{total}")
    return out


def _usage(data: dict, name: str) -> dict:
    """Token counts from an Ollama or OpenAI response; warns near the context window."""
    usage = {}
    openai = data.get("usage") if isinstance(data.get("usage"), dict) else None
    if openai and openai.get("prompt_tokens") is not None:
        usage["input"] = openai["prompt_tokens"]
        usage["output"] = openai.get("completion_tokens") or 0
    else:
        if data.get("prompt_eval_count") is not None:
            usage["input"] = data["prompt_eval_count"]
        if data.get("eval_count") is not None:
            usage["output"] = data["eval_count"]
    total = usage.get("input", 0) + usage.get("output", 0)
    if total >= LLM_CONTEXT_WINDOW * CONTEXT_WARN_RATIO:
        print(
            f"warning: {name} used {total}/{LLM_CONTEXT_WINDOW} tokens "
            f"(input={usage.get('input')} output={usage.get('output')}); "
            "prompt may be truncated"
        )
    return usage


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _assistant_text(message: dict) -> str:
    """Final user-visible reply from an OpenAI-style message dict."""
    return _strip_think(message.get("content") or "")


def _post_chat(messages, *, format, timeout, temperature, model, think, name):
    if LLM_BACKEND == "vllm":
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": OLLAMA_SEED,
            "max_tokens": LLM_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": bool(think)},
        }
        if format is not None:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = _assistant_text(message)
        if not content:
            # Qwen3 + reasoning-parser: thinking can burn max_tokens so
            # content is "" while reasoning/reasoning_content is filled.
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            print(
                f"warning: {name} empty content "
                f"(finish={choice.get('finish_reason')}, think={think}, "
                f"reasoning_chars={len(reasoning or '')})"
            )
        return content, _usage(data, name)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "seed": OLLAMA_SEED,
            "num_ctx": LLM_CONTEXT_WINDOW,
        },
    }
    if format is not None:
        payload["format"] = format
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return _assistant_text(data.get("message") or {}), _usage(data, name)


def ollama_chat(
    messages,
    *,
    format=None,
    timeout=OLLAMA_CHAT_TIMEOUT,
    temperature=OLLAMA_TEMPERATURE,
    model: str | None = None,
    think: bool = False,
    name: str = "llm",
):
    """Chat with retries. Backend is LLM_BACKEND (vllm or ollama)."""
    model = model or CHAT_MODEL

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
            "num_ctx": LLM_CONTEXT_WINDOW,
            "backend": LLM_BACKEND,
            "think": think,
            "max_tokens": LLM_MAX_TOKENS,
        },
    ) as gen:
        last_err = None
        for attempt in range(1, OLLAMA_CHAT_RETRIES + 1):
            try:
                content, usage = _post_chat(
                    messages,
                    format=format,
                    timeout=timeout,
                    temperature=temperature,
                    model=model,
                    think=think,
                    name=name,
                )
                # Thinking often exhausts max_tokens before the final answer.
                # One non-think retry keeps MySQL/Langfuse from storing "".
                if not content and think:
                    print(f"warning: {name} empty with think=True; retry think=False")
                    content, usage = _post_chat(
                        messages,
                        format=format,
                        timeout=timeout,
                        temperature=temperature,
                        model=model,
                        think=False,
                        name=name,
                    )
                if gen is not None:
                    gen.update(output=content, usage_details=usage or None)
                    if not content:
                        gen.update(
                            level="WARNING",
                            status_message="empty model content",
                        )
                return content
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as e:
                last_err = e
                if attempt < OLLAMA_CHAT_RETRIES:
                    wait = 5 * attempt
                    print(
                        f"{LLM_BACKEND} chat timeout/error "
                        f"(attempt {attempt}/{OLLAMA_CHAT_RETRIES}), "
                        f"{type(e).__name__}: {e}; "
                        f"retry in {wait}s..."
                    )
                    time.sleep(wait)
        if gen is not None:
            gen.update(
                level="ERROR",
                status_message=str(last_err),
            )
        raise last_err
