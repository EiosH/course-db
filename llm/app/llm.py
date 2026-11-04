"""Chat (Ollama or vLLM) + HTTP client for the embedding service (embed / rerank)."""

import re
import time

import requests

from config import (
    CHAT_MODEL,
    CONTEXT_WARN_RATIO,
    EMBEDDING_URL,
    LLM_BACKEND,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    OLLAMA_CHAT_RETRIES,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_SEED,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    VLLM_BASE_URL,
)
from tracing import observation


def warmup_retrieval_models() -> None:
    """Wait until the embedding service is healthy (models load there on startup)."""
    url = f"{EMBEDDING_URL}/health"
    last_err = None
    for attempt in range(1, 31):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            print(f"embedding service ready at {EMBEDDING_URL}")
            return
        except (requests.exceptions.RequestException, ValueError) as e:
            last_err = e
            print(f"waiting for embedding service ({attempt}/30): {e}")
            time.sleep(2)
    raise RuntimeError(f"embedding service not ready at {url}: {last_err}")


def embed(texts, *, retries: int = 5, is_query: bool = False):
    """Embed via embedding service. Queries should pass is_query=True."""
    if not texts:
        return []
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{EMBEDDING_URL}/embed",
                json={"texts": texts, "is_query": is_query},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()["vectors"]
        except (requests.exceptions.RequestException, KeyError, TypeError) as e:
            last_err = e
            if attempt < retries:
                wait = 2 * attempt
                print(f"embed error (attempt {attempt}/{retries}), retry in {wait}s...")
                time.sleep(wait)
    raise last_err


def rerank(query: str, documents: list[str], *, retries: int = 5) -> list[float]:
    """Score (query, document) pairs via embedding service."""
    if not documents:
        return []
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{EMBEDDING_URL}/rerank",
                json={"query": query, "documents": documents},
                timeout=120,
            )
            r.raise_for_status()
            return [float(s) for s in r.json()["scores"]]
        except (requests.exceptions.RequestException, KeyError, TypeError) as e:
            last_err = e
            if attempt < retries:
                wait = 2 * attempt
                print(f"rerank error (attempt {attempt}/{retries}), retry in {wait}s...")
                time.sleep(wait)
    raise last_err


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


def _reasoning_text(message: dict) -> str:
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return _strip_think(str(reasoning or ""))


def _assistant_text(message: dict, *, think: bool) -> str:
    """
    Final user-visible reply from an OpenAI-style message dict.

    Prefer `content`. Only fall back to `reasoning` / `reasoning_content` when
    thinking was off — vLLM Qwen3 `--reasoning-parser` often misroutes the
    whole answer there with `content` null. While thinking is on, empty
    content means the caller should retry without think (do not treat the
    reasoning dump as the answer).
    """
    content = _strip_think(message.get("content") or "")
    if content:
        return content
    if think:
        return ""
    return _reasoning_text(message)


def _post_chat(
    messages,
    *,
    format,
    timeout,
    temperature,
    model,
    think,
    name,
    max_tokens: int,
):
    if LLM_BACKEND == "vllm":
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": OLLAMA_SEED,
            "max_tokens": max_tokens,
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
        finish = choice.get("finish_reason")
        content = _assistant_text(message, think=think)
        raw_content = message.get("content")
        reasoning = _reasoning_text(message)
        if not content:
            print(
                f"warning: {name} empty assistant text "
                f"(finish={finish}, think={think}, max_tokens={max_tokens}, "
                f"content_chars={len(raw_content or '')}, "
                f"reasoning_chars={len(reasoning)})"
            )
        elif not (raw_content or "").strip() and reasoning:
            print(
                f"warning: {name} used reasoning fallback "
                f"(content empty, reasoning_chars={len(reasoning)})"
            )
        elif finish == "length":
            print(
                f"warning: {name} truncated at max_tokens={max_tokens} "
                f"(think={think}, chars={len(content)})"
            )
        return content, _usage(data, name), finish

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "seed": OLLAMA_SEED,
            "num_ctx": LLM_CONTEXT_WINDOW,
            "num_predict": max_tokens,
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
    # Ollama: done_reason may be "length" when capped.
    finish = data.get("done_reason") or ("stop" if data.get("done") else None)
    content = _assistant_text(data.get("message") or {}, think=think)
    if finish == "length":
        print(
            f"warning: {name} truncated at num_predict={max_tokens} "
            f"(think={think}, chars={len(content)})"
        )
    return content, _usage(data, name), finish


def ollama_chat(
    messages,
    *,
    format=None,
    timeout=OLLAMA_CHAT_TIMEOUT,
    temperature=OLLAMA_TEMPERATURE,
    model: str | None = None,
    think: bool = False,
    name: str = "llm",
    max_tokens: int | None = None,
):
    """Chat with retries. Backend is LLM_BACKEND (vllm or ollama)."""
    model = model or CHAT_MODEL
    # Structured JSON stays on the smaller cap; free-form answers pass ANSWER_MAX_TOKENS.
    max_tokens = LLM_MAX_TOKENS if max_tokens is None else max_tokens

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
            "max_tokens": max_tokens,
        },
    ) as gen:
        last_err = None
        for attempt in range(1, OLLAMA_CHAT_RETRIES + 1):
            try:
                content, usage, finish = _post_chat(
                    messages,
                    format=format,
                    timeout=timeout,
                    temperature=temperature,
                    model=model,
                    think=think,
                    name=name,
                    max_tokens=max_tokens,
                )
                # Thinking burns the budget (empty or truncated) → one plain retry.
                if think and (not content or finish == "length"):
                    why = "empty" if not content else "truncated"
                    print(
                        f"warning: {name} {why} with think=True; "
                        f"retry think=False max_tokens={max_tokens}"
                    )
                    content, usage, finish = _post_chat(
                        messages,
                        format=format,
                        timeout=timeout,
                        temperature=temperature,
                        model=model,
                        think=False,
                        name=name,
                        max_tokens=max_tokens,
                    )
                if gen is not None:
                    gen.update(output=content, usage_details=usage or None)
                    if not content:
                        gen.update(
                            level="WARNING",
                            status_message="empty model content",
                        )
                    elif finish == "length":
                        gen.update(
                            level="WARNING",
                            status_message=f"truncated at max_tokens={max_tokens}",
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
