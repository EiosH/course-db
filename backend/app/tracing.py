"""Optional Langfuse tracing — no-op unless LANGFUSE_ENABLED=true."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from config import LANGFUSE_ENABLED

_client = None
_warned = False
# Thread-local parent stack. Do NOT fall back to a new root for nested calls —
# that was splitting one question into many traces in the Langfuse UI.
_tls = threading.local()


def enabled() -> bool:
    return bool(LANGFUSE_ENABLED)


def _stack() -> list:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def current_parent():
    stack = _stack()
    return stack[-1] if stack else None


def _get_client():
    global _client, _warned
    if not enabled():
        return None
    if _client is not None:
        return _client
    try:
        from langfuse import get_client
    except ImportError:
        if not _warned:
            print("langfuse: package not installed; tracing disabled")
            _warned = True
        return None
    _client = get_client()
    return _client


def run_with_parent(parent, fn, /, *args, **kwargs):
    """
    Run fn on a worker thread with Langfuse parent stacked, so nested
    observation(..., require_parent=True) still attaches under `parent`.
    """
    stack = _stack()
    pushed = False
    if parent is not None:
        stack.append(parent)
        pushed = True
    try:
        return fn(*args, **kwargs)
    finally:
        if pushed and stack and stack[-1] is parent:
            stack.pop()


@contextmanager
def observation(
    *,
    name: str,
    as_type: str = "span",
    require_parent: bool = False,
    **kwargs: Any,
) -> Iterator[Any]:
    """
    Yield a Langfuse observation, or None when tracing is off.

    require_parent=True: only nest under the current parent. If there is no
    parent, skip (do not open a new root trace). Use this for LLM generations
    so a lost parent never creates orphan ollama_chat traces.
    """
    client = _get_client()
    if client is None:
        yield None
        return

    stack = _stack()
    parent = stack[-1] if stack else None

    if require_parent:
        if parent is None or not hasattr(parent, "start_as_current_observation"):
            yield None
            return
        cm = parent.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        )
    elif parent is not None and hasattr(parent, "start_as_current_observation"):
        cm = parent.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        )
    else:
        cm = client.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        )

    with cm as obs:
        stack.append(obs)
        try:
            yield obs
        finally:
            stack.pop()


def flush() -> None:
    client = _get_client()
    if client is not None:
        client.flush()


def hits_preview(hits, *, limit: int = 4, text_chars: int = 160) -> dict:
    """Compact hit summary for Langfuse retriever input/output."""
    preview = []
    for h in (hits or [])[:limit]:
        p = getattr(h, "payload", None) or {}
        text = (p.get("text") or "").strip()
        preview.append(
            {
                "id": getattr(h, "id", None),
                "type": p.get("type"),
                "timestamp": p.get("timestamp"),
                "score": getattr(h, "score", None),
                "rerank": p.get("rerank_score"),
                "text": text[:text_chars],
            }
        )
    return {"count": len(hits or []), "preview": preview}
