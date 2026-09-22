"""Optional Langfuse tracing — no-op unless LANGFUSE_ENABLED=true."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from config import LANGFUSE_ENABLED

_client = None
_warned = False
# Thread-local parent stack (not ContextVar): LangChain LCEL may replace
# contextvars during invoke, which orphaned nested ollama_chat into new traces.
_tls = threading.local()


def enabled() -> bool:
    return bool(LANGFUSE_ENABLED)


def _stack() -> list:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


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


@contextmanager
def observation(
    *,
    name: str,
    as_type: str = "span",
    **kwargs: Any,
) -> Iterator[Any]:
    """Yield a Langfuse observation nested under the current parent, or None."""
    client = _get_client()
    if client is None:
        yield None
        return

    stack = _stack()
    parent = stack[-1] if stack else None
    if parent is not None and hasattr(parent, "start_as_current_observation"):
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
