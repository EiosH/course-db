"""Optional Langfuse tracing — no-op unless LANGFUSE_ENABLED=true."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from config import LANGFUSE_ENABLED

_client = None
_warned = False


def enabled() -> bool:
    return bool(LANGFUSE_ENABLED)


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
    """Yield a Langfuse observation, or None when tracing is off."""
    client = _get_client()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(
        name=name,
        as_type=as_type,
        **kwargs,
    ) as obs:
        yield obs


def langchain_callbacks() -> list:
    """LCEL callbacks for pipeline stages; empty when tracing is off."""
    if not enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return []
    return [CallbackHandler()]


def flush() -> None:
    client = _get_client()
    if client is not None:
        client.flush()
