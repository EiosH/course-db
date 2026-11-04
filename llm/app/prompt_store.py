"""Fetch Langfuse text prompts by name; fall back to local strings."""

from __future__ import annotations

from config.models import LANGFUSE_ENABLED

_PROMPT_LABEL = "production"
_warned: set[str] = set()


def _client():
    if not LANGFUSE_ENABLED:
        return None
    try:
        from langfuse import get_client
    except ImportError:
        if "import" not in _warned:
            print("prompt_store: langfuse not installed; using local prompts")
            _warned.add("import")
        return None
    return get_client()


def get(name: str, *, fallback: str, **vars: str) -> str:
    """
    Return prompt text for `name`.

    When LANGFUSE_ENABLED, fetch label=production and compile Mustache vars.
    On any failure (or when disabled), return `fallback` (already-rendered local text).
    """
    client = _client()
    if client is None:
        return fallback
    try:
        prompt = client.get_prompt(
            name,
            label=_PROMPT_LABEL,
            type="text",
            fallback=fallback,
        )
        if vars:
            return prompt.compile(**vars)
        # No variables: compile() still works for plain text prompts.
        return prompt.compile()
    except Exception as e:
        if name not in _warned:
            print(f"prompt_store: {name!r} → local fallback ({e})")
            _warned.add(name)
        return fallback
