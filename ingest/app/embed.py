"""HTTP client for the embedding service (dense embed)."""

import time

import requests

from config import EMBEDDING_URL


def embed(texts, *, retries: int = 5, is_query: bool = False):
    """Embed texts via embedding service."""
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
            vectors = r.json()["vectors"]
            print(f"embed {len(vectors)}/{len(texts)} via {EMBEDDING_URL}")
            return vectors
        except (requests.exceptions.RequestException, KeyError, TypeError) as e:
            last_err = e
            if attempt < retries:
                wait = 2 * attempt
                print(f"embed error (attempt {attempt}/{retries}), retry in {wait}s...")
                time.sleep(wait)
    raise last_err
