"""Local dense embedder for ingest."""

from config import EMBED_BATCH, EMBED_DEVICE, EMBED_DIM, EMBED_MODEL

_embedder = None


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
    """Embed texts locally with Qwen3-Embedding-4B."""
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
