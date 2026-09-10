"""Qdrant recall, merge, rerank, and phrase-anchor expansion."""

from types import SimpleNamespace

from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from anchors import (
    bm25_phrase_query,
    extract_anchor_phrases,
    phrase_match_score,
    phrase_token,
    text_has_phrase,
)
from config import (
    ANCHOR_EXPAND_CHARS,
    ANCHOR_LIMIT,
    BASE_MUST,
    BM25_LIMIT,
    BM25_MODEL,
    COLLECTION_NAME,
    DENSE_LIMIT,
    PREPROBE_BM25_LIMIT,
    PREPROBE_DENSE_LIMIT,
    PREPROBE_TOP_K,
    RERANK_MIN_SCORE,
    RERANK_TOP_K,
    TIME_NEAR_TOP_K,
    TIME_WINDOW_SEC,
)
from llm import embed, get_reranker
from time_utils import time_distance, timestamp_constraint_value, ts_to_sec


def _scroll_time_filter(
    center: float, ts_value: str, doc_type: str | None, *, use_range: bool
):
    must = list(BASE_MUST)
    if doc_type:
        must.append(FieldCondition(key="type", match=MatchValue(value=doc_type)))
    if use_range:
        must.extend(
            [
                FieldCondition(
                    key="start_sec",
                    range=Range(lte=center + TIME_WINDOW_SEC),
                ),
                FieldCondition(
                    key="end_sec",
                    range=Range(gte=center - TIME_WINDOW_SEC),
                ),
            ]
        )
    else:
        must.append(FieldCondition(key="timestamp", match=MatchValue(value=ts_value)))
    return Filter(must=must)


def _scroll_all(client, scroll_filter: Filter, page_size: int = 256):
    """Paginate Qdrant scroll so large time windows are not truncated."""
    out = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
        )
        out.extend(points)
        if offset is None or not points:
            break
    return out


def search_time_window(
    client, constraints, doc_type: str | None = None, limit=TIME_NEAR_TOP_K
):
    """Filter-only recall: chunks nearest to the playback timestamp (no semantic search)."""
    ts_value = timestamp_constraint_value(constraints)
    if ts_value is None:
        return []

    center = ts_to_sec(ts_value)
    points = _scroll_all(
        client,
        _scroll_time_filter(center, ts_value, doc_type, use_range=True),
    )
    if not points:
        # 兼容旧索引：仅有 timestamp 精确字段、无 start_sec/end_sec
        points = _scroll_all(
            client,
            _scroll_time_filter(center, ts_value, doc_type, use_range=False),
        )

    ranked = sorted(points, key=lambda p: time_distance(center, p.payload))
    kept = []
    for p in ranked[:limit]:
        payload = dict(p.payload)
        payload["time_distance"] = time_distance(center, payload)
        kept.append(SimpleNamespace(id=p.id, payload=payload, score=None))
    return kept


def build_filter(doc_type: str, constraints) -> Filter:
    must = BASE_MUST + [FieldCondition(key="type", match=MatchValue(value=doc_type))]
    ts_value = timestamp_constraint_value(constraints)
    if ts_value is not None:
        # 时间范围重叠：chunk.start <= center+W AND chunk.end >= center-W
        # transcript 存的是区间，不能再用 timestamp KEYWORD eq
        center = ts_to_sec(ts_value)
        must.append(
            FieldCondition(
                key="start_sec",
                range=Range(lte=center + TIME_WINDOW_SEC),
            )
        )
        must.append(
            FieldCondition(
                key="end_sec",
                range=Range(gte=center - TIME_WINDOW_SEC),
            )
        )
    return Filter(must=must)


def search_dense(client, q, doc_type, constraints, limit=DENSE_LIMIT):
    flt = build_filter(doc_type, constraints)
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embed([q])[0],
        using="dense",
        query_filter=flt,
        limit=limit,
    ).points


def search_bm25(client, q, doc_type, constraints, limit=BM25_LIMIT):
    flt = build_filter(doc_type, constraints)
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=Document(text=q, model=BM25_MODEL),
        using="bm25",
        query_filter=flt,
        limit=limit,
    ).points


def search_phrase_bm25(
    client, phrase: str, doc_type: str, constraints, limit=ANCHOR_LIMIT
):
    """BM25 with the glued phrase token (needs ingest/backfill that wrote those tokens)."""
    token = phrase_token(phrase)
    hits = search_bm25(client, token, doc_type, constraints, limit=limit)
    kept = [h for h in hits if text_has_phrase(h.payload.get("text", ""), phrase)]
    return kept or hits


def merge_unique_hits(*hit_lists):
    seen = set()
    merged = []
    for hits in hit_lists:
        for h in hits:
            if h.id in seen:
                continue
            seen.add(h.id)
            merged.append(h)
    return merged


def pick_by_type_quota(hits, top_k=RERANK_TOP_K):
    """Keep both screen_shot and transcript when available (half/half, fill remainder)."""
    if not hits or top_k <= 0:
        return []
    ss = [h for h in hits if h.payload.get("type") == "screen_shot"]
    tr = [h for h in hits if h.payload.get("type") == "transcript"]
    other = [
        h for h in hits if h.payload.get("type") not in ("screen_shot", "transcript")
    ]
    ss_n = min(len(ss), (top_k + 1) // 2)
    tr_n = min(len(tr), top_k - ss_n)
    # 一侧不足时把名额补给另一侧
    if ss_n + tr_n < top_k:
        ss_n = min(len(ss), top_k - tr_n)
    if ss_n + tr_n < top_k:
        tr_n = min(len(tr), top_k - ss_n)
    picked = merge_unique_hits(ss[:ss_n], tr[:tr_n], other)
    return picked[:top_k]


def rerank_and_filter(query: str, hits, top_k=RERANK_TOP_K, min_score=RERANK_MIN_SCORE):
    if not hits:
        return []
    reranker = get_reranker()
    pairs = [(query, h.payload["text"]) for h in hits]
    scores = reranker.predict(pairs)
    if isinstance(scores, (int, float)):
        scores = [scores]
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    scored = []
    for h, s in ranked:
        if s < min_score:
            continue
        payload = dict(h.payload)
        payload["rerank_score"] = float(s)
        scored.append(SimpleNamespace(id=h.id, payload=payload, score=float(s)))
    # 按分数已排序；配额保证 transcript 不被 screen_shot 全部挤掉
    return pick_by_type_quota(scored, top_k)


def search_preprobe(
    client,
    entity: str,
    constraints=None,
    top_k=PREPROBE_TOP_K,
    *,
    referent_unclear: bool = False,
):
    """
    Lightweight entity / referent recall: one query, small dense+BM25 top-k, no rerank.
    Named entity → "What is {entity}".
    Unclear deixis → resolve what the vague phrase refers to (optionally time-filtered
    via constraints from the query plan).
    """
    constraints = constraints or []
    if referent_unclear:
        q = f"What is being referred to by '{entity}'"
    else:
        q = f"What is {entity}"
    hits = merge_unique_hits(
        search_dense(
            client, q, "screen_shot", constraints, limit=PREPROBE_DENSE_LIMIT
        ),
        search_dense(
            client, q, "transcript", constraints, limit=PREPROBE_DENSE_LIMIT
        ),
        search_bm25(
            client, q, "screen_shot", constraints, limit=PREPROBE_BM25_LIMIT
        ),
        search_bm25(
            client, q, "transcript", constraints, limit=PREPROBE_BM25_LIMIT
        ),
    )
    # Prefer higher native score when present; keep order stable otherwise
    ranked = sorted(
        hits,
        key=lambda h: (
            getattr(h, "score", None)
            if getattr(h, "score", None) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return ranked[: max(1, min(top_k, PREPROBE_TOP_K))], q


def expand_query_with_anchors(client, query: str, rewritten_q: str, constraints):
    """
    Query-side phrase expansion (generic):
    - detect any Label+number locator in the user question
    - BM25-search its glued token (same token written at ingest)
    - append matching chunk text into the search query
    """
    phrases = extract_anchor_phrases(query, rewritten_q)
    if not phrases:
        return rewritten_q, []

    phrase_q = bm25_phrase_query(query, rewritten_q)
    print(f"phrase expand tokens: {phrase_q!r} from {phrases}")

    anchor_hits, snippets = [], []
    for phrase in phrases:
        raw_hits = merge_unique_hits(
            search_phrase_bm25(
                client, phrase, "screen_shot", constraints, limit=ANCHOR_LIMIT
            ),
            search_phrase_bm25(
                client,
                phrase,
                "transcript",
                constraints,
                limit=max(3, ANCHOR_LIMIT // 2),
            ),
        )
        ranked = sorted(
            raw_hits,
            key=lambda h: (
                -phrase_match_score(h.payload.get("text", ""), phrase),
                -(getattr(h, "score", None) or 0),
            ),
        )
        for h in ranked[:4]:
            if any(a.id == h.id for a in anchor_hits):
                continue
            anchor_hits.append(h)
            text = h.payload.get("text", "").strip()
            if text:
                snippets.append(text[:ANCHOR_EXPAND_CHARS])

    base = rewritten_q if not phrase_q else f"{rewritten_q}\n{phrase_q}"

    if not snippets:
        return base, []

    unique_snips = []
    for s in snippets:
        head = s[:160]
        if any(head in u or u[:160] in s for u in unique_snips):
            continue
        unique_snips.append(s)
        if len(unique_snips) >= 2:
            break

    expanded = (
        f"{base}\n"
        f"Related lecture excerpt for {' / '.join(phrases)}:\n"
        + "\n---\n".join(unique_snips)
    )
    print(
        f"anchor expand phrases={phrases} hits={len(anchor_hits)} "
        f"snippet_chars={sum(len(s) for s in unique_snips)}"
    )
    return expanded, anchor_hits
