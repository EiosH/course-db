import argparse
import json
import re
import time
from datetime import datetime
from types import SimpleNamespace

import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Range,
    SparseVectorParams,
    VectorParams,
)

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3.8:27b"
EMBED_MODEL = "bge-m3"
BM25_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_CHAT_RETRIES = 3

# mock 预过滤
COURSE_ID = "CSC447"
QUARTER = "2026-Spring"
LECTURER = "Eric J. Fredericks"
TIMESTAMP = "01:22:09"  # mock 当前播放位置，仅时间类问题启用
TIME_WINDOW_SEC = 120  # 时间戳约束：±2 分钟，优先当前画面/台词
BASE_MUST = [
    FieldCondition(key="course_id", match=MatchValue(value=COURSE_ID)),
    FieldCondition(key="quarter", match=MatchValue(value=QUARTER)),
    FieldCondition(key="lecturer", match=MatchValue(value=LECTURER)),
]

# 每路召回候选数 → rerank 后保留
DENSE_LIMIT = 10
BM25_LIMIT = 10
TIME_NEAR_TOP_K = 4  # 时间类问题：最终保留条数（screen_shot / transcript 各半）
RERANK_TOP_K = 4
RERANK_MIN_SCORE = 0.0  # Qwen3-Reranker: yes/no logit diff，>0 表示相关

REWRITE_SYSTEM = f"""You are a search query rewrite assistant for lecture retrieval.

Current playback timestamp (mock): {TIMESTAMP}

Output JSON only:
{{
  "rewritten_query": "...",
  "hard_constraints": []
}}

Rules:
- rewritten_query: concise technical search query for lecture notes, screenshot OCR, and transcripts
- Focus on domain terms (e.g. foldLeft, concurrency, operational semantics). Do NOT include course codes, instructor names, or quarter — those are already filtered
- hard_constraints: pre-filters for retrieval
- ONLY add {{"field":"timestamp","operator":"range","value":"{TIMESTAMP}"}} when the user explicitly asks about what is being discussed RIGHT NOW at the current moment (e.g. "现在在讲什么", "what are we covering now", "what does this mean now")
- When adding a timestamp constraint, rewritten_query can be a short placeholder (e.g. "current slide and transcript"); retrieval will use the timestamp, not keywords
- Do NOT add timestamp for general knowledge questions, definitions, or past/future topics
- If no timestamp constraint is needed, hard_constraints must be []"""

ANSWER_SYSTEM = """You answer student questions about a lecture using ONLY the retrieved materials.

Strict rules:
1. Use ONLY facts present in the retrieved lecture materials. Never use outside knowledge.
2. If the materials are empty, irrelevant, or insufficient, reply exactly: I don't know based on the retrieved lecture materials.
3. Answer directly and concisely. No role-play, persona, course-intro filler, or speculative asides.
4. Prefer short factual answers grounded in the materials; quote key phrases when helpful."""

NOW_ANSWER_SYSTEM = """You explain what the lecture is covering RIGHT NOW using ONLY the retrieved materials (current slide OCR and/or transcript near the playback timestamp).

Strict rules:
1. The materials ARE the current lecture content. Summarize/explain them to answer the student.
2. Do NOT refuse with "I don't know" when materials are present — even if the question is vague ("what does this mean now").
3. Use ONLY facts in the materials. Never invent outside knowledge.
4. Answer directly and concisely; quote key phrases (slide title, code, spoken lines) when helpful.
5. Only if materials are completely empty, reply exactly: I don't know based on the retrieved lecture materials."""

# transcript：合并相邻字幕，约 400 token；静音超过 15s 切新块
TRANSCRIPT_MAX_CHARS = 1600
TRANSCRIPT_MAX_GAP_SEC = 15
CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\n(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)",
    re.S,
)
SCREEN_LABEL_RE = re.compile(
    r"(?im)^(?:Type|Top|Left|Right|Bottom|Content|Title|Bullet Points|Notes|Question)\s*:\s*"
)
SCREEN_BOILERPLATE_RE = re.compile(
    r"(?im)(?:reed\.cs\.depaul\.edu[^\n]*|to exit full screen[^\n]*|press Esc[^\n]*)\n?"
)

EMBED_BATCH = 32
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"loading rerank model {RERANK_MODEL}...")
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker


def embed(texts):
    out = []
    total = len(texts)
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=300,
        )
        if not r.ok:
            raise RuntimeError(f"Ollama embed {r.status_code}: {r.text}")
        out.extend(r.json()["embeddings"])
        done = min(i + EMBED_BATCH, total)
        print(f"embed {done}/{total}")
    return out


def ollama_chat(messages, *, format=None, timeout=OLLAMA_CHAT_TIMEOUT):
    """Call Ollama /api/chat with retries; think=False avoids qwen thinking overhead."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if format is not None:
        payload["format"] = format

    last_err = None
    for attempt in range(1, OLLAMA_CHAT_RETRIES + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < OLLAMA_CHAT_RETRIES:
                wait = 5 * attempt
                print(
                    f"ollama chat timeout/error (attempt {attempt}/{OLLAMA_CHAT_RETRIES}), "
                    f"retry in {wait}s..."
                )
                time.sleep(wait)
    raise last_err


def rewrite(query: str) -> dict:
    raw = ollama_chat(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": query},
        ],
        format="json",
    )
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    return {
        "rewritten_query": data["rewritten_query"],
        "hard_constraints": data.get("hard_constraints", []),
    }


def load_queries(path="query.txt"):
    queries = []
    for line in open(path, encoding="utf-8"):
        line = re.sub(r"^\d+\.\s*", "", line.strip())
        if line:
            queries.append(line)
    return queries


def ts_to_sec(ts: str) -> float:
    h, m, s = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def clean_screenshot_text(body: str) -> str:
    """Strip OCR layout labels / UI chrome that pollute BM25 keyword recall."""
    text = SCREEN_LABEL_RE.sub("", body)
    text = SCREEN_BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or body.strip()


def load_screenshots(path="doc.txt"):
    raw = open(path, encoding="utf-8").read()
    parts = re.split(r"=+\n时间戳: (\d{2}:\d{2}:\d{2}).*?\n=+\n", raw)
    chunks = []
    for i in range(1, len(parts), 2):
        ts, body = parts[i], parts[i + 1].strip()
        body = re.sub(r"\n=+\n视频处理统计[\s\S]*$", "", body).strip()
        body = clean_screenshot_text(body)
        if body:
            sec = ts_to_sec(ts)
            chunks.append(
                {
                    "timestamp": ts,
                    "start_sec": sec,
                    "end_sec": sec,
                    "text": body,
                    "type": "screen_shot",
                }
            )
    return chunks


def load_transcript(path="transcript.vtt"):
    try:
        raw = open(path, encoding="utf-8").read().strip()
    except FileNotFoundError:
        return []
    if not raw:
        return []
    raw = re.sub(r"^WEBVTT\s*", "", raw, flags=re.I)
    cues = []
    for m in CUE_RE.finditer(raw):
        text = re.sub(r"\s+", " ", m.group(3)).strip()
        if text:
            cues.append({"start": m.group(1), "end": m.group(2), "text": text})

    chunks, buf, start, end, n = [], [], None, None, 0

    def flush():
        nonlocal buf, start, end, n
        if buf:
            chunks.append(
                {
                    "timestamp": f"{start} --> {end}",
                    "start_sec": ts_to_sec(start),
                    "end_sec": ts_to_sec(end),
                    "text": " ".join(buf),
                    "type": "transcript",
                }
            )
        buf, start, end, n = [], None, None, 0

    for c in cues:
        gap = ts_to_sec(c["start"]) - ts_to_sec(end) if end else 0
        if buf and (
            gap > TRANSCRIPT_MAX_GAP_SEC
            or n + len(c["text"]) + 1 > TRANSCRIPT_MAX_CHARS
        ):
            flush()
        if start is None:
            start = c["start"]
        buf.append(c["text"])
        end = c["end"]
        n += len(c["text"]) + 1
    flush()
    return chunks


def timestamp_constraint_value(constraints):
    for c in constraints:
        if c.get("field") == "timestamp":
            return c.get("value", TIMESTAMP)
    return None


def time_distance(center: float, payload: dict) -> float:
    start = payload.get("start_sec")
    end = payload.get("end_sec")
    if start is not None and end is not None:
        if start <= center <= end:
            return 0.0
        return min(abs(start - center), abs(end - center))
    ts = payload.get("timestamp", "")
    if ts and "-->" not in ts:
        try:
            return abs(ts_to_sec(ts) - center)
        except (ValueError, AttributeError):
            pass
    return float("inf")


def _scroll_time_filter(center: float, ts_value: str, doc_type: str | None, *, use_range: bool):
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
        must.append(
            FieldCondition(key="timestamp", match=MatchValue(value=ts_value))
        )
    return Filter(must=must)


def _scroll_all(client, scroll_filter: Filter, page_size: int = 256):
    """Paginate Qdrant scroll so large time windows are not truncated."""
    out = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name="docs",
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
        )
        out.extend(points)
        if offset is None or not points:
            break
    return out


def search_time_window(client, constraints, doc_type: str | None = None, limit=TIME_NEAR_TOP_K):
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
        collection_name="docs",
        query=embed([q])[0],
        using="dense",
        query_filter=flt,
        limit=limit,
    ).points


def search_bm25(client, q, doc_type, constraints, limit=BM25_LIMIT):
    flt = build_filter(doc_type, constraints)
    return client.query_points(
        collection_name="docs",
        query=Document(text=q, model=BM25_MODEL),
        using="bm25",
        query_filter=flt,
        limit=limit,
    ).points


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
    other = [h for h in hits if h.payload.get("type") not in ("screen_shot", "transcript")]
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


def build_prompt(question, hits):
    parts = [
        f"Question: {question}",
        "Retrieved lecture materials:",
    ]
    if not hits:
        parts.append("(no materials retrieved)")
    else:
        for h in hits:
            p = h.payload
            parts.append(f"[{p['type']}] ({p['timestamp']})\n{p['text']}")
    return "\n\n".join(parts)


def answer(prompt: str, *, system: str = ANSWER_SYSTEM) -> str:
    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )


def format_hits(hits, channel: str = "") -> str:
    if not hits:
        return ""
    blocks = []
    for h in hits:
        p = h.payload
        score = f"{h.score:.4f}" if h.score is not None else ""
        tag = f" channel={channel}" if channel else ""
        rerank = p.get("rerank_score")
        extra = f" rerank={rerank:.4f}" if rerank is not None else ""
        td = p.get("time_distance")
        if td is not None:
            extra += f" dt={td:.1f}s"
        blocks.append(f"[{p['timestamp']}] score={score}{tag}{extra}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)



def write_excel(rows, path: str):
    wb = Workbook()
    ws = wb.active
    ws.title = "answers"
    headers = [
        "序号",
        "原问题",
        "rewritten_query",
        "hard_constraints",
        "course_id",
        "quarter",
        "lecturer",
        "screen_shot语义召回",
        "screen_shot关键词召回",
        "transcript语义召回",
        "transcript关键词召回",
        "rerank后检索",
        "答案",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    wrap = Alignment(vertical="top", wrap_text=True)
    for row in rows:
        ws.append(row)
        for cell in ws[ws.max_row]:
            cell.alignment = wrap

    widths = {
        "A": 6,
        "B": 36,
        "C": 40,
        "D": 28,
        "E": 12,
        "F": 14,
        "G": 20,
        "H": 42,
        "I": 42,
        "J": 42,
        "K": 42,
        "L": 50,
        "M": 50,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    wb.save(path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ingest",
    action="store_true",
    help="chunk + embed + upsert; default skips ingest and searches existing data",
)
args = parser.parse_args()

client = QdrantClient(url="http://localhost:6333", check_compatibility=False)

if args.ingest:
    chunks = load_screenshots() + load_transcript()
    print(
        f"screen_shot {sum(c['type']=='screen_shot' for c in chunks)} 段, "
        f"transcript {sum(c['type']=='transcript' for c in chunks)} 段"
    )

    vectors = embed([c["text"] for c in chunks])

    if client.collection_exists("docs"):
        client.delete_collection("docs")

    client.create_collection(
        "docs",
        vectors_config={
            "dense": VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    for field in ("course_id", "quarter", "lecturer", "type", "timestamp"):
        client.create_payload_index(
            "docs", field, field_schema=PayloadSchemaType.KEYWORD
        )
    for field in ("start_sec", "end_sec"):
        client.create_payload_index(
            "docs", field, field_schema=PayloadSchemaType.FLOAT
        )

    UPSERT_BATCH = 64
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH):
        batch = chunks[start : start + UPSERT_BATCH]
        client.upsert(
            "docs",
            points=[
                PointStruct(
                    id=start + j,
                    vector={
                        "dense": vectors[start + j],
                        "bm25": Document(text=c["text"], model=BM25_MODEL),
                    },
                    payload={
                        "text": c["text"],
                        "timestamp": c["timestamp"],
                        "start_sec": c["start_sec"],
                        "end_sec": c["end_sec"],
                        "type": c["type"],
                        "course_id": COURSE_ID,
                        "quarter": QUARTER,
                        "lecturer": LECTURER,
                    },
                )
                for j, c in enumerate(batch)
            ],
        )
        print(f"upsert {min(start + UPSERT_BATCH, total)}/{total}")
else:
    print("skip ingest, search existing collection")

out_rows = []
for i, query in enumerate(load_queries(), 1):
    rewritten = rewrite(query)
    q = rewritten["rewritten_query"]
    constraints = rewritten["hard_constraints"]
    print(f"\nquery:    {query}")
    print("rewrite:")
    print(json.dumps(rewritten, ensure_ascii=False, indent=2))

    ts_value = timestamp_constraint_value(constraints)

    if ts_value:
        # “现在在讲什么”：只按时间邻近取内容，不做语义检索 / rerank
        # 每类各取 top_k，再按类型配额合并，避免截图挤掉 transcript
        tw_ss = search_time_window(
            client, constraints, "screen_shot", limit=TIME_NEAR_TOP_K
        )
        tw_tr = search_time_window(
            client, constraints, "transcript", limit=TIME_NEAR_TOP_K
        )
        center = ts_to_sec(ts_value)
        # 各类内部已按时间距离排序；配额合并保证两边都进最终结果
        final_hits = pick_by_type_quota(
            merge_unique_hits(tw_ss, tw_tr),
            TIME_NEAR_TOP_K,
        )
        # 最终再按时间距离排一下，方便阅读
        final_hits = sorted(
            final_hits,
            key=lambda h: time_distance(center, h.payload),
        )
        ss_dense = tw_ss
        ss_bm25 = []
        tr_dense = tw_tr
        tr_bm25 = []
        print(
            f"time-only recall ss={len(tw_ss)} tr={len(tw_tr)} → kept {len(final_hits)} "
            f"(ss={sum(1 for h in final_hits if h.payload.get('type')=='screen_shot')} "
            f"tr={sum(1 for h in final_hits if h.payload.get('type')=='transcript')})"
        )
        if not final_hits:
            print(
                "warning: timestamp filter matched 0 chunks — "
                "re-run with --ingest to rebuild start_sec/end_sec indexes"
            )
        prompt = build_prompt(query, final_hits)
        answer_text = answer(prompt, system=NOW_ANSWER_SYSTEM)
    else:
        ss_dense = search_dense(client, q, "screen_shot", constraints)
        ss_bm25 = search_bm25(client, q, "screen_shot", constraints)
        tr_dense = search_dense(client, q, "transcript", constraints)
        tr_bm25 = search_bm25(client, q, "transcript", constraints)
        print(
            f"recall screen_shot dense={len(ss_dense)} bm25={len(ss_bm25)} | "
            f"transcript dense={len(tr_dense)} bm25={len(tr_bm25)}"
        )
        candidates = merge_unique_hits(ss_dense, ss_bm25, tr_dense, tr_bm25)
        rerank_query = f"{query}\n{q}"
        final_hits = rerank_and_filter(rerank_query, candidates)
        print(f"rerank kept {len(final_hits)}/{len(candidates)}")
        prompt = build_prompt(query, final_hits)
        answer_text = answer(prompt)

    print(prompt)
    print(f"\nanswer:\n{answer_text}")

    out_rows.append(
        [
            i,
            query,
            q,
            json.dumps(constraints, ensure_ascii=False),
            COURSE_ID,
            QUARTER,
            LECTURER,
            format_hits(ss_dense, "semantic" if not ts_value else "time"),
            format_hits(ss_bm25, "keyword"),
            format_hits(tr_dense, "semantic" if not ts_value else "time"),
            format_hits(tr_bm25, "keyword"),
            format_hits(final_hits, "time" if ts_value else "rerank"),
            answer_text,
        ]
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"answer_{stamp}.xlsx"
write_excel(out_rows, out_path)
print(f"wrote {out_path}")
