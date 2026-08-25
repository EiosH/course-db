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
TIME_WINDOW_SEC = 600  # 时间戳约束：±10 分钟窗口，覆盖 transcript 区间块
BASE_MUST = [
    FieldCondition(key="course_id", match=MatchValue(value=COURSE_ID)),
    FieldCondition(key="quarter", match=MatchValue(value=QUARTER)),
    FieldCondition(key="lecturer", match=MatchValue(value=LECTURER)),
]

# 每路召回候选数 → rerank 后保留
DENSE_LIMIT = 10
BM25_LIMIT = 10
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
- Do NOT add timestamp for general knowledge questions, definitions, or past/future topics
- If no timestamp constraint is needed, hard_constraints must be []"""

ANSWER_SYSTEM = """You answer student questions about a lecture using ONLY the retrieved materials.

Strict rules:
1. Use ONLY facts present in the retrieved lecture materials. Never use outside knowledge.
2. If the materials are empty, irrelevant, or insufficient, reply exactly: I don't know based on the retrieved lecture materials.
3. Answer directly and concisely. No role-play, persona, course-intro filler, or speculative asides.
4. Prefer short factual answers grounded in the materials; quote key phrases when helpful."""

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


def rerank_and_filter(query: str, hits, top_k=RERANK_TOP_K, min_score=RERANK_MIN_SCORE):
    if not hits:
        return []
    reranker = get_reranker()
    pairs = [(query, h.payload["text"]) for h in hits]
    scores = reranker.predict(pairs)
    if isinstance(scores, (int, float)):
        scores = [scores]
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    kept = []
    for h, s in ranked:
        if s < min_score:
            continue
        payload = dict(h.payload)
        payload["rerank_score"] = float(s)
        kept.append(SimpleNamespace(id=h.id, payload=payload, score=float(s)))
        if len(kept) >= top_k:
            break
    return kept


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


def answer(prompt: str) -> str:
    return ollama_chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
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

    ss_dense = search_dense(client, q, "screen_shot", constraints)
    ss_bm25 = search_bm25(client, q, "screen_shot", constraints)
    tr_dense = search_dense(client, q, "transcript", constraints)
    tr_bm25 = search_bm25(client, q, "transcript", constraints)

    print(
        f"recall screen_shot dense={len(ss_dense)} bm25={len(ss_bm25)} | "
        f"transcript dense={len(tr_dense)} bm25={len(tr_bm25)}"
    )

    candidates = merge_unique_hits(ss_dense, ss_bm25, tr_dense, tr_bm25)
    # 用原问题 + rewritten query 一起做相关性判断，减少无关噪声进入答案
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
            format_hits(ss_dense, "semantic"),
            format_hits(ss_bm25, "keyword"),
            format_hits(tr_dense, "semantic"),
            format_hits(tr_bm25, "keyword"),
            format_hits(final_hits, "rerank"),
            answer_text,
        ]
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"answer_{stamp}.xlsx"
write_excel(out_rows, out_path)
print(f"wrote {out_path}")
