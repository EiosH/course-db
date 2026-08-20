import argparse
import json
import re
import time
from datetime import datetime

import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:32b"
EMBED_MODEL = "bge-m3"
BM25_MODEL = "Qdrant/bm25"
OLLAMA_CHAT_TIMEOUT = 600  # qwen3:32b 生成较慢，120s 易超时
OLLAMA_CHAT_RETRIES = 3

# mock 预过滤
COURSE_ID = "CSC447"
QUARTER = "2026-Spring"
LECTURER = "Eric J. Fredericks"
TIMESTAMP = "01:22:09"  # mock 当前播放位置，仅时间类问题启用
BASE_MUST = [
    FieldCondition(key="course_id", match=MatchValue(value=COURSE_ID)),
    FieldCondition(key="quarter", match=MatchValue(value=QUARTER)),
    FieldCondition(key="lecturer", match=MatchValue(value=LECTURER)),
]

REWRITE_SYSTEM = f"""You are a search query rewrite assistant for lecture retrieval.

Current playback timestamp (mock): {TIMESTAMP}

Output JSON only:
{{
  "rewritten_query": "...",
  "hard_constraints": []
}}

Rules:
- rewritten_query: better for retrieving lecture notes, screenshot OCR, and transcripts
- hard_constraints: pre-filters for retrieval
- ONLY add {{"field":"timestamp","operator":"eq","value":"{TIMESTAMP}"}} when the user explicitly asks about what is being discussed RIGHT NOW at the current moment in the lecture (e.g. "现在在讲什么", "what are we covering now")
- Do NOT add timestamp for general knowledge questions, definitions, or past/future topics
- If no timestamp constraint is needed, hard_constraints must be []"""

# transcript：合并相邻字幕，约 400 token；静音超过 15s 切新块
TRANSCRIPT_MAX_CHARS = 1600
TRANSCRIPT_MAX_GAP_SEC = 15
CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\n(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)",
    re.S,
)


EMBED_BATCH = 32


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
    """Call Ollama /api/chat with retries; think=False avoids qwen3 long thinking."""
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
                print(f"ollama chat timeout/error (attempt {attempt}/{OLLAMA_CHAT_RETRIES}), retry in {wait}s...")
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
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_screenshots(path="doc.txt"):
    raw = open(path, encoding="utf-8").read()
    parts = re.split(r"=+\n时间戳: (\d{2}:\d{2}:\d{2}).*?\n=+\n", raw)
    chunks = []
    for i in range(1, len(parts), 2):
        ts, body = parts[i], parts[i + 1].strip()
        body = re.sub(r"\n=+\n视频处理统计[\s\S]*$", "", body).strip()
        if body:
            chunks.append({"timestamp": ts, "text": body, "type": "screen_shot"})
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


def constraints_to_must(constraints):
    must = []
    for c in constraints:
        if c.get("field") == "timestamp" and c.get("operator", "eq") == "eq":
            must.append(
                FieldCondition(key="timestamp", match=MatchValue(value=c["value"]))
            )
    return must


def build_filter(doc_type: str, constraints) -> Filter:
    return Filter(
        must=BASE_MUST
        + [FieldCondition(key="type", match=MatchValue(value=doc_type))]
        + constraints_to_must(constraints)
    )


def search(client, q, doc_type, constraints, limit=3):
    flt = build_filter(doc_type, constraints)
    return client.query_points(
        collection_name="docs",
        prefetch=[
            Prefetch(query=embed([q])[0], using="dense", limit=20, filter=flt),
            Prefetch(
                query=Document(text=q, model=BM25_MODEL),
                using="bm25",
                limit=20,
                filter=flt,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=flt,
        limit=limit,
    ).points


def build_prompt(question, grouped_hits):
    parts = [
        "Answer the student's question using the lecture materials below.",
        f"Question: {question}",
        "Lecture materials:",
    ]
    any_hit = False
    for hits in grouped_hits.values():
        for h in hits:
            p = h.payload
            any_hit = True
            parts.append(f"[{p['type']}] ({p['timestamp']})\n{p['text']}")
    if not any_hit:
        parts.append("(no materials retrieved)")
    return "\n\n".join(parts)


def answer(prompt: str) -> str:
    return ollama_chat([{"role": "user", "content": prompt}])


def format_hits(hits) -> str:
    if not hits:
        return ""
    blocks = []
    for h in hits:
        p = h.payload
        score = f"{h.score:.4f}" if h.score is not None else ""
        blocks.append(f"[{p['timestamp']}] score={score}\n{p['text']}")
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
        "screen_shot检索",
        "transcript检索",
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
        "H": 50,
        "I": 50,
        "J": 50,
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

    grouped = {
        "screen_shot": search(client, q, "screen_shot", constraints),
        "transcript": search(client, q, "transcript", constraints),
    }
    prompt = build_prompt(query, grouped)
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
            format_hits(grouped["screen_shot"]),
            format_hits(grouped["transcript"]),
            answer_text,
        ]
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"answer_{stamp}.xlsx"
write_excel(out_rows, out_path)
print(f"wrote {out_path}")
