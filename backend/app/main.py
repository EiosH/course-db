import re
import requests
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
OLLAMA_MODEL = "qwen2.5:14b"
EMBED_MODEL = "bge-m3"
BM25_MODEL = "Qdrant/bm25"

# mock 预过滤
COURSE_ID = "CSC447"
QUARTER = "2026-Spring"
LECTURER = "Eric J. Fredericks"
TIMESTAMP = "01:22:09"
MOCK_MUST = [
    FieldCondition(key="course_id", match=MatchValue(value=COURSE_ID)),
    FieldCondition(key="quarter", match=MatchValue(value=QUARTER)),
    FieldCondition(key="lecturer", match=MatchValue(value=LECTURER)),
    FieldCondition(key="timestamp", match=MatchValue(value=TIMESTAMP)),
]

REWRITE_SYSTEM = """You are a search query rewrite assistant. Rewrite the user's question into a query that is better for retrieving lecture notes, screenshot OCR text, and lecture transcripts.
Resolve pronouns, expand abbreviations, and keep key terms. Output only the rewritten query, with no explanation."""

# transcript：合并相邻字幕，约 400 token；静音超过 15s 切新块
TRANSCRIPT_MAX_CHARS = 1600
TRANSCRIPT_MAX_GAP_SEC = 15
CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\n(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)",
    re.S,
)


def embed(texts):
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def rewrite(query: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


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


def type_filter(doc_type: str) -> Filter:
    return Filter(
        must=MOCK_MUST + [FieldCondition(key="type", match=MatchValue(value=doc_type))]
    )


def search(client, q, doc_type, limit=3):
    flt = type_filter(doc_type)
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


chunks = load_screenshots() + load_transcript()
print(
    f"screen_shot {sum(c['type']=='screen_shot' for c in chunks)} 段, "
    f"transcript {sum(c['type']=='transcript' for c in chunks)} 段"
)

vectors = embed([c["text"] for c in chunks])

client = QdrantClient(url="http://localhost:6333")

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
for field in ("course_id", "quarter", "lecturer", "type"):
    client.create_payload_index("docs", field, field_schema=PayloadSchemaType.KEYWORD)

client.upsert(
    "docs",
    points=[
        PointStruct(
            id=i,
            vector={
                "dense": vectors[i],
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
        for i, c in enumerate(chunks)
    ],
)

out = []
for i, query in enumerate(load_queries(), 1):
    q = rewrite(query)
    print(f"\nquery:    {query}")
    print(f"rewrite:  {q}")

    grouped = {
        "screen_shot": search(client, q, "screen_shot"),
        "transcript": search(client, q, "transcript"),
    }
    prompt = build_prompt(query, grouped)
    print(prompt)

    answers = []
    for doc_type, hits in grouped.items():
        for h in hits:
            p = h.payload
            answers.append(f"[{p['type']}] ({p['timestamp']})\n{p['text']}")

    out.append(
        f"{i}. 原问题: {query}\nrewrite: {q}\n\n"
        f"prompt:\n{prompt}\n\n"
        f"答案:\n" + ("\n\n".join(answers) if answers else "(无结果)")
    )

open("answer.txt", "w", encoding="utf-8").write("\n\n==========\n\n".join(out) + "\n")
print("wrote answer.txt")
