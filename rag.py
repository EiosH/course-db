import re
import requests
from pymilvus import DataType, MilvusClient

EMBED_URL = "http://127.0.0.1:8080/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
QUERY = "00:00:50 这里讲了什么？"

# mock 元数据
COURSE_ID = "CSC447"
QUARTER = "2026-Spring"
LECTURER = "Eric J. Fredericks"


def embed(texts):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


raw = open("doc.txt", encoding="utf-8").read()
parts = re.split(r"=+\n时间戳: (\d{2}:\d{2}:\d{2}).*?\n=+\n", raw)

chunks = []
# parts: [前缀, ts1, body1, ts2, body2, ...]
for i in range(1, len(parts), 2):
    ts, body = parts[i], parts[i + 1].strip()
    body = re.sub(r"\n=+\n视频处理统计[\s\S]*$", "", body).strip()
    if body:
        chunks.append({"timestamp": ts, "text": body})

print(f"按时间戳切成 {len(chunks)} 段")

vectors = embed([c["text"] for c in chunks])

client = MilvusClient("milvus_demo.db")
if client.has_collection("docs"):
    client.drop_collection("docs")

schema = MilvusClient.create_schema(auto_id=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=len(vectors[0]))
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("timestamp", DataType.VARCHAR, max_length=16)
schema.add_field("course_id", DataType.VARCHAR, max_length=32)
schema.add_field("quarter", DataType.VARCHAR, max_length=32)
schema.add_field("lecturer", DataType.VARCHAR, max_length=64)

index_params = client.prepare_index_params()
index_params.add_index("vector", metric_type="COSINE")
client.create_collection("docs", schema=schema, index_params=index_params)

rows = [
    {
        "id": i,
        "vector": vectors[i],
        "text": c["text"],
        "timestamp": c["timestamp"],
        "course_id": COURSE_ID,
        "quarter": QUARTER,
        "lecturer": LECTURER,
    }
    for i, c in enumerate(chunks)
]
client.insert("docs", rows)

hits = client.search(
    "docs",
    data=embed([QUERY]),
    limit=3,
    output_fields=["text", "timestamp", "course_id", "quarter", "lecturer"],
)

print(QUERY)
for h in hits[0]:
    e = h["entity"]
    print(
        f"{h['distance']:.4f}  [{e['timestamp']}] {e['course_id']} {e['quarter']} / {e['lecturer']}"
    )
    print(f"       {e['text'][:80]}\n")


# 双路召回   一路做语义向量 search；另一路直接用queryAPI，按时间窗口捞锚点附近的 chunk；结果合并去重
# 将每个截屏内容按 400 token chunk，原始文字保存 MongoDB, 向量库保存 ids
