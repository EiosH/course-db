import requests
from pymilvus import MilvusClient

EMBED_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBED_MODEL = "nomic-embed-text"

DOCS = [
    "Milvus 是一个开源的向量数据库，专为大规模相似度搜索设计。",
    "RAG 先检索相关文档，再交给大模型生成答案。",
    "Embedding 模型把文本映射成向量，语义相近的文本距离更近。",
    "HNSW 和 IVF_FLAT 是常见的向量索引类型。",
]

QUERY = "什么是向量数据库？"


def embed(texts):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


vectors = embed(DOCS)

client = MilvusClient("milvus_demo.db")
client.drop_collection("docs") if client.has_collection("docs") else None
client.create_collection("docs", dimension=len(vectors[0]))
client.insert("docs", [{"id": i, "vector": v, "text": t} for i, (v, t) in enumerate(zip(vectors, DOCS))])

hits = client.search("docs", data=embed([QUERY]), limit=3, output_fields=["text"])

print(QUERY)
for h in hits[0]:
    print(round(h["distance"], 4), h["entity"]["text"])
