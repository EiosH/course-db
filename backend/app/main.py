"""CLI entry: ingest / backfill / batch eval."""

import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    PointVectors,
    SparseVectorParams,
    VectorParams,
)

from anchors import bm25_index_text
from config import (
    BM25_MODEL,
    COLLECTION_NAME,
    QDRANT_URL,
    UPSERT_BATCH,
)
from eval import default_reporters, run_batch
from llm import embed
from loaders import load_all_lecture_chunks, load_dialogs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="chunk + embed + upsert; default skips ingest and searches existing data",
    )
    parser.add_argument(
        "--backfill-bm25-phrases",
        action="store_true",
        help="rebuild BM25 sparse vectors with phrase tokens from existing payloads (no dense re-embed)",
    )
    parser.add_argument(
        "--dialogs",
        metavar="PATH",
        help="eval input: dialogs .yaml (default data/dialogs.yaml) "
        "or legacy .txt with one question per line",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="eval: also write data/out/answer.xlsx (default: console + Langfuse only)",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="eval: also write data/out/answer_<stamp>.txt",
    )
    return parser.parse_args()


def backfill_bm25_phrases(client):
    # 只重写 bm25 稀疏向量，不重算 dense
    offset = None
    updated = with_tok = 0
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=64,
            offset=offset,
            with_payload=True,
        )
        if not points:
            break
        batch_vecs = []
        for p in points:
            text = p.payload.get("text", "")
            idx = bm25_index_text(text)
            if idx != text:
                with_tok += 1
            batch_vecs.append(
                PointVectors(
                    id=p.id,
                    vector={"bm25": Document(text=idx, model=BM25_MODEL)},
                )
            )
            updated += 1
        client.update_vectors(collection_name=COLLECTION_NAME, points=batch_vecs)
        print(f"backfill bm25 phrases {updated} (with tokens {with_tok})")
        if offset is None:
            break
    print(f"done: updated={updated} with_phrase_tokens={with_tok}")


def ingest_docs(client):
    chunks = load_all_lecture_chunks()
    if not chunks:
        raise SystemExit("no lecture chunks to ingest (check data/lectures/*/)")
    print(
        f"total screen_shot {sum(c['type']=='screen_shot' for c in chunks)} 段, "
        f"transcript {sum(c['type']=='transcript' for c in chunks)} 段"
    )

    vectors = embed([c["text"] for c in chunks])

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    for field in ("course_id", "quarter", "lecturer", "type", "timestamp", "lecture_id"):
        client.create_payload_index(
            COLLECTION_NAME, field, field_schema=PayloadSchemaType.KEYWORD
        )
    for field in ("start_sec", "end_sec"):
        client.create_payload_index(
            COLLECTION_NAME, field, field_schema=PayloadSchemaType.FLOAT
        )

    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH):
        batch = chunks[start : start + UPSERT_BATCH]
        client.upsert(
            COLLECTION_NAME,
            points=[
                PointStruct(
                    id=start + j,
                    vector={
                        "dense": vectors[start + j],
                        # BM25 用带短语整体 token 的文本；payload.text 仍是干净原文
                        "bm25": Document(
                            text=bm25_index_text(c["text"]), model=BM25_MODEL
                        ),
                    },
                    payload={
                        "text": c["text"],
                        "timestamp": c["timestamp"],
                        "start_sec": c["start_sec"],
                        "end_sec": c["end_sec"],
                        "type": c["type"],
                        "lecture_id": c.get("lecture_id") or "",
                        "source_file": c.get("source_file") or "",
                        # per-lecture meta from data/lectures/<id>/meta.json
                        "course_id": c["course_id"],
                        "quarter": c["quarter"],
                        "lecturer": c["lecturer"],
                    },
                )
                for j, c in enumerate(batch)
            ],
        )
        print(f"upsert {min(start + UPSERT_BATCH, total)}/{total}")


def build_reporters(args):
    """Assemble pluggable eval sinks from CLI flags."""
    return default_reporters(excel=args.excel, txt=args.txt)


def main():
    args = parse_args()
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    if args.backfill_bm25_phrases:
        backfill_bm25_phrases(client)
        return

    if args.ingest:
        ingest_docs(client)
    else:
        print("skip ingest, search existing collection")

    # Batch Q&A is eval-layer only; core stays in pipeline.answer_query
    run_batch(
        client,
        dialogs=load_dialogs(args.dialogs),
        reporters=build_reporters(args),
    )


if __name__ == "__main__":
    main()
