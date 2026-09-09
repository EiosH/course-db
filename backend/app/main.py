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
    COURSE_ID,
    DOC_PATH,
    LECTURER,
    QDRANT_URL,
    QUARTER,
    TRANSCRIPT_PATH,
    UPSERT_BATCH,
)
from eval import default_reporters, run_batch
from eval.reporters import ConsoleReporter, ExcelReporter, TxtReporter
from llm import embed
from loaders import load_screenshots, load_transcript


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
        "--no-excel",
        action="store_true",
        help="eval: skip Excel reporter",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="eval: skip txt reporter",
    )
    parser.add_argument(
        "--excel-only",
        action="store_true",
        help="eval: only Excel (+ console)",
    )
    parser.add_argument(
        "--txt-only",
        action="store_true",
        help="eval: only txt (+ console)",
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
    chunks = load_screenshots(DOC_PATH) + load_transcript(TRANSCRIPT_PATH)
    print(
        f"screen_shot {sum(c['type']=='screen_shot' for c in chunks)} 段, "
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
    for field in ("course_id", "quarter", "lecturer", "type", "timestamp"):
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
                        "course_id": COURSE_ID,
                        "quarter": QUARTER,
                        "lecturer": LECTURER,
                    },
                )
                for j, c in enumerate(batch)
            ],
        )
        print(f"upsert {min(start + UPSERT_BATCH, total)}/{total}")


def build_reporters(args):
    """Assemble pluggable eval sinks from CLI flags."""
    if args.excel_only:
        return [ConsoleReporter(), ExcelReporter()]
    if args.txt_only:
        return [ConsoleReporter(), TxtReporter()]
    return default_reporters(excel=not args.no_excel, txt=not args.no_txt)


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
    run_batch(client, reporters=build_reporters(args))


if __name__ == "__main__":
    main()
