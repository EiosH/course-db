"""CLI entry: ingest / backfill. Default: idle keep-alive for Docker."""

import argparse
import time

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
from embed import embed
from loaders import load_all_lecture_chunks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="embed + upsert only lectures whose (course_id, lecture_id) is not in the collection yet",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection and re-ingest every lecture",
    )
    parser.add_argument(
        "--backfill-bm25-phrases",
        action="store_true",
        help="rebuild BM25 sparse vectors with phrase tokens from existing payloads (no dense re-embed)",
    )
    return parser.parse_args()


def backfill_bm25_phrases(client):
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

    upsert_chunks(client, chunks, vectors, first_id=0)


def ingest_new_docs(client):
    if not client.collection_exists(COLLECTION_NAME):
        print("collection missing, falling back to full ingest")
        ingest_docs(client)
        return

    existing = set()
    for course in client.facet(COLLECTION_NAME, "course_id", limit=10_000).hits:
        lectures = client.facet(
            COLLECTION_NAME,
            "lecture_id",
            facet_filter=Filter(
                must=[FieldCondition(key="course_id", match=MatchValue(value=course.value))]
            ),
            limit=10_000,
        )
        existing.update((course.value, lec.value) for lec in lectures.hits)

    chunks = load_all_lecture_chunks(skip=existing)
    if not chunks:
        print("no new lectures to ingest")
        return
    new_lectures = sorted({f"{c['course_id']}/{c['lecture_id']}" for c in chunks})
    print(f"new lectures: {', '.join(new_lectures)} ({len(chunks)} chunks)")

    next_id = client.count(COLLECTION_NAME, exact=True).count
    upsert_chunks(client, chunks, embed([c["text"] for c in chunks]), first_id=next_id)


def upsert_chunks(client, chunks, vectors, first_id):
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH):
        batch = chunks[start : start + UPSERT_BATCH]
        client.upsert(
            COLLECTION_NAME,
            points=[
                PointStruct(
                    id=first_id + start + j,
                    vector={
                        "dense": vectors[start + j],
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
                        "course_id": c["course_id"],
                        "quarter": c["quarter"],
                        "lecturer": c["lecturer"],
                    },
                )
                for j, c in enumerate(batch)
            ],
        )
        print(f"upsert {min(start + UPSERT_BATCH, total)}/{total}")


def main():
    args = parse_args()
    if not (args.ingest or args.rebuild or args.backfill_bm25_phrases):
        print("ingest service idle; waiting for jobs")
        while True:
            time.sleep(3600)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    if args.backfill_bm25_phrases:
        backfill_bm25_phrases(client)
        return

    if args.rebuild:
        ingest_docs(client)
    elif args.ingest:
        ingest_new_docs(client)


if __name__ == "__main__":
    main()
