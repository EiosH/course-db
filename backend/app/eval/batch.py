"""Batch eval runner: core pipeline + pluggable reporters."""

from __future__ import annotations

from loaders import load_queries
from pipeline import answer_query

from .reporters import Reporter, default_reporters


def run_batch(client, queries: list[str] | None = None, reporters: list[Reporter] | None = None):
    """
    Run core answer_query for each question and fan out to reporters.

    reporters default to console + excel + txt under data/out/.
    Pass a custom list to plug in other sinks (JSONL, DB, …).
    """
    queries = queries if queries is not None else load_queries()
    reporters = reporters if reporters is not None else default_reporters()

    for r in reporters:
        r.start()

    for i, query in enumerate(queries, 1):
        result = answer_query(client, query)
        for r in reporters:
            r.record(i, result)

    for r in reporters:
        r.finish()
