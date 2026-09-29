"""Batch eval runner: core pipeline + pluggable reporters."""

from __future__ import annotations

from datetime import datetime

from loaders import Dialog, load_dialogs
from pipeline import answer_query
from tracing import flush as langfuse_flush

from .reporters import Reporter, TurnRef, default_reporters


def _trace_attrs(ref: TurnRef, run_id: str) -> dict:
    """Langfuse session / tags / metadata for one turn."""
    tags = ["eval", "multi_turn" if ref.n_turns > 1 else "single_turn"]
    if ref.turn > 1:
        tags.append("follow_up")
    metadata = {
        "eval_run": run_id,
        "index": str(ref.index),
        "turn": str(ref.turn),
        "n_turns": str(ref.n_turns),
    }
    if ref.dialog_id is not None:
        metadata["dialog_id"] = ref.dialog_id
    return {
        # one session per dialog per run, so reruns don't merge into old sessions
        "session_id": f"{run_id}/{ref.dialog_id}" if ref.dialog_id else None,
        "tags": tags,
        "trace_metadata": metadata,
    }


def run_batch(
    client,
    dialogs: list[Dialog] | None = None,
    reporters: list[Reporter] | None = None,
):
    """
    Run core answer_query for every turn of every dialog and fan out to reporters.

    dialogs default to data/dialogs.yaml.
    reporters default to console + excel + txt under data/out/.
    Pass a custom list to plug in other sinks (JSONL, DB, …).
    """
    dialogs = dialogs if dialogs is not None else load_dialogs()
    reporters = reporters if reporters is not None else default_reporters()

    for r in reporters:
        r.start()

    run_id = datetime.now().strftime("eval-%Y%m%d-%H%M%S")
    try:
        index = 0
        for dialog in dialogs:
            history: list[dict] = []
            for turn, query in enumerate(dialog.turns, 1):
                index += 1
                ref = TurnRef(
                    index=index,
                    dialog_id=dialog.id,
                    turn=turn,
                    n_turns=len(dialog.turns),
                )
                result = answer_query(
                    client, query, history=history, **_trace_attrs(ref, run_id)
                )
                history.append(
                    {"question": result.standalone_query, "answer": result.answer_text}
                )
                for r in reporters:
                    r.record(ref, result)
    finally:
        for r in reporters:
            r.finish()
        langfuse_flush()
