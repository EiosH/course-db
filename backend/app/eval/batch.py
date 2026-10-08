"""Batch eval runner: core pipeline + pluggable reporters."""

from __future__ import annotations

from datetime import datetime

from loaders import SESSION_KEYS, Dialog, load_dialogs
from pipeline import answer_query
from tracing import flush as langfuse_flush
from tracing import observation, trace_attributes

from .reporters import Reporter, TurnRef, default_reporters


def _run_dialog(client, dialog: Dialog, start_index: int, run_id: str, reporters):
    """One Langfuse trace per dialog; each turn is a child observation."""
    n_turns = len(dialog.turns)
    trace_meta = {"eval_run": run_id, "n_turns": str(n_turns)}
    if dialog.id is not None:
        trace_meta["dialog_id"] = dialog.id
    session_meta = {k: dialog.session[k] for k in SESSION_KEYS}
    trace_meta.update(session_meta)
    tags = ["eval", "multi_turn" if n_turns > 1 else "single_turn"]

    history: list[dict] = []
    memory: dict | None = None
    with observation(
        name=dialog.id or "single_turn",
        as_type="chain",
        input={"dialog_id": dialog.id, "session": session_meta, "turns": dialog.turns},
    ) as root, trace_attributes(
        # all dialogs of one run share a session
        root, session_id=run_id, tags=tags, metadata=trace_meta
    ):
        for turn, query in enumerate(dialog.turns, 1):
            ref = TurnRef(
                index=start_index + turn - 1,
                dialog_id=dialog.id,
                turn=turn,
                n_turns=n_turns,
            )
            result = answer_query(
                client,
                query,
                history=history,
                memory=memory,
                session=dialog.session,
                name=f"turn {turn}",
                metadata={"turn": turn, "index": ref.index},
            )
            memory = result.memory
            history.append(
                {"question": result.standalone_query, "answer": result.answer_text}
            )
            for r in reporters:
                r.record(ref, result)
        if root is not None:
            root.update(
                output={
                    "turns": [
                        {
                            "question": q,
                            "standalone_query": h["question"],
                            "answer": h["answer"],
                        }
                        for q, h in zip(dialog.turns, history)
                    ],
                    # summary as of the last turn (older turns only)
                    "summary": (memory or {}).get("summary", ""),
                }
            )
            print(f"langfuse: trace_id={root.trace_id}")
    return n_turns


def run_batch(
    client,
    dialogs: list[Dialog] | None = None,
    reporters: list[Reporter] | None = None,
):
    """
    Run core answer_query for every turn of every dialog and fan out to reporters.

    dialogs default to data/dialogs.yaml.
    reporters default to console only; Langfuse gets one trace per dialog.
    Pass a custom list to plug in other sinks (Excel, txt, JSONL, …).
    """
    dialogs = dialogs if dialogs is not None else load_dialogs()
    reporters = reporters if reporters is not None else default_reporters()

    for r in reporters:
        r.start()

    run_id = datetime.now().strftime("eval-%Y%m%d-%H%M%S")
    try:
        index = 1
        for dialog in dialogs:
            index += _run_dialog(client, dialog, index, run_id, reporters)
    finally:
        for r in reporters:
            r.finish()
        langfuse_flush()
