"""Batch eval runner: core pipeline + pluggable reporters."""

from __future__ import annotations

from datetime import datetime

from conversation_store import (
    HistoryIncompleteError,
    answer_session_turn,
    ensure_schema,
    mock_session_id,
)
from llm import warmup_retrieval_models
from loaders import SESSION_KEYS, Dialog, load_dialogs
from tracing import flush as langfuse_flush
from tracing import observation, trace_attributes

from .reporters import Reporter, TurnRef, default_reporters


def _run_dialog(client, dialog: Dialog, start_index: int, run_id: str, reporters):
    """
    One Langfuse root trace per dialog; each turn nests under it.
    Flush after every turn so pipeline spans ship without waiting for the
    whole dialog. MySQL only stores conversation history (via answer_session_turn).
    """
    n_turns = len(dialog.turns)
    # One conversation session for all turns of this dialog.
    conv_session_id = mock_session_id()
    print(f"conversation session_id={conv_session_id} dialog={dialog.id or 'single_turn'}")

    trace_meta = {
        "eval_run": run_id,
        "n_turns": str(n_turns),
        "conversation_session_id": conv_session_id,
    }
    if dialog.id is not None:
        trace_meta["dialog_id"] = dialog.id
    session_meta = {k: dialog.session[k] for k in SESSION_KEYS}
    trace_meta.update(session_meta)
    tags = ["eval", "multi_turn" if n_turns > 1 else "single_turn"]

    recorded: list[dict] = []
    memory: dict | None = None

    with observation(
        name=dialog.id or "single_turn",
        as_type="chain",
        input={
            "dialog_id": dialog.id,
            "session": session_meta,
            "conversation_session_id": conv_session_id,
            "turns": dialog.turns,
        },
    ) as root, trace_attributes(
        # Eval batch Langfuse session; each dialog still has its own root trace.
        root, session_id=run_id, tags=tags, metadata=trace_meta
    ):
        try:
            for turn, query in enumerate(dialog.turns, 1):
                ref = TurnRef(
                    index=start_index + turn - 1,
                    dialog_id=dialog.id,
                    turn=turn,
                    n_turns=n_turns,
                )
                result = answer_session_turn(
                    client,
                    query,
                    conv_session_id,
                    session=dialog.session,
                    name=f"turn {turn}",
                    metadata={
                        "turn": turn,
                        "index": ref.index,
                        "conversation_session_id": conv_session_id,
                    },
                )
                # answer_session_turn already flushed the turn subtree.
                memory = result.memory
                recorded.append(
                    {"question": result.standalone_query, "answer": result.answer_text}
                )
                for r in reporters:
                    r.record(ref, result)
                if root is not None:
                    root.update(
                        output={
                            "conversation_session_id": conv_session_id,
                            "turns": [
                                {
                                    "question": q,
                                    "standalone_query": h["question"],
                                    "answer": h["answer"],
                                }
                                for q, h in zip(dialog.turns, recorded)
                            ],
                            "summary": (memory or {}).get("summary", ""),
                        }
                    )
                    print(f"langfuse: trace_id={root.trace_id} turn={turn}/{n_turns}")
                    # Push root progress for this turn (children already flushed).
                    langfuse_flush()
        except HistoryIncompleteError as e:
            print(f"warning: abort dialog {dialog.id or 'single_turn'}: {e}")
            if root is not None:
                root.update(level="WARNING", status_message=str(e))
                langfuse_flush()
            raise SystemExit(1) from e
    return n_turns


def run_batch(
    client,
    dialogs: list[Dialog] | None = None,
    reporters: list[Reporter] | None = None,
):
    """
    Run core answer_query for every turn of every dialog and fan out to reporters.

    dialogs default to data/dialogs.yaml.
    reporters default to console only; Langfuse gets one root trace per dialog.
    Pass a custom list to plug in other sinks (Excel, txt, JSONL, …).
    """
    dialogs = dialogs if dialogs is not None else load_dialogs()
    reporters = reporters if reporters is not None else default_reporters()

    ensure_schema()
    # Turn 1 is often outline-skip (no embed). Warm here so turn 2 recall
    # does not pay the CUDA load in the middle of the dialog.
    warmup_retrieval_models()

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
