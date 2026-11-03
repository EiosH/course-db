"""Persist / load multi-turn dialog history in MySQL by session_id."""

from __future__ import annotations

import uuid

import pymysql
from pymysql.cursors import DictCursor

from config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


def mock_session_id() -> str:
    """Temporary session id until a real client/session layer exists."""
    return uuid.uuid4().hex


def _connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=DictCursor,
    )


def load_session(session_id: str) -> tuple[list[dict], dict | None]:
    """
    Load earlier turns for this session.

    Returns (history, memory):
      history: [{"question", "answer"}, ...] oldest first
      memory: {"summary", "summarized_turns"} from the latest turn, or None
    """
    if not session_id:
        raise ValueError("session_id is required")
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT question, answer, summary, summarized_turns
            FROM conversation
            WHERE session_id = %s
            ORDER BY turn_index ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return [], None
    history = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    last = rows[-1]
    memory = {
        "summary": last["summary"] or "",
        "summarized_turns": int(last["summarized_turns"] or 0),
    }
    return history, memory


def append_turn(
    session_id: str,
    question: str,
    answer: str,
    memory: dict | None,
) -> int:
    """
    Append one turn under session_id. Returns the new turn_index (1-based).

    question / answer should match what answer_query expects in history
    (standalone question + answer text). memory is the rolling summary state
    after this turn.
    """
    if not session_id:
        raise ValueError("session_id is required")
    summary = (memory or {}).get("summary") or ""
    summarized_turns = int((memory or {}).get("summarized_turns") or 0)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(MAX(turn_index), 0) AS n FROM conversation WHERE session_id = %s",
            (session_id,),
        )
        turn_index = int(cur.fetchone()["n"]) + 1
        cur.execute(
            """
            INSERT INTO conversation
              (session_id, turn_index, question, answer, summary, summarized_turns)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (session_id, turn_index, question, answer, summary, summarized_turns),
        )
    return turn_index


def answer_session_turn(client, query: str, session_id: str, *, session: dict, **kwargs):
    """
    One turn keyed by session_id: load history+memory → answer → persist.

    session_id: conversation key (mock now; real client id later).
    session: playback / enrollment context for the pipeline (not the DB key).
    Extra kwargs go to answer_query (name, metadata, …).
    """
    from pipeline import answer_query

    history, memory = load_session(session_id)
    result = answer_query(
        client,
        query,
        history=history,
        memory=memory,
        session=session,
        **kwargs,
    )
    append_turn(
        session_id,
        result.standalone_query,
        result.answer_text,
        result.memory,
    )
    return result
