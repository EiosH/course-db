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

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  turn_index INT UNSIGNED NOT NULL,
  question TEXT NOT NULL,
  answer MEDIUMTEXT NOT NULL,
  summary TEXT NOT NULL,
  summarized_turns INT UNSIGNED NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_session_turn (session_id, turn_index),
  KEY idx_session_id (session_id)
)
"""

_schema_ready = False


class HistoryIncompleteError(RuntimeError):
    """Prior turn(s) in MySQL have an empty answer; multi-turn cannot continue."""


def mock_session_id() -> str:
    """Temporary session id until a real client/session layer exists."""
    return uuid.uuid4().hex


def _require_history_answers(history: list[dict]) -> None:
    """Abort if any earlier turn is missing an answer — do not keep going blind."""
    for i, turn in enumerate(history, 1):
        if (turn.get("answer") or "").strip():
            continue
        q = (turn.get("question") or "").strip()
        q_preview = q if len(q) <= 80 else q[:77] + "..."
        raise HistoryIncompleteError(
            f"history turn {i}/{len(history)} has empty answer "
            f"(question={q_preview!r}); refusing to continue this session"
        )


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


def ensure_schema() -> None:
    """Create conversation table if missing (init scripts only run on empty datadir)."""
    global _schema_ready
    if _schema_ready:
        return
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    _schema_ready = True
    print(
        f"mysql: conversation ready at {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
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
    ensure_schema()
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
        print(f"mysql: load session={session_id[:8]}… turns=0")
        return [], None
    history = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    last = rows[-1]
    memory = {
        "summary": last["summary"] or "",
        "summarized_turns": int(last["summarized_turns"] or 0),
    }
    print(
        f"mysql: load session={session_id[:8]}… turns={len(history)} "
        f"summarized_turns={memory['summarized_turns']}"
    )
    _require_history_answers(history)
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
    if not (answer or "").strip():
        raise RuntimeError(
            f"refusing to persist empty answer "
            f"(session={session_id[:8]}… question={(question or '')[:80]!r})"
        )
    summary = (memory or {}).get("summary") or ""
    summarized_turns = int((memory or {}).get("summarized_turns") or 0)
    ensure_schema()
    # One connection + commit: index allocation and INSERT land together.
    # uk_session_turn rejects concurrent duplicate turn_index.
    conn = _connect()
    try:
        conn.autocommit(False)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(turn_index), 0) AS n "
                "FROM conversation WHERE session_id = %s",
                (session_id,),
            )
            turn_index = int(cur.fetchone()["n"]) + 1
            cur.execute(
                """
                INSERT INTO conversation
                  (session_id, turn_index, question, answer, summary, summarized_turns)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    turn_index,
                    question or "",
                    answer or "",
                    summary,
                    summarized_turns,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(
        f"mysql: append session={session_id[:8]}… turn={turn_index} "
        f"q_chars={len(question or '')} a_chars={len(answer or '')}"
    )
    return turn_index


def answer_session_turn(client, query: str, session_id: str, *, session: dict, **kwargs):
    """
    One turn keyed by session_id: load history+memory → answer → persist Q/A.

    MySQL stores only conversation history (standalone question, answer, rolling
    summary). Pipeline detail (plan / retrieve / tokens) lives in Langfuse only.

    session_id: conversation key (mock now; real client id later).
    session: playback / enrollment context for the pipeline (not the DB key).
    Extra kwargs go to answer_query (name, metadata, …).
    """
    from pipeline import answer_query
    from tracing import flush as langfuse_flush

    # Reject kwargs that would clash with the loaded session state.
    for key in ("history", "memory", "session"):
        if key in kwargs:
            raise TypeError(
                f"answer_session_turn() got unexpected keyword argument {key!r}; "
                "history/memory come from MySQL, session is a dedicated argument"
            )

    history, memory = load_session(session_id)
    try:
        result = answer_query(
            client,
            query,
            history=history,
            memory=memory,
            session=session,
            **kwargs,
        )
    finally:
        # Turn observation subtree has ended; ship it now (do not wait for
        # dialog root / batch). Parent root trace stays open for later turns.
        langfuse_flush()
    append_turn(
        session_id,
        result.standalone_query,
        result.answer_text,
        result.memory,
    )
    return result
