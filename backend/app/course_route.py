"""Resolve which lecture(s) of the current course a question is about.

`session` = {timestamp, current_quarter, current_course, current_lecture,
user_courses} — one dialog's entry in data/dialogs.yaml (see loaders.Dialog).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import COURSE_ROUTE_MODEL, DEFAULT_LECTURE_MAX_TS, LECTURES_DIR
from config.prompts import COURSE_ROUTE_SYSTEM
from llm import ollama_chat


def _load_lecture_meta(course_id: str, lecture_id: str) -> dict:
    path = LECTURES_DIR / course_id / lecture_id / "meta.json"
    if not path.exists():
        # legacy flat: data/lectures/<lecture_id>/meta.json
        path = LECTURES_DIR / lecture_id / "meta.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lecture_ids(course: dict) -> list[str]:
    raw = course.get("lecture_id") or []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _current_course(session: dict) -> dict:
    for c in session["user_courses"]:
        if c.get("course_id") == session["current_course"]:
            return c
    raise RuntimeError(
        f"current_course={session['current_course']!r} not found in user_courses"
    )


def _normalize_lecture_ids(raw) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    out = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def build_course_ctx(
    course: dict,
    lecture_ids: list[str],
    session: dict,
    *,
    reason: str = "",
) -> dict:
    """Merge enrollment row + lecture meta + session playback timestamp.

    Empty lecture_ids → whole current course (no lecture_id filter).
    course_id / quarter / lecturer always come from the current-course enrollment row.
    """
    catalog = _lecture_ids(course)
    allowed = set(catalog)
    current = session["current_lecture"]
    lids = [x for x in _normalize_lecture_ids(lecture_ids) if x in allowed]
    # Model dumped the full catalog → treat as whole-course (no lecture_id pin)
    if lids and allowed and set(lids) >= allowed:
        lids = []
    if lids:
        primary = lids[0]
    else:
        primary = current if current in allowed else (catalog[0] if catalog else current)
    meta = _load_lecture_meta(course["course_id"], primary)
    return {
        "lecture_id": primary,  # primary (e.g. playback / max_ts)
        "lecture_ids": lids,
        "course_id": course["course_id"],
        "quarter": course.get("quarter") or meta.get("quarter", ""),
        "lecturer": course.get("lecturer") or meta.get("lecturer", ""),
        "lecture_max_ts": meta.get("lecture_max_ts", DEFAULT_LECTURE_MAX_TS),
        "timestamp": session["timestamp"],
        "reason": reason,
    }


def session_course_ctx(session: dict, reason: str = "session baseline") -> dict:
    """Current course + current lecture without LLM — for parallel plan siblings."""
    course = _current_course(session)
    lids = _lecture_ids(course)
    current = session["current_lecture"]
    lecture_id = current if current in lids else (lids[0] if lids else current)
    return build_course_ctx(course, [lecture_id], session, reason=reason)


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def route_course(query: str, session: dict) -> dict:
    """
    Always scoped to current_course.
    - use_current → [current_lecture]
    - named lecture(s) → those ids
    - whole-course (null/empty/full catalog) → [] (course_id filter only)
    """
    course = _current_course(session)
    allowed = _lecture_ids(course)
    catalog = {
        "course_id": course["course_id"],
        "quarter": course["quarter"],
        "lecturer": course["lecturer"],
        "lecture_ids": allowed,
    }
    raw = ollama_chat(
        [
            {"role": "system", "content": COURSE_ROUTE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"current_quarter: {session['current_quarter']}\n"
                    f"current_course: {session['current_course']}\n"
                    f"current_lecture: {session['current_lecture']}\n"
                    f"catalog: {json.dumps(catalog, ensure_ascii=False)}\n"
                    f"question: {query}"
                ),
            },
        ],
        format="json",
        model=COURSE_ROUTE_MODEL,
        name="llm.course_route",
    )
    data = _parse_json(raw)
    reason = str(data.get("reason") or "").strip()
    use_current = data.get("use_current", True)
    if isinstance(use_current, str):
        use_current = use_current.strip().lower() in ("1", "true", "yes")

    if use_current:
        return session_course_ctx(session, reason or "no other lecture referenced")

    allowed_set = set(allowed)
    requested = _normalize_lecture_ids(data.get("lecture_ids"))
    if not requested:
        requested = _normalize_lecture_ids(data.get("lecture_id"))

    lids = [x for x in requested if x in allowed_set]
    return build_course_ctx(course, lids, session, reason=reason)
