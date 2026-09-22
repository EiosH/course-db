"""Resolve which enrolled course / lecture a question is about."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import (
    COURSE_ROUTE_MODEL,
    CURRENT_COURSE,
    CURRENT_LECTURE,
    CURRENT_QUARTER,
    DEFAULT_LECTURE_MAX_TS,
    LECTURES_DIR,
    MOCK_SESSION,
    USER_COURSES,
)
from config.prompts import COURSE_ROUTE_SYSTEM
from llm import ollama_chat


def _load_lecture_meta(lecture_id: str) -> dict:
    path = LECTURES_DIR / lecture_id / "meta.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _courses() -> list[dict]:
    courses = list(USER_COURSES.get("courses") or [])
    if not courses:
        raise RuntimeError("MOCK_SESSION.user_courses.courses is empty")
    return courses


def _lecture_ids(course: dict) -> list[str]:
    raw = course.get("lecture_id") or []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _find_course(course_id: str) -> dict | None:
    for c in _courses():
        if c.get("course_id") == course_id:
            return c
    return None


def _catalog() -> list[dict]:
    """Flatten courses × lecture_id[] for the router."""
    rows = []
    for c in _courses():
        for lid in _lecture_ids(c):
            rows.append(
                {
                    "course_id": c["course_id"],
                    "quarter": c["quarter"],
                    "lecturer": c["lecturer"],
                    "lecture_id": lid,
                }
            )
    return rows


def build_course_ctx(
    course: dict,
    lecture_id: str,
    *,
    reason: str = "",
) -> dict:
    """Merge enrollment row + lecture meta + session playback timestamp."""
    meta = _load_lecture_meta(lecture_id)
    return {
        "lecture_id": lecture_id,
        "course_id": course.get("course_id") or meta.get("course_id", ""),
        "quarter": course.get("quarter") or meta.get("quarter", ""),
        "lecturer": course.get("lecturer") or meta.get("lecturer", ""),
        "lecture_max_ts": meta.get("lecture_max_ts", DEFAULT_LECTURE_MAX_TS),
        "timestamp": MOCK_SESSION["timestamp"],
        "reason": reason,
    }


def _default_ctx(reason: str = "current course/lecture") -> dict:
    course = _find_course(CURRENT_COURSE) or _courses()[0]
    lecture_id = CURRENT_LECTURE
    lids = _lecture_ids(course)
    if lecture_id not in lids:
        lecture_id = lids[0] if lids else CURRENT_LECTURE
    return build_course_ctx(course, lecture_id, reason=reason)


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def route_course(query: str) -> dict:
    """
    Route to course + lecture.
    No deixis to another course/lecture → current_course / current_lecture.
    Otherwise pick from catalog by semantics.
    """
    catalog = _catalog()
    raw = ollama_chat(
        [
            {"role": "system", "content": COURSE_ROUTE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"current_quarter: {CURRENT_QUARTER}\n"
                    f"current_course: {CURRENT_COURSE}\n"
                    f"current_lecture: {CURRENT_LECTURE}\n"
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
        return _default_ctx(reason or "no other course/lecture referenced")

    course_id = str(data.get("course_id") or "").strip() or CURRENT_COURSE
    lecture_id = str(data.get("lecture_id") or "").strip() or CURRENT_LECTURE
    valid = {(r["course_id"], r["lecture_id"]) for r in catalog}
    if (course_id, lecture_id) not in valid:
        # course ok but lecture missing → first lecture of that course
        course = _find_course(course_id)
        if course:
            lids = _lecture_ids(course)
            if lids:
                return build_course_ctx(
                    course, lids[0], reason=reason or "fallback lecture on course"
                )
        return _default_ctx(reason or "invalid route; fallback to current")

    return build_course_ctx(
        _find_course(course_id),
        lecture_id,
        reason=reason,
    )
