"""Resolve which lecture(s) of the current course a question is about."""

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


def _load_lecture_meta(course_id: str, lecture_id: str) -> dict:
    path = LECTURES_DIR / course_id / lecture_id / "meta.json"
    if not path.exists():
        # legacy flat: data/lectures/<lecture_id>/meta.json
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


def _current_course() -> dict:
    course = _find_course(CURRENT_COURSE)
    if not course:
        raise RuntimeError(
            f"current_course={CURRENT_COURSE!r} not found in user_courses.courses"
        )
    return course


def _current_lecture_catalog() -> list[str]:
    """lecture_id values allowed for routing — current_course only."""
    return _lecture_ids(_current_course())


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
    *,
    reason: str = "",
) -> dict:
    """Merge enrollment row + lecture meta + session playback timestamp."""
    allowed = set(_lecture_ids(course))
    lids = [x for x in _normalize_lecture_ids(lecture_ids) if x in allowed]
    if not lids:
        lids = list(allowed) or [CURRENT_LECTURE]
    primary = lids[0]
    meta = _load_lecture_meta(course.get("course_id", ""), primary)
    return {
        "lecture_id": primary,  # primary (e.g. playback / max_ts)
        "lecture_ids": lids,
        "course_id": course.get("course_id") or meta.get("course_id", ""),
        "quarter": course.get("quarter") or meta.get("quarter", ""),
        "lecturer": course.get("lecturer") or meta.get("lecturer", ""),
        "lecture_max_ts": meta.get("lecture_max_ts", DEFAULT_LECTURE_MAX_TS),
        "timestamp": MOCK_SESSION["timestamp"],
        "reason": reason,
    }


def _default_ctx(reason: str = "current course/lecture") -> dict:
    course = _current_course()
    lids = _lecture_ids(course)
    lecture_id = (
        CURRENT_LECTURE if CURRENT_LECTURE in lids else (lids[0] if lids else CURRENT_LECTURE)
    )
    return build_course_ctx(course, [lecture_id], reason=reason)


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def route_course(query: str) -> dict:
    """
    Pick lecture_ids only from current_course.
    No lecture deixis → [current_lecture]; else one/many/all of current_course.
    """
    course = _current_course()
    allowed = _current_lecture_catalog()
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
        return _default_ctx(reason or "no other lecture referenced")

    allowed_set = set(allowed)
    requested = _normalize_lecture_ids(data.get("lecture_ids"))
    if not requested:
        requested = _normalize_lecture_ids(data.get("lecture_id"))

    # only keep ids that belong to current_course
    lids = [x for x in requested if x in allowed_set]
    if not lids:
        # whole-course / invalid → all lectures of current_course
        lids = list(allowed) or [CURRENT_LECTURE]
    return build_course_ctx(course, lids, reason=reason)
