"""Load eval dialogs (lecture chunk parsing lives in the ingest service)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from config import DIALOGS_PATH, LECTURES_DIR
from time_utils import normalize_ts_hms

SESSION_KEYS = ("timestamp", "current_quarter", "current_course", "current_lecture")
COURSE_KEYS = ("course_id", "quarter", "lecturer", "lecture_id")


@dataclass(frozen=True)
class Dialog:
    """One eval conversation; turns run in order and share context."""

    id: str | None  # None = anonymous single-turn entry
    turns: list[str]
    # SESSION_KEYS + user_courses (the file-level enrollment list)
    session: dict


def load_dialogs(path: str | Path | None = None) -> list[Dialog]:
    """Eval input: data/dialogs.yaml by default."""
    path = Path(path) if path else DIALOGS_PATH
    return parse_dialogs(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}, source=path.name
    )


def _mock_outlines(course_id: str, lecture_ids: list[str]) -> dict[str, str]:
    """Mock: {lecture_id: text} from data/lectures/<course>/<lecture>/outline.txt."""
    outlines = {}
    for lid in lecture_ids:
        path = LECTURES_DIR / course_id / lid / "outline.txt"
        if path.exists():
            outlines[lid] = path.read_text(encoding="utf-8").strip()
    return outlines


def _parse_user_courses(raw, source: str) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: 'user_courses' must be a non-empty list")
    courses = []
    for i, c in enumerate(raw, 1):
        where = f"{source} user_courses item {i}"
        if not isinstance(c, dict) or any(not c.get(k) for k in COURSE_KEYS):
            raise ValueError(f"{where}: needs {', '.join(COURSE_KEYS)}")
        lids = c["lecture_id"]
        if not isinstance(lids, list):
            raise ValueError(f"{where}: lecture_id must be a list")
        lids = [str(x) for x in lids]
        courses.append(
            {**c, "lecture_id": lids, "outlines": _mock_outlines(c["course_id"], lids)}
        )
    return courses


def _parse_session(item: dict, courses: list[dict], where: str) -> dict:
    """Required per-dialog session, validated against the enrollment list."""
    unknown = set(item) - {"id", "turns", *SESSION_KEYS}
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}; "
            f"allowed: id, turns, {', '.join(SESSION_KEYS)}"
        )
    session = {}
    for key in SESSION_KEYS:
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f'{where}: {key} is required and must be a quoted string, e.g. "01:22:09"'
            )
        session[key] = value.strip()
    try:
        session["timestamp"] = normalize_ts_hms(session["timestamp"])
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None

    course = next(
        (c for c in courses if c["course_id"] == session["current_course"]), None
    )
    if course is None:
        raise ValueError(
            f"{where}: current_course {session['current_course']!r} not in user_courses"
        )
    if session["current_lecture"] not in course["lecture_id"]:
        raise ValueError(
            f"{where}: current_lecture {session['current_lecture']!r} "
            f"not in {session['current_course']} lectures"
        )
    return {**session, "user_courses": courses}


def parse_dialogs(raw, *, source: str = "dialogs") -> list[Dialog]:
    """
    Top level: {user_courses: [...], dialogs: [...]}.
    Each dialog: {id, turns: [question, ...], timestamp, current_quarter,
    current_course, current_lecture}. Multi-turn dialogs need a unique id.
    """
    if not isinstance(raw, dict) or "dialogs" not in raw:
        raise ValueError(f"{source}: top level must be {{user_courses, dialogs}}")
    courses = _parse_user_courses(raw.get("user_courses"), source)
    items = raw["dialogs"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{source}: 'dialogs' must be a non-empty list")
    dialogs: list[Dialog] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(items, 1):
        where = f"{source} dialog {i}"
        if not isinstance(item, dict) or "turns" not in item:
            raise ValueError(
                f"{where}: expected {{id, turns, timestamp, ...}} "
                "(quote questions that contain ': ')"
            )
        turns = item["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{where}: 'turns' must be a non-empty list")
        for t in turns:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"{where}: every turn must be a non-empty string")
        dialog_id = str(item.get("id") or "").strip() or None
        if dialog_id is None and len(turns) > 1:
            raise ValueError(f"{where}: multi-turn dialogs need an 'id'")
        if dialog_id is not None:
            if dialog_id in seen_ids:
                raise ValueError(f"{where}: duplicate id {dialog_id!r}")
            seen_ids.add(dialog_id)
        session = _parse_session(item, courses, where)
        dialogs.append(Dialog(dialog_id, [t.strip() for t in turns], session))
    return dialogs
