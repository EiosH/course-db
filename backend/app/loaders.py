"""Load eval dialogs, screenshot OCR dumps, and VTT transcript chunks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from config import (
    CUE_RE,
    DIALOGS_PATH,
    LECTURES_DIR,
    SCREEN_BOILERPLATE_RE,
    SCREEN_LABEL_RE,
    TRANSCRIPT_MAX_CHARS,
    TRANSCRIPT_MAX_GAP_SEC,
)
from time_utils import normalize_ts_hms, ts_to_sec

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
        courses.append({**c, "lecture_id": [str(x) for x in lids]})
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


def load_lecture_meta(lecture_dir: str | Path) -> dict:
    """Per-course mock metadata written into Qdrant payloads."""
    lecture_dir = Path(lecture_dir)
    path = lecture_dir / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("course_id", "quarter", "lecturer"):
        if key not in meta or not meta[key]:
            raise ValueError(f"{path} missing required field {key!r}")
    return {
        "course_id": meta["course_id"],
        "quarter": meta["quarter"],
        "lecturer": meta["lecturer"],
        "lecture_max_ts": meta.get("lecture_max_ts", "03:30:00"),
    }


def clean_screenshot_text(body: str) -> str:
    """Strip OCR layout labels / UI chrome that pollute BM25 keyword recall."""
    text = SCREEN_LABEL_RE.sub("", body)
    text = SCREEN_BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or body.strip()


def load_screenshots(path: str | Path, *, lecture_id: str = "", meta: dict | None = None):
    path = Path(path)
    meta = meta or {}
    raw = open(path, encoding="utf-8").read()
    parts = re.split(r"=+\n时间戳: (\d{2}:\d{2}:\d{2}).*?\n=+\n", raw)
    chunks = []
    for i in range(1, len(parts), 2):
        ts, body = parts[i], parts[i + 1].strip()
        body = re.sub(r"\n=+\n视频处理统计[\s\S]*$", "", body).strip()
        body = clean_screenshot_text(body)
        if body:
            sec = ts_to_sec(ts)
            chunks.append(
                {
                    "timestamp": ts,
                    "start_sec": sec,
                    "end_sec": sec,
                    "text": body,
                    "type": "screen_shot",
                    "lecture_id": lecture_id,
                    "source_file": path.name,
                    "course_id": meta.get("course_id", ""),
                    "quarter": meta.get("quarter", ""),
                    "lecturer": meta.get("lecturer", ""),
                }
            )
    return chunks


def load_transcript(path: str | Path, *, lecture_id: str = "", meta: dict | None = None):
    path = Path(path)
    meta = meta or {}
    try:
        raw = open(path, encoding="utf-8").read().strip()
    except FileNotFoundError:
        return []
    if not raw:
        return []
    raw = re.sub(r"^WEBVTT\s*", "", raw, flags=re.I)
    cues = []
    for m in CUE_RE.finditer(raw):
        text = re.sub(r"\s+", " ", m.group(3)).strip()
        if text:
            cues.append({"start": m.group(1), "end": m.group(2), "text": text})

    chunks, buf, start, end, n = [], [], None, None, 0

    def flush():
        nonlocal buf, start, end, n
        if buf:
            chunks.append(
                {
                    "timestamp": f"{start} --> {end}",
                    "start_sec": ts_to_sec(start),
                    "end_sec": ts_to_sec(end),
                    "text": " ".join(buf),
                    "type": "transcript",
                    "lecture_id": lecture_id,
                    "source_file": path.name,
                    "course_id": meta.get("course_id", ""),
                    "quarter": meta.get("quarter", ""),
                    "lecturer": meta.get("lecturer", ""),
                }
            )
        buf, start, end, n = [], None, None, 0

    for c in cues:
        gap = ts_to_sec(c["start"]) - ts_to_sec(end) if end else 0
        if buf and (
            gap > TRANSCRIPT_MAX_GAP_SEC
            or n + len(c["text"]) + 1 > TRANSCRIPT_MAX_CHARS
        ):
            flush()
        if start is None:
            start = c["start"]
        buf.append(c["text"])
        end = c["end"]
        n += len(c["text"]) + 1
    flush()
    return chunks


def iter_lecture_dirs(lectures_dir: str | Path | None = None) -> list[Path]:
    """
    Lecture material folders:
      data/lectures/<course_id>/<lecture_id>/{doc.txt, transcript.vtt, meta.json}
    Also accepts legacy flat: data/lectures/<lecture_id>/...
    """
    root = Path(lectures_dir) if lectures_dir else LECTURES_DIR
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "doc.txt").exists() or (child / "transcript.vtt").exists():
            # legacy flat lecture folder
            found.append(child)
            continue
        for lec in sorted(child.iterdir()):
            if not lec.is_dir():
                continue
            if (lec / "doc.txt").exists() or (lec / "transcript.vtt").exists():
                found.append(lec)
    return found


def load_all_lecture_chunks(
    lectures_dir: str | Path | None = None, skip: set[tuple[str, str]] = frozenset()
) -> list[dict]:
    """Load every lecture folder under data/lectures/ except (course_id, lecture_id) in skip."""
    chunks = []
    dirs = iter_lecture_dirs(lectures_dir)
    if not dirs:
        print(f"warning: no lecture folders under {lectures_dir or LECTURES_DIR}")
        return chunks
    for d in dirs:
        lecture_id = d.name
        meta = load_lecture_meta(d)
        # prefer path parent as course when nested: lectures/<course>/<lec>/
        if d.parent != (Path(lectures_dir) if lectures_dir else LECTURES_DIR):
            meta = {
                **meta,
                "course_id": meta.get("course_id") or d.parent.name,
            }
        if (meta["course_id"], lecture_id) in skip:
            continue
        doc = d / "doc.txt"
        vtt = d / "transcript.vtt"
        n_ss = n_tr = 0
        if doc.exists():
            ss = load_screenshots(doc, lecture_id=lecture_id, meta=meta)
            chunks.extend(ss)
            n_ss = len(ss)
        if vtt.exists():
            tr = load_transcript(vtt, lecture_id=lecture_id, meta=meta)
            chunks.extend(tr)
            n_tr = len(tr)
        print(
            f"lecture {meta.get('course_id', '?')}/{lecture_id}: "
            f"course={meta['course_id']} quarter={meta['quarter']} "
            f"lecturer={meta['lecturer']!r} "
            f"screen_shot={n_ss} transcript={n_tr}"
        )
    return chunks
