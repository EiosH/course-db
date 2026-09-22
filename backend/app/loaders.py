"""Load queries, screenshot OCR dumps, and VTT transcript chunks."""

import re
from pathlib import Path

from config import (
    CUE_RE,
    LECTURES_DIR,
    QUERY_PATH,
    SCREEN_BOILERPLATE_RE,
    SCREEN_LABEL_RE,
    TRANSCRIPT_MAX_CHARS,
    TRANSCRIPT_MAX_GAP_SEC,
)
from time_utils import ts_to_sec


def load_queries(path: str | Path | None = None):
    path = Path(path) if path else QUERY_PATH
    queries = []
    for line in open(path, encoding="utf-8"):
        line = re.sub(r"^\d+\.\s*", "", line.strip())
        if line:
            queries.append(line)
    return queries


def clean_screenshot_text(body: str) -> str:
    """Strip OCR layout labels / UI chrome that pollute BM25 keyword recall."""
    text = SCREEN_LABEL_RE.sub("", body)
    text = SCREEN_BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or body.strip()


def load_screenshots(path: str | Path, *, lecture_id: str = ""):
    path = Path(path)
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
                }
            )
    return chunks


def load_transcript(path: str | Path, *, lecture_id: str = ""):
    path = Path(path)
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
    """Subdirs of data/lectures/ that contain doc.txt and/or transcript.vtt."""
    root = Path(lectures_dir) if lectures_dir else LECTURES_DIR
    if not root.is_dir():
        return []
    found = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "doc.txt").exists() or (d / "transcript.vtt").exists():
            found.append(d)
    return found


def load_all_lecture_chunks(lectures_dir: str | Path | None = None) -> list[dict]:
    """Load every lecture folder under data/lectures/."""
    chunks = []
    dirs = iter_lecture_dirs(lectures_dir)
    if not dirs:
        print(f"warning: no lecture folders under {lectures_dir or LECTURES_DIR}")
        return chunks
    for d in dirs:
        lecture_id = d.name
        doc = d / "doc.txt"
        vtt = d / "transcript.vtt"
        n_ss = n_tr = 0
        if doc.exists():
            ss = load_screenshots(doc, lecture_id=lecture_id)
            chunks.extend(ss)
            n_ss = len(ss)
        if vtt.exists():
            tr = load_transcript(vtt, lecture_id=lecture_id)
            chunks.extend(tr)
            n_tr = len(tr)
        print(f"lecture {lecture_id}: screen_shot={n_ss} transcript={n_tr}")
    return chunks
