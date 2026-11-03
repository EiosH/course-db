"""Ingest config: paths, embed/Qdrant settings, lecture parsing regexes."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(APP_DIR / ".env")

from config.models import *  # noqa: E402,F401,F403

LECTURES_DIR = Path(
    os.getenv("LECTURES_DIR", str(REPO_ROOT / "data" / "lectures"))
)

DEFAULT_LECTURE_MAX_TS = "03:30:00"

ANCHOR_PHRASE_RE = re.compile(r"(?i)\b([a-z]+)(?:\s*#\s*|\s+)(\d+)\b")
ANCHOR_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "be",
        "by",
        "at",
        "as",
        "it",
        "this",
        "that",
        "with",
        "from",
        "not",
        "do",
        "does",
        "did",
        "if",
        "so",
        "no",
        "yes",
        "my",
        "your",
        "our",
        "their",
        "line",
        "row",
        "col",
        "item",
    }
)
ANCHOR_HINT_WORDS = frozenset(
    {
        "question",
        "problem",
        "exercise",
        "quiz",
        "homework",
        "hw",
        "slide",
        "page",
        "part",
        "section",
        "chapter",
        "lecture",
        "unit",
        "module",
        "topic",
        "task",
        "lab",
        "assignment",
        "q",
    }
)

TRANSCRIPT_MAX_CHARS = 1600
TRANSCRIPT_MAX_GAP_SEC = 15
CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\n(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)",
    re.S,
)
SCREEN_LABEL_RE = re.compile(
    r"(?im)^(?:Type|Top|Left|Right|Bottom|Content|Title|Bullet Points|Notes|Question)\s*:\s*"
)
SCREEN_BOILERPLATE_RE = re.compile(
    r"(?im)(?:reed\.cs\.depaul\.edu[^\n]*|to exit full screen[^\n]*|press Esc[^\n]*)\n?"
)
