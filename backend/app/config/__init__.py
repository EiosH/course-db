"""
App config package.

- config.models   — Ollama / embed / rerank / Qdrant / Langfuse
- config.mock     — MOCK_SESSION (enrollment + playback)
- config.prompts  — LLM system prompts
- this module     — paths, retrieval limits, regex helpers

`from config import X` still works via re-exports below.
"""

import re
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent.parent
# Load .env before models.py reads LANGFUSE_* / COURSE_ROUTE_MODEL.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(APP_DIR / ".env")  # optional local override

from config.mock import *  # noqa: E402,F401,F403
from config.models import *  # noqa: E402,F401,F403
from config.prompts import *  # noqa: E402,F401,F403

DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = DATA_DIR / "out"
LECTURES_DIR = DATA_DIR / "lectures"  # each subdir: optional doc.txt + transcript.vtt
QUERY_PATH = DATA_DIR / "query.txt"

TIME_WINDOW_SEC = 120  # 时间戳约束：±2 分钟

# 每路召回候选数 → rerank 后保留（略降以减轻 CrossEncoder 负担）
DENSE_LIMIT = 5
BM25_LIMIT = 5
TIME_NEAR_TOP_K = 4  # 时间类问题：最终保留条数（screen_shot / transcript 各半）
RERANK_TOP_K = 4
RERANK_MIN_SCORE = 0.0  # Qwen3-Reranker: yes/no logit diff，>0 表示相关

# 可选 resolve：问题里说不清的指称，小 top-k 召回后抽出具体名字（不 rerank）
PREPROBE_TOP_K = 3
PREPROBE_DENSE_LIMIT = 3
PREPROBE_BM25_LIMIT = 3
PREPROBE_MIN_CONFIDENCE = 0.35  # 低于此置信度的名字不当作可靠消解结果

# 多词定位短语（任意 "Label + 数字"）：收成单个 BM25 token，查询侧同步扩词。
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
ANCHOR_LIMIT = 8
ANCHOR_EXPAND_CHARS = 900

# transcript：合并相邻字幕，约 400 token；静音超过 15s 切新块
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
