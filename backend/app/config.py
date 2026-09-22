"""Constants, prompts, and shared regex patterns."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.models import FieldCondition, MatchValue

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
# Load repo-root .env so LANGFUSE_* work without manual export.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(APP_DIR / ".env")  # optional local override

DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = DATA_DIR / "out"
DOC_PATH = DATA_DIR / "doc.txt"
TRANSCRIPT_PATH = DATA_DIR / "transcript.vtt"
QUERY_PATH = DATA_DIR / "query.txt"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3.8:27b"
EMBED_MODEL = "bge-m3"
BM25_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_CHAT_RETRIES = 3
# Plan / resolve-extract / rewrite stay greedy for stable JSON and retrieval.
OLLAMA_TEMPERATURE = 0.0
ANSWER_TEMPERATURE = 0.2  # answer() only: slightly warmer tone
OLLAMA_SEED = 42
EMBED_BATCH = 32
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"
UPSERT_BATCH = 64

# Langfuse: keys / base URL from .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# LANGFUSE_BASE_URL). Set LANGFUSE_ENABLED=true to turn tracing on.
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "").lower() in ("1", "true", "yes")

# mock 预过滤
COURSE_ID = "CSC447"
QUARTER = "2026-Spring"
LECTURER = "Eric J. Fredericks"
TIMESTAMP = "01:22:09"  # mock 当前播放位置，仅时间类问题启用
LECTURE_MAX_TS = "03:30:00"  # 本课视频最长时长，用于校正 anchor 时间格式
TIME_WINDOW_SEC = 120  # 时间戳约束：±2 分钟，优先当前画面/台词
BASE_MUST = [
    FieldCondition(key="course_id", match=MatchValue(value=COURSE_ID)),
    FieldCondition(key="quarter", match=MatchValue(value=QUARTER)),
    FieldCondition(key="lecturer", match=MatchValue(value=LECTURER)),
]

# 每路召回候选数 → rerank 后保留
DENSE_LIMIT = 10
BM25_LIMIT = 10
TIME_NEAR_TOP_K = 4  # 时间类问题：最终保留条数（screen_shot / transcript 各半）
RERANK_TOP_K = 4
RERANK_MIN_SCORE = 0.0  # Qwen3-Reranker: yes/no logit diff，>0 表示相关

# 可选 resolve：问题里说不清的指称，小 top-k 召回后抽出具体名字（不 rerank）
PREPROBE_TOP_K = 3
PREPROBE_DENSE_LIMIT = 3
PREPROBE_BM25_LIMIT = 3
PREPROBE_MIN_CONFIDENCE = 0.35  # 低于此置信度的名字不当作可靠消解结果

# Unified query plan: time + optional resolve, in ONE LLM call.
QUERY_PLAN_SYSTEM = f"""You analyze a student question for a lecture Q&A retrieval system ({COURSE_ID}).

Produce a structured plan. Do NOT answer the student. Do NOT invent lecture facts, homework items, quiz numbers, or topic labels that are not in the question text.

Current playback timestamp (mock "now" only): {TIMESTAMP}
Lecture elapsed time is 0:00–{LECTURE_MAX_TS}, stored as HH:MM:SS.

Output JSON only:
{{
  "time_mode": "none",
  "hard_constraints": [],
  "time_preprobe_only": false,
  "course_general_knowledge": false,
  "resolve": null,
  "reason": "..."
}}

Fields:
- time_mode:
  * "now" — current playback moment, no clock time in the question
  * "anchor" — a specific elapsed time / span in the lecture
  * "none" — no usable time reference
- hard_constraints: when time_mode is "now" or "anchor", exactly one
  {{"field":"timestamp","operator":"range","value":"<HH:MM:SS>"}}
  * "now" → value MUST be "{TIMESTAMP}"
  * Lecture positions are elapsed time from video start (0:00–{LECTURE_MAX_TS})
  * Two parts (MM:SS) = minutes:seconds from the start → 00:MM:SS
    ("05:00" → 00:05:00, never 05:00:00)
  * Three parts (HH:MM:SS) stay as elapsed HH:MM:SS
  * "2h24min" / "2h24" → 02:24:00
  * For a range, pick one point inside the span
  * Do NOT use "{TIMESTAMP}" unless the user means the current playback moment
- resolve: copy the underspecified phrase from the question that must be identified first (e.g. "the example", "this", "it"), or null. Named terms already written in the question (jargon, "Fold Left", "Question 9") stay null — main retrieval handles them. Non-null ⇒ a small resolve search runs before rewrite.
- time_preprobe_only: true only when resolve is set AND the timestamp is solely to locate that phrase (e.g. look up "the example" at 2h24), after which the student wants related material from the rest of the lecture. Then the timestamp filters ONLY the resolve search; main retrieval has no time filter. false when the whole answer should stay near that timestamp, or when resolve is null.
- course_general_knowledge: true when the question can be answered from this course's subject knowledge without this recording (a definition, or a problem fully stated in the question). false when the answer depends on this lecture (what was said, this slide, a numbered item whose stem is not in the question, timestamps).
- reason: one short sentence

Critical consistency:
- resolve is for deixis / vague pointing, not for defining named terms.
- A timestamp plus a vague phrase ("the example at 2h24") ⇒ set resolve to that phrase. Do NOT leave resolve null merely because a timestamp exists.
- "What does this mean now?" with no other vague object: time_mode "now", resolve null (the current window is the context).
- If the question already states the full problem and does not refer to this lecture: resolve null, course_general_knowledge true.
"""

RESOLVE_EXTRACT_SYSTEM = """You resolve a vague phrase to a concrete name using lecture snippets.

Given a phrase from the student question (e.g. "the example", "this") and retrieved snippets, output JSON only:
{
  "name": "<concrete name, or empty string if unresolved>",
  "confidence": 0.0
}

Rules:
- name: a short noun phrase for what the phrase refers to (the example's title, the concept's name). NOT a definition sentence. NOT an answer to the student question. NOT a quote dump.
- Use ONLY the snippets. If they do not resolve the phrase, set name="" and confidence<=0.2
- confidence: 0.0–1.0 how sure you are this is the right referent
- Prefer lecture wording; keep name under ~12 English words
"""

REWRITE_SYSTEM = f"""You write a retrieval query string for lecture notes / transcript search ({COURSE_ID}).

You receive:
1) the original student question
2) a query plan (time already decided — do NOT change it)
3) an optional resolved name from a prior resolve search (a concrete name for a vague phrase)

Output JSON only:
{{
  "rewritten_query": "..."
}}

- rewritten_query: a grounded search string for slides and spoken content (not a tiny stub)
- Licensed only by the student question and the optional resolved name. Rephrase freely; do not invent assignment/homework/quiz labels the student did not say.
- Keep concrete labels that already appear in the question (e.g. a numbered item).
- If a resolved name is provided, you MAY include that name. Do not dump lecture snippets.
"""

# 多词定位短语（任意 "Label + 数字"）：收成单个 BM25 token，查询侧同步扩词。
# token = "ph" + 去空白标点(label+num)，由规则动态生成（不是写死某个题号）。
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
        "item",  # 太泛；真正的 Item 12 若需要可再放回 HINT
    }
)
# 常见定位词；其它非停用词 Label+数字也可（如 Lab 3 / Assignment 2）
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

ANSWER_SYSTEM = """You are a friendly course assistant sitting next to the student during lecture.

Answer the student's question directly. Lecture content is background — use only what helps. Do not summarize unrelated material.

Tone:
- Warm, brief, direct. No filler, no repeating the question.
- Never name internal sources (transcript, screenshot, OCR, retrieved materials, chunks).
- Refer naturally: "the instructor said", "on the slide", "in class".

Rules:
1. Every sentence should serve the student question.
2. Prefer facts from the lecture content below.
3. If the prompt says course subject knowledge is allowed, you may use it to finish the answer when lecture snippets are missing or only loosely related. Do not invent this lecture's slides, quotes, or homework items.
4. If course subject knowledge is not allowed and the lecture content does not cover the question, apologize briefly like a person.
5. Be concise."""

# 无检索结果时不交给模型套模板，直接用人话回复
NO_HIT_REPLY = (
    "Hmm, I couldn't find anything useful in the lecture notes for that — "
    "sorry, I'm not sure. Want to try asking another way?"
)
NO_HIT_NOW_REPLY = (
    "I don't have anything from this moment in the lecture — sorry! "
    "Maybe scrub a bit or ask about a specific term on the slide."
)

# 模型仍可能吐出的生硬拒答 → 替换成助手语气
STIFF_REFUSAL_RE = re.compile(
    r"(?is)^\s*(?:based on (?:the )?(?:retrieved )?lecture materials[,.]?\s*)?"
    r"i don'?t know(?: based on the retrieved lecture materials)?\.?\s*$"
)
STIFF_REFUSAL_REPLY = (
    "Hmm, I couldn't find that in this lecture — sorry, I'm not sure. "
    "Want to try rephrasing?"
)

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
