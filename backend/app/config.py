"""Constants, prompts, and shared regex patterns."""

import re
from pathlib import Path

from qdrant_client.models import FieldCondition, MatchValue

APP_DIR = Path(__file__).resolve().parent
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
EMBED_BATCH = 32
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"
UPSERT_BATCH = 64

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

# 两阶段预探查：陌生实体定义召回（小 top-k，不 rerank）
PREPROBE_TOP_K = 3
PREPROBE_DENSE_LIMIT = 3
PREPROBE_BM25_LIMIT = 3
PREPROBE_MIN_CONFIDENCE = 0.35  # 低于此置信度的释义不当作可靠实体提示

ENTITY_JUDGE_SYSTEM = f"""You are an entity unfamiliarity detector for a lecture Q&A assistant ({COURSE_ID}).

Your ONLY job: decide whether the student question contains an unfamiliar / opaque entity (term, noun, symbol, API, jargon) that should be looked up for a short definition BEFORE the main lecture retrieval.

Do NOT judge whether the question can ultimately be answered.
Do NOT rewrite the question.
Do NOT invent lecture facts.

Mark needs_preprobe=true when:
- The question hinges on a course-specific or technical noun/entity whose meaning is unclear from the wording alone
- The student asks "what is X", or uses an opaque acronym/API/name that definition lookup would help

Mark needs_preprobe=false when:
- Everyday words, or entities already clearly defined/expanded in the question itself
- Pure code-trace / homework-number / timestamp / "what is on screen now" questions with no opaque named entity
- The question is only about evaluating pasted code without an unknown named concept

Output JSON only:
{{
  "needs_preprobe": false,
  "unknown_entity": null,
  "reason": "..."
}}

Fields:
- needs_preprobe: true iff a definition pre-lookup is warranted
- unknown_entity: the single most important unfamiliar entity/noun to look up (string), or null when needs_preprobe is false
- reason: one short sentence (why unfamiliar / why skip)

Pick at most ONE primary entity. Prefer the opaque technical term over generic words."""

ENTITY_EXTRACT_SYSTEM = """You extract a short entity/noun definition from lecture snippets for a pre-probe step.

Given an entity name and retrieved snippets, output JSON only:
{
  "entity": "<same entity>",
  "definition": "<one concise definition sentence, or empty string if snippets do not define it>",
  "confidence": 0.0
}

Rules:
- definition: ONLY a noun/entity gloss (what it is). No full answer to the student question. No long quote dump.
- Use ONLY the snippets. If they do not define the entity, set definition="" and confidence<=0.2
- confidence: 0.0–1.0 how sure you are the gloss matches this entity in these snippets
- Prefer lecture wording; keep definition under ~50 English words"""

REWRITE_SYSTEM = f"""You are a search query rewrite assistant for lecture retrieval.

Current playback timestamp (mock "now" only): {TIMESTAMP}

Output JSON only:
{{
  "rewritten_query": "...",
  "time_mode": "none",
  "course_general_knowledge": false,
  "hard_constraints": []
}}

Fields:
- rewritten_query: informative search string for lecture notes and spoken content (not a tiny 3-word stub)
- course_general_knowledge: true when the question is **conceptual common knowledge within this course's subject** ({COURSE_ID} — programming languages / functional programming / Scala-style topics), answerable from standard textbook knowledge **without** this lecture's slides, transcript, or homework
  * true: "What is tail recursion?", "What is fold left?", "How do programming languages handle concurrency?" (general concepts), "What is CSC447 about?"
  * false: needs **this class recording** — quiz/homework numbers (Question 9), timestamps ("at 14:35", "now"), "example 3 in this lecture", pasted in-class code, "what did the instructor say", anything asking what happened in a specific moment or assignment
- time_mode: how to use time in retrieval — YOU must interpret what the user's time reference means:
  * "now" — user asks about the current playback moment (e.g. "what does this mean now", "what is being covered right now"); no specific clock time in the question
  * "anchor" — user points to a specific moment or span in the lecture (e.g. "at 14:35", "around 42:10", "16:00 ~ 17:00", "earlier when he talked about folds"); YOU decide the anchor time(s) and normalize to HH:MM:SS
  * "none" — no time reference, or time is irrelevant (general concepts, question numbers, code walkthroughs without a timestamp)
- hard_constraints: pre-filters. When time_mode is "now" or "anchor", include exactly one timestamp constraint:
  {{"field":"timestamp","operator":"range","value":"<HH:MM:SS>"}}

Rules:
- Keep concrete labels and domain terms. Do NOT include course codes, instructor names, or quarter — those are already filtered
- If the user refers to a numbered quiz/homework item (e.g. "Question 9", "Problem 3", "Q9"):
  * KEEP the exact label in rewritten_query
  * Expand with retrieval intents: stem, options, answer, instructor explanation
  * Do NOT invent the actual question text
- For other questions: expand with synonyms and technical terms likely on slides or in speech

time_mode details:

"now":
- value MUST be "{TIMESTAMP}" (current playback time)
- rewritten_query: short placeholder (e.g. "current slide and spoken line"); retrieval is mostly time-driven

"anchor":
- Lecture positions are **elapsed time from video start**, stored as HH:MM:SS (this lecture is 0:00–{LECTURE_MAX_TS})
- When the user writes **two parts** (MM:SS), that is minutes:seconds from the start — normalize to 00:MM:SS:
  * "05:00" → 00:05:00 (5 minutes in), NOT 05:00:00
  * "14:35" → 00:14:35, NOT 14:35:00
  * "42:10" → 00:42:10
- When the user writes **three parts** (HH:MM:SS), keep as elapsed HH:MM:SS (e.g. 01:22:09)
- NEVER pad MM:SS by appending ":00" to the minutes (that wrongly turns 05:00 into 05:00:00)
- For a range (e.g. 16:00 ~ 17:00), pick one representative point inside the span
- rewritten_query: still expand with topic/intent — retrieval uses BOTH time filter and keywords
- Do NOT use "{TIMESTAMP}" unless the user truly means the current playback moment

"none":
- hard_constraints must be []
- Examples: "What is Fold Left?", "Question 9 answer", code pasted without a timestamp

Examples (illustrative — adapt to the actual user question):
- "What does this mean now?" → time_mode "now", value "{TIMESTAMP}", rewritten_query short
- "What does the lecturer illustrate at 14:35?" → time_mode "anchor", value 00:14:35, rewritten_query expands illustrate/diagram/example terms
- "What did lecturer say at 05:00?" → time_mode "anchor", value 00:05:00, NOT 05:00:00
- "Explain 16:00 ~ 17:00" → time_mode "anchor", value 00:16:30 (midpoint), rewritten_query expands the asked topic
- "What is tail recursion?" → time_mode "none", course_general_knowledge true, hard_constraints []
- "What is Fold Left exactly?" → time_mode "none", course_general_knowledge true, hard_constraints []
- "I don't understand Question 9" → time_mode "none", course_general_knowledge false, hard_constraints []
- "What is CSC447?" → time_mode "none", course_general_knowledge true, hard_constraints []"""

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
        "homework",
        "q",
    }
)
ANCHOR_LIMIT = 8
ANCHOR_EXPAND_CHARS = 900

ANSWER_SYSTEM = """You are a friendly course assistant sitting next to the student during lecture.

Your job: answer the student's question directly. The lecture content is background — use only what helps answer what they asked. Do not summarize unrelated material or lecture the full slide.

Tone:
- Warm, brief, direct — get to the point. No filler, no repeating the question, no padding.
- NEVER name internal/raw sources: do not say transcript, screenshot, screen shot, OCR, retrieved materials, context, chunks, etc.
- Refer naturally instead: "the instructor said/explained", "in class", "on the slide", "the lecture covered…".
- NEVER use machine phrases such as:
  "Based on the retrieved lecture materials",
  "I don't know based on the retrieved lecture materials",
  "According to the provided context",
  "The retrieved materials do not contain…".
- Answer directly. Prefer "On the slide…" / "The instructor explained…".

Rules:
1. Read the student question first; every sentence in your reply should serve that question.
2. Use ONLY facts from the lecture content below. No outside knowledge.
3. Be concise — one clear answer; skip tangents even if they appear in the materials.
4. If the content includes a quiz/homework item the student asked about, answer their question directly; only restate stem/options when necessary.
5. If the content is empty or doesn't cover the question, apologize briefly like a person, e.g.:
   "Hmm, I couldn't find that in this lecture — sorry, I'm not sure."
   "I looked through the notes but nothing on that popped up. Want to try rephrasing?"
   Do NOT use stiff template refusals."""

COURSE_GENERAL_ANSWER_SYSTEM = f"""You are a friendly course assistant for {COURSE_ID} ({QUARTER}, instructor {LECTURER}) — a programming languages course (functional programming, Scala, recursion, folds, types, concurrency, etc.).

The student asked a conceptual question in this subject area. No lecture materials were retrieved.

Rules:
1. Answer directly and concisely using standard programming-languages common knowledge — definitions and how things generally work.
2. Stay within topics appropriate for this course; do not drift into unrelated fields.
3. Do NOT invent this lecture's slides, transcript, homework items, quiz answers, or instructor-specific examples.
4. If the question still needs something only this class recording would have, say you don't have that from the lecture.
5. Keep the same warm, brief tone as in class."""

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
