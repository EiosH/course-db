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

# Unified query plan: time + what to resolve, in ONE LLM call.
QUERY_PLAN_SYSTEM = f"""You analyze a student question for a lecture Q&A retrieval system ({COURSE_ID}).

Produce a structured plan. Do NOT answer the student. Do NOT invent lecture facts, homework items, quiz numbers, or topic labels that are not licensed by the question text.

Current playback timestamp (mock "now" only): {TIMESTAMP}
Lecture elapsed time is 0:00–{LECTURE_MAX_TS}, stored as HH:MM:SS.

Output JSON only:
{{
  "time_mode": "none",
  "hard_constraints": [],
  "time_preprobe_only": false,
  "referents": [],
  "entities": [],
  "preprobe": null,
  "reason": "..."
}}

Fields:
- time_mode:
  * "now" — current playback moment with no clock time in the question (e.g. "what does this mean now")
  * "anchor" — a specific elapsed time / span in the lecture (e.g. "at 14:35", "around 42:10", "2h24min", "16:00 ~ 17:00")
  * "none" — no usable time reference
- hard_constraints: when time_mode is "now" or "anchor", exactly one
  {{"field":"timestamp","operator":"range","value":"<HH:MM:SS>"}}
  * "now" → value MUST be "{TIMESTAMP}"
  * Lecture positions are **elapsed time from video start** (0:00–{LECTURE_MAX_TS})
  * When the user writes **two parts** (MM:SS), that is minutes:seconds from the start — normalize to 00:MM:SS:
    - "05:00" → 00:05:00 (5 minutes in), NOT 05:00:00
    - "14:35" → 00:14:35, NOT 14:35:00
    - "42:10" → 00:42:10
  * When the user writes **three parts** (HH:MM:SS), keep as elapsed HH:MM:SS (e.g. 01:22:09)
  * NEVER pad MM:SS by appending ":00" to the minutes (that wrongly turns 05:00 into 05:00:00)
  * "2h24min" / "2h24" → 02:24:00; prefer hours when the user wrote "h"
  * For a range (e.g. 16:00 ~ 17:00), pick one representative point inside the span (e.g. 00:16:30)
  * Do NOT use "{TIMESTAMP}" unless the user truly means the current playback moment
- time_preprobe_only: true when the timestamp is mainly to resolve a local referent/example, but the student ALSO asks for related material that may appear elsewhere (e.g. find the example at 2h24, then locate questions about that concept). false when the whole answer should stay near that timestamp.
- referents: underspecified phrases that need context to resolve — e.g. "the example", "this concept", "that idea", "it". Include EVEN IF a timestamp is also present.
- entities: clearly named jargon/terms that may need a short definition. Empty if none.
- preprobe: the single string to look up first (prefer a referent, else an entity), or null to skip pre-probe. Non-null ⇒ pre-probe runs.
- reason: one short sentence

Critical consistency:
- A question can have BOTH a time anchor AND referents. Then preprobe MUST be set (not null).
- Do NOT leave preprobe null merely because a timestamp exists.
- For "example around TIME + questions about this concept", prefer time_preprobe_only=true.

Illustrative examples (adapt to the actual question):
- "What does this mean now?" → time_mode "now", value "{TIMESTAMP}", referents ["this"], preprobe "this", time_preprobe_only false
- "What does the lecturer illustrate at 14:35?" → time_mode "anchor", value 00:14:35, preprobe null
- "What did lecturer say at 05:00?" → time_mode "anchor", value 00:05:00, NOT 05:00:00
- "Explain 16:00 ~ 17:00" → time_mode "anchor", value 00:16:30 (midpoint)
- "Could you find the example around 2h24min, and locate the questions about this concept?" → time_mode "anchor", value 02:24:00, referents ["the example","this concept"], preprobe "the example" (or "this concept"), time_preprobe_only true
- "What is tail recursion?" → time_mode "none", entities ["tail recursion"], preprobe "tail recursion" if opaque lookup helps
- "I don't understand Question 9" → time_mode "none", preprobe null (concrete labeled item)
- "Is there a simpler example for this concept?" → time_mode "none", referents ["this concept"], preprobe "this concept"
"""

ENTITY_EXTRACT_SYSTEM = """You extract a short entity/noun definition from lecture snippets for a pre-probe step.

Given an entity name (or a vague deictic phrase like "this concept" / "the example") and retrieved snippets, output JSON only:
{
  "entity": "<resolved concrete entity/noun if deixis, otherwise the same entity>",
  "definition": "<one concise definition sentence, or empty string if snippets do not define it>",
  "confidence": 0.0
}

Rules:
- definition: ONLY a noun/entity gloss (what it is). No full answer to the student question. No long quote dump.
- If the input is a vague referent ("this concept", "the example", "that", "it"), resolve it to the concrete concept/example named in the snippets when possible, and put that concrete name in "entity".
- Use ONLY the snippets. If they do not define / resolve the entity, set definition="" and confidence<=0.2
- confidence: 0.0–1.0 how sure you are the gloss matches this entity in these snippets
- Prefer lecture wording; keep definition under ~50 English words"""

REWRITE_SYSTEM = f"""You write a retrieval query string for lecture notes / transcript search ({COURSE_ID}).

You receive:
1) the original student question
2) an optional query plan (time already decided elsewhere — do NOT change time)
3) optional resolved entity glosses from a prior pre-probe

Output JSON only:
{{
  "rewritten_query": "...",
  "course_general_knowledge": false
}}

Fields:
- rewritten_query: a grounded search string for slides and spoken content (not a tiny 3-word stub)
- course_general_knowledge: true when the question is **conceptual common knowledge within this course's subject** ({COURSE_ID} — programming languages / functional programming / Scala-style topics), answerable from standard textbook knowledge **without** this lecture's slides, transcript, or homework
  * true: "What is tail recursion?", "What is fold left?", "How do programming languages handle concurrency?", "What is CSC447 about?"
  * false: needs **this class recording** — quiz/homework numbers (Question 9), timestamps, "example in this lecture", pasted in-class code, "what did the instructor say"

Grounding (architectural constraint — not optional):
- rewritten_query MUST be licensed by (a) words/phrases in the student question, and/or (b) resolved gloss entities/definitions provided below.
- You may rephrase, lemmatize, or lightly add close paraphrases of those licensed terms.
- Do NOT introduce new assignment genres or inventory labels the student did not say and glosses do not mention — e.g. do not add homework / quiz / assignment / problem / exam unless those words (or the gloss) already license them.
- If the student says "questions about this concept", keep that intent; do not upgrade it into homework/assignment language.
- If glosses resolved a vague referent, you MAY include the resolved concrete name.
- Prefer staying close to the student's wording over aggressive synonym stuffing.
- Keep concrete labels (e.g. Question 9). Do NOT invent the actual question text.
- If the user refers to a numbered quiz/homework item already present in the question (e.g. "Question 9", "Problem 3", "Q9"):
  * KEEP the exact label in rewritten_query
  * You may add retrieval intents licensed by that label: stem, options, answer, instructor explanation

Illustrative examples (adapt; stay grounded):
- "What does this mean now?" + plan time_mode now → short placeholder about current slide / spoken line
- "What does the lecturer illustrate at 14:35?" → expand illustrate / diagram / example terms; do not invent homework
- "example around 2h24min ... questions about this concept" + gloss resolves concept → include example / resolved concept name / questions; do NOT invent "homework assignment problem"
- "I don't understand Question 9" → keep "Question 9"; may add stem / options / answer / explanation
- "What is tail recursion?" → conceptual terms; course_general_knowledge true
- "What is Fold Left exactly?" → course_general_knowledge true
- "What is CSC447?" → course_general_knowledge true"""

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
