"""Query plan, rewrite, and answer generation."""

import json
import re

from config import (
    ANSWER_SYSTEM,
    ANSWER_TEMPERATURE,
    NO_HIT_NOW_REPLY,
    NO_HIT_REPLY,
    PREPROBE_MIN_CONFIDENCE,
    RESOLVE_EXTRACT_SYSTEM,
    STIFF_REFUSAL_RE,
    STIFF_REFUSAL_REPLY,
    knowledge_plan_system,
    query_plan_system,
    resolve_plan_system,
    rewrite_system,
    time_plan_system,
)
from llm import ollama_chat
from time_utils import normalize_rewrite_timestamps


def _parse_json_content(raw: str) -> dict:
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def plan_resolve(plan: dict | None) -> str | None:
    """Vague phrase to resolve first, or None to skip preprobe."""
    if not plan:
        return None
    raw = plan.get("resolve")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def plan_time(query: str, course_ctx: dict) -> dict:
    """MAIN retrieval time_mode + timestamp hard_constraints."""
    raw = ollama_chat(
        [
            {"role": "system", "content": time_plan_system(course_ctx)},
            {"role": "user", "content": query},
        ],
        format="json",
        name="llm.plan_time",
    )
    data = _parse_json_content(raw)
    planned = normalize_rewrite_timestamps(
        {
            "time_mode": data.get("time_mode", "none"),
            "hard_constraints": data.get("hard_constraints", []),
        },
        lecture_max_ts=course_ctx.get("lecture_max_ts"),
    )
    return {
        "time_mode": planned.get("time_mode", "none"),
        "hard_constraints": planned.get("hard_constraints", []),
        "reason": str(data.get("reason") or "").strip(),
    }


def plan_resolve_part(query: str, course_ctx: dict) -> dict:
    """resolve phrase + probe_hard_constraints (independent of main time)."""
    raw = ollama_chat(
        [
            {"role": "system", "content": resolve_plan_system(course_ctx)},
            {"role": "user", "content": query},
        ],
        format="json",
        name="llm.plan_resolve",
    )
    data = _parse_json_content(raw)
    resolve = data.get("resolve")
    if isinstance(resolve, str):
        resolve = resolve.strip() or None
    else:
        resolve = None
    probe_planned = normalize_rewrite_timestamps(
        {"hard_constraints": data.get("probe_hard_constraints", [])},
        lecture_max_ts=course_ctx.get("lecture_max_ts"),
    )
    probe_constraints = list(probe_planned.get("hard_constraints") or [])
    if not resolve:
        probe_constraints = []
    return {
        "resolve": resolve,
        "probe_hard_constraints": probe_constraints,
        "reason": str(data.get("reason") or "").strip(),
    }


def plan_knowledge(query: str, course_ctx: dict) -> dict:
    """course_general_knowledge flag."""
    raw = ollama_chat(
        [
            {"role": "system", "content": knowledge_plan_system(course_ctx)},
            {"role": "user", "content": query},
        ],
        format="json",
        name="llm.plan_knowledge",
    )
    data = _parse_json_content(raw)
    return {
        "course_general_knowledge": bool(data.get("course_general_knowledge")),
        "reason": str(data.get("reason") or "").strip(),
    }


def assemble_plan(
    *,
    time_part: dict,
    resolve_part: dict,
    knowledge_part: dict,
) -> dict:
    """Merge parallel plan siblings into one plan dict (before course constraints)."""
    reasons = [
        p.get("reason")
        for p in (time_part, resolve_part, knowledge_part)
        if p.get("reason")
    ]
    return {
        "time_mode": time_part.get("time_mode", "none"),
        "hard_constraints": list(time_part.get("hard_constraints") or []),
        "probe_hard_constraints": list(resolve_part.get("probe_hard_constraints") or []),
        "course_general_knowledge": bool(
            knowledge_part.get("course_general_knowledge")
        ),
        "resolve": resolve_part.get("resolve"),
        "reason": " | ".join(reasons),
    }


def plan_query(query: str, course_ctx: dict) -> dict:
    """Legacy monolithic plan (unused by parallel pipeline; kept for scripts)."""
    raw = ollama_chat(
        [
            {"role": "system", "content": query_plan_system(course_ctx)},
            {"role": "user", "content": query},
        ],
        format="json",
        name="llm.plan",
    )
    data = _parse_json_content(raw)
    resolve = data.get("resolve")
    if isinstance(resolve, str):
        resolve = resolve.strip() or None
    else:
        resolve = None

    max_ts = course_ctx.get("lecture_max_ts")
    planned = normalize_rewrite_timestamps(
        {
            "time_mode": data.get("time_mode", "none"),
            "hard_constraints": data.get("hard_constraints", []),
        },
        lecture_max_ts=max_ts,
    )
    probe_planned = normalize_rewrite_timestamps(
        {"hard_constraints": data.get("probe_hard_constraints", [])},
        lecture_max_ts=max_ts,
    )
    probe_constraints = list(probe_planned.get("hard_constraints") or [])
    if not resolve:
        probe_constraints = []

    return {
        "time_mode": planned.get("time_mode", "none"),
        "hard_constraints": planned.get("hard_constraints", []),
        "probe_hard_constraints": probe_constraints,
        "course_general_knowledge": bool(data.get("course_general_knowledge")),
        "resolve": resolve,
        "reason": str(data.get("reason") or "").strip(),
    }


def extract_resolved_name(phrase: str, hits) -> dict:
    empty = {
        "phrase": phrase,
        "name": "",
        "confidence": 0.0,
        "found": False,
    }
    if not hits:
        return empty
    snippets = []
    for h in hits:
        p = h.payload
        label = "slide" if p.get("type") == "screen_shot" else "speech"
        snippets.append(f"[{label}] ({p.get('timestamp', '')})\n{p.get('text', '')}")
    raw = ollama_chat(
        [
            {"role": "system", "content": RESOLVE_EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Phrase: {phrase}\n\nRetrieved snippets:\n"
                    + "\n\n---\n\n".join(snippets)
                ),
            },
        ],
        format="json",
        name="llm.resolve",
    )
    data = _parse_json_content(raw)
    name = str(data.get("name") or "").strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    same = bool(name) and name.lower() == phrase.lower()
    found = bool(name) and not same and confidence >= PREPROBE_MIN_CONFIDENCE
    return {
        "phrase": phrase,
        "name": name if found else "",
        "confidence": confidence,
        "found": found,
    }


def _rewrite_user_message(
    query: str,
    plan: dict,
    resolved_name: str | None = None,
) -> str:
    parts = [
        f"Student question:\n{query}",
        "Query plan (time already fixed — do not change it):\n"
        + json.dumps(
            {
                "time_mode": plan.get("time_mode"),
                "hard_constraints": plan.get("hard_constraints", []),
                "probe_hard_constraints": plan.get("probe_hard_constraints", []),
                "course_general_knowledge": bool(plan.get("course_general_knowledge")),
                "resolve": plan.get("resolve"),
            },
            ensure_ascii=False,
            indent=2,
        ),
    ]
    if resolved_name:
        parts.append(f"Resolved name for {plan.get('resolve')!r}: {resolved_name}")
    return "\n\n".join(parts)


def rewrite(
    query: str,
    plan: dict,
    course_ctx: dict,
    resolved_name: str | None = None,
) -> dict:
    """Build rewritten_query. Time / course_general_knowledge come from plan."""
    raw = ollama_chat(
        [
            {"role": "system", "content": rewrite_system(course_ctx)},
            {
                "role": "user",
                "content": _rewrite_user_message(query, plan, resolved_name),
            },
        ],
        format="json",
        name="llm.rewrite",
    )
    data = _parse_json_content(raw)
    rewritten_query = str(data.get("rewritten_query") or "").strip() or query
    return {
        "rewritten_query": rewritten_query,
        "time_mode": plan.get("time_mode", "none"),
        "course_general_knowledge": bool(plan.get("course_general_knowledge")),
        "hard_constraints": list(plan.get("hard_constraints") or []),
    }


def build_prompt(
    question,
    hits,
    *,
    course_general: bool = False,
):
    parts = [f"Student question: {question}"]
    if course_general:
        parts.append(
            "Course subject knowledge is allowed. Lecture snippets below are "
            "optional support — solve the question even if they are missing "
            "or only loosely related. Do not invent this lecture's slides, "
            "quotes, or homework items."
        )

    parts.append("Lecture content:")
    if not hits:
        parts.append("(nothing found)")
    else:
        for h in hits:
            p = h.payload
            label = "slide" if p.get("type") == "screen_shot" else "speech"
            parts.append(f"[{label}] ({p.get('timestamp', '')})\n{p.get('text', '')}")
    return "\n\n".join(parts)


def answer(prompt: str) -> str:
    raw = ollama_chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=ANSWER_TEMPERATURE,
        name="llm.answer",
    )
    text = (raw or "").strip()
    if STIFF_REFUSAL_RE.match(text):
        return STIFF_REFUSAL_REPLY
    return text


def no_hit_reply(*, now: bool = False) -> str:
    return NO_HIT_NOW_REPLY if now else NO_HIT_REPLY


def format_hits(hits, channel: str = "") -> str:
    if not hits:
        return ""
    blocks = []
    for h in hits:
        p = h.payload
        score = f"{h.score:.4f}" if h.score is not None else ""
        tag = f" channel={channel}" if channel else ""
        rerank = p.get("rerank_score")
        extra = f" rerank={rerank:.4f}" if rerank is not None else ""
        td = p.get("time_distance")
        if td is not None:
            extra += f" dt={td:.1f}s"
        blocks.append(f"[{p['timestamp']}] score={score}{tag}{extra}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)
