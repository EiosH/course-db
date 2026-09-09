"""Query rewrite and answer generation."""

import json
import re

from config import (
    ANSWER_SYSTEM,
    COURSE_GENERAL_ANSWER_SYSTEM,
    ENTITY_EXTRACT_SYSTEM,
    ENTITY_JUDGE_SYSTEM,
    NO_HIT_NOW_REPLY,
    NO_HIT_REPLY,
    PREPROBE_MIN_CONFIDENCE,
    REWRITE_SYSTEM,
    STIFF_REFUSAL_RE,
    STIFF_REFUSAL_REPLY,
)
from llm import ollama_chat
from time_utils import normalize_rewrite_timestamps


def _parse_json_content(raw: str) -> dict:
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def judge_unknown_entity(query: str) -> dict:
    """LLM: only whether an unfamiliar entity needs pre-probe (not answerability)."""
    raw = ollama_chat(
        [
            {"role": "system", "content": ENTITY_JUDGE_SYSTEM},
            {"role": "user", "content": query},
        ],
        format="json",
    )
    data = _parse_json_content(raw)
    needs = bool(data.get("needs_preprobe", False))
    entity = data.get("unknown_entity")
    if isinstance(entity, str):
        entity = entity.strip() or None
    else:
        entity = None
    if needs and not entity:
        needs = False
    return {
        "needs_preprobe": needs,
        "unknown_entity": entity if needs else None,
        "reason": str(data.get("reason") or "").strip(),
    }


def extract_entity_gloss(entity: str, hits) -> dict:
    """From pre-probe hits, extract a short noun definition + confidence."""
    if not hits:
        return {
            "entity": entity,
            "definition": "",
            "confidence": 0.0,
            "found": False,
        }
    snippets = []
    for h in hits:
        p = h.payload
        label = "slide" if p.get("type") == "screen_shot" else "speech"
        snippets.append(f"[{label}] ({p.get('timestamp', '')})\n{p.get('text', '')}")
    user = (
        f"Entity: {entity}\n\n"
        "Retrieved snippets:\n"
        + "\n\n---\n\n".join(snippets)
    )
    raw = ollama_chat(
        [
            {"role": "system", "content": ENTITY_EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ],
        format="json",
    )
    data = _parse_json_content(raw)
    definition = str(data.get("definition") or "").strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    found = bool(definition) and confidence >= PREPROBE_MIN_CONFIDENCE
    return {
        "entity": entity,
        "definition": definition,
        "confidence": confidence,
        "found": found,
    }


def _rewrite_user_message(query: str, entity_hints: list[dict] | None = None) -> str:
    """Original student query + refined entity glosses (never dump full pre-probe text)."""
    parts = [f"Student question:\n{query}"]
    hints = [
        h
        for h in (entity_hints or [])
        if h.get("found") and h.get("entity") and h.get("definition")
    ]
    if hints:
        lines = [
            "Entity glosses from a prior definition lookup "
            "(clarify terms only; do NOT replace the student question; "
            "still ground rewritten_query on the original question):"
        ]
        for h in hints:
            conf = h.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            lines.append(f"- {h['entity']} (confidence={conf_s}): {h['definition']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def rewrite(query: str, *, entity_hints: list[dict] | None = None) -> dict:
    raw = ollama_chat(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": _rewrite_user_message(query, entity_hints)},
        ],
        format="json",
    )
    data = _parse_json_content(raw)
    return normalize_rewrite_timestamps(
        {
            "rewritten_query": data["rewritten_query"],
            "time_mode": data.get("time_mode", "none"),
            "course_general_knowledge": bool(
                data.get("course_general_knowledge", False)
            ),
            "hard_constraints": data.get("hard_constraints", []),
        }
    )


def build_prompt(question, hits, *, preprobe: dict | None = None):
    parts = [f"Student question: {question}"]

    if preprobe and preprobe.get("needs_preprobe"):
        entity = preprobe.get("unknown_entity") or ""
        gloss = preprobe.get("gloss") or {}
        if gloss.get("found") and gloss.get("definition"):
            conf = gloss.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            parts.append(
                "Entity glossary (from pre-probe; use only as term definition):\n"
                f"- {gloss.get('entity') or entity} "
                f"(confidence={conf_s}): {gloss['definition']}"
            )
        else:
            note = "No related entity definition was found in the lecture materials"
            if entity:
                note += f" for '{entity}'"
            parts.append(
                "Entity pre-probe note: "
                f"{note}. Answer using lecture content below; do not invent a definition."
            )

    parts.append(
        "Lecture content (use only what is needed to answer the question above):"
    )
    if not hits:
        parts.append("(nothing found)")
    else:
        for h in hits:
            p = h.payload
            label = "On slide" if p["type"] == "screen_shot" else "Instructor said"
            parts.append(f"[{label}] ({p['timestamp']})\n{p['text']}")
    return "\n\n".join(parts)


def soften_stiff_refusal(text: str) -> str:
    """Replace machine-style refusals with a natural assistant reply."""
    if STIFF_REFUSAL_RE.match(text.strip()):
        return STIFF_REFUSAL_REPLY
    # 句中仍夹带那句机器话术时，整句换成助手语气
    bad = (
        "i don't know based on the retrieved lecture materials",
        "based on the retrieved lecture materials",
    )
    low = text.lower()
    if any(p in low for p in bad) and ("don't know" in low or "do not know" in low):
        return STIFF_REFUSAL_REPLY
    return text


def answer(prompt: str, *, system: str = ANSWER_SYSTEM) -> str:
    raw = ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
    return soften_stiff_refusal(raw)


def no_hit_answer(query: str, rewritten: dict, *, now: bool = False) -> str:
    """No retrieval hits: answer subject common knowledge only; else fixed refusal."""
    if rewritten.get("course_general_knowledge"):
        prompt = f"Student question: {query}"
        return answer(prompt, system=COURSE_GENERAL_ANSWER_SYSTEM)
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
