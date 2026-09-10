"""Query plan, rewrite, and answer generation."""

import json
import re

from config import (
    ANSWER_SYSTEM,
    COURSE_GENERAL_ANSWER_SYSTEM,
    ENTITY_EXTRACT_SYSTEM,
    NO_HIT_NOW_REPLY,
    NO_HIT_REPLY,
    PREPROBE_MIN_CONFIDENCE,
    QUERY_PLAN_SYSTEM,
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


def _as_str_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def plan_preprobe(plan: dict | None) -> str | None:
    """Target string to pre-probe, or None to skip. Derived: bool(this) ⇒ run preprobe."""
    if not plan:
        return None
    raw = plan.get("preprobe")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def plan_preprobe_is_referent(plan: dict | None) -> bool:
    """True when preprobe target comes from referents (not named entities)."""
    target = plan_preprobe(plan)
    if not target or not plan:
        return False
    return any(
        target.lower() == r.lower() for r in (plan.get("referents") or [])
    )


def plan_query(query: str) -> dict:
    """
    Single structured plan: time + referents/entities + optional preprobe string.
    needs_preprobe / is_referent are derived from `preprobe` + `referents` — not stored.
    """
    raw = ollama_chat(
        [
            {"role": "system", "content": QUERY_PLAN_SYSTEM},
            {"role": "user", "content": query},
        ],
        format="json",
    )
    data = _parse_json_content(raw)
    # Accept short names; fall back to older keys if a model still emits them
    referents = _as_str_list(data.get("referents") or data.get("unresolved_referents"))
    entities = _as_str_list(data.get("entities") or data.get("opaque_entities"))
    target = data.get("preprobe")
    if target is None:
        target = data.get("preprobe_target")
    if isinstance(target, str):
        target = target.strip() or None
    else:
        target = None
    if not target:
        target = referents[0] if referents else (entities[0] if entities else None)

    planned = normalize_rewrite_timestamps(
        {
            "time_mode": data.get("time_mode", "none"),
            "hard_constraints": data.get("hard_constraints", []),
            "course_general_knowledge": False,
        }
    )
    time_preprobe_only = bool(
        data.get("time_preprobe_only", data.get("scope_time_to_preprobe_only", False))
    )
    return {
        "time_mode": planned.get("time_mode", "none"),
        "hard_constraints": planned.get("hard_constraints", []),
        "time_preprobe_only": time_preprobe_only,
        "referents": referents,
        "entities": entities,
        "preprobe": target,
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
    resolved = str(data.get("entity") or entity).strip() or entity
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    found = bool(definition) and confidence >= PREPROBE_MIN_CONFIDENCE
    return {
        "entity": resolved if found else entity,
        "definition": definition,
        "confidence": confidence,
        "found": found,
    }


def _rewrite_user_message(
    query: str,
    *,
    plan: dict | None = None,
    entity_hints: list[dict] | None = None,
) -> str:
    """Original student query + plan summary + glosses (no full pre-probe dump)."""
    parts = [f"Student question:\n{query}"]
    if plan:
        parts.append(
            "Query plan (time already fixed — do not change it):\n"
            + json.dumps(
                {
                    "time_mode": plan.get("time_mode"),
                    "hard_constraints": plan.get("hard_constraints", []),
                    "referents": plan.get("referents", []),
                    "entities": plan.get("entities", []),
                    "preprobe": plan.get("preprobe"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    hints = [
        h
        for h in (entity_hints or [])
        if h.get("found") and h.get("entity") and h.get("definition")
    ]
    if hints:
        lines = [
            "Resolved entity glosses from pre-probe "
            "(licensed grounding only; do NOT replace the student question):"
        ]
        for h in hints:
            conf = h.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            lines.append(f"- {h['entity']} (confidence={conf_s}): {h['definition']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def rewrite(
    query: str,
    *,
    plan: dict | None = None,
    entity_hints: list[dict] | None = None,
) -> dict:
    """
    Grounded rewritten_query only. Time / constraints come from plan when present.
    """
    raw = ollama_chat(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {
                "role": "user",
                "content": _rewrite_user_message(
                    query, plan=plan, entity_hints=entity_hints
                ),
            },
        ],
        format="json",
    )
    data = _parse_json_content(raw)
    rewritten_query = str(data.get("rewritten_query") or "").strip() or query
    course_general = bool(data.get("course_general_knowledge", False))

    if plan:
        return {
            "rewritten_query": rewritten_query,
            "time_mode": plan.get("time_mode", "none"),
            "course_general_knowledge": course_general,
            "hard_constraints": list(plan.get("hard_constraints") or []),
        }
    # Fallback if called without a plan (should be rare)
    return normalize_rewrite_timestamps(
        {
            "rewritten_query": rewritten_query,
            "time_mode": data.get("time_mode", "none"),
            "course_general_knowledge": course_general,
            "hard_constraints": data.get("hard_constraints", []),
        }
    )


def build_prompt(question, hits, *, preprobe: dict | None = None):
    parts = [f"Student question: {question}"]

    plan = (preprobe or {}).get("plan") or {}
    target = plan_preprobe(plan)
    if target:
        gloss = (preprobe or {}).get("gloss") or {}
        if gloss.get("found") and gloss.get("definition"):
            conf = gloss.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            parts.append(
                "Entity glossary (from pre-probe; use only as term definition):\n"
                f"- {gloss.get('entity') or target} "
                f"(confidence={conf_s}): {gloss['definition']}"
            )
        else:
            parts.append(
                "Entity pre-probe note: "
                f"No related entity definition was found in the lecture materials"
                f" for '{target}'. "
                "Answer using lecture content below; do not invent a definition."
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
