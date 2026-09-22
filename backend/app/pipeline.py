"""Core Q&A pipeline as a minimal LangChain LCEL chain."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.runnables import RunnableBranch, RunnableLambda

from answering import (
    answer,
    build_prompt,
    extract_resolved_name,
    no_hit_reply,
    plan_query,
    plan_resolve,
    rewrite,
)
from config import TIME_NEAR_TOP_K
from search import (
    expand_query_with_anchors,
    merge_unique_hits,
    pick_by_type_quota,
    rerank_and_filter,
    search_bm25,
    search_dense,
    search_preprobe,
    search_time_window,
)
from time_utils import (
    resolve_time_mode,
    time_distance,
    timestamp_constraint_value,
    ts_to_sec,
)
from tracing import observation


@dataclass
class QueryResult:
    """Structured outcome of one question — reporters consume this."""

    query: str
    rewritten: dict
    search_query: str
    time_mode: str
    ts_value: str | None
    ss_dense: list = field(default_factory=list)
    ss_bm25: list = field(default_factory=list)
    tr_dense: list = field(default_factory=list)
    tr_bm25: list = field(default_factory=list)
    final_hits: list = field(default_factory=list)
    prompt: str = ""
    answer_text: str = ""
    preprobe: dict = field(default_factory=dict)


def _empty_preprobe() -> dict:
    return {
        "plan": None,
        "query": None,
        "hits": [],
        "resolved": None,
    }


def _plan(state: dict) -> dict:
    """Stage 0a: unified plan (time + optional resolve + course_general)."""
    plan = plan_query(state["query"])
    print(
        f"plan: time_mode={plan.get('time_mode')} "
        f"course_general={bool(plan.get('course_general_knowledge'))} "
        f"resolve={plan.get('resolve')!r} "
        f"time_preprobe_only={bool(plan.get('time_preprobe_only'))} "
        f"({plan.get('reason')})"
    )
    return {**state, "plan": plan}


def _preprobe(state: dict) -> dict:
    """
    Stage 0b: optional resolve search.
    When resolve is set, a small recall (optionally time-filtered) yields one
    concrete name for rewrite. Main retrieval is separate.
    """
    plan = state.get("plan") or {}
    info = _empty_preprobe()
    info["plan"] = plan
    target = plan_resolve(plan)

    if not target:
        print(f"preprobe: skip ({plan.get('reason') or 'no resolve'})")
        return {**state, "preprobe": info}

    constraints = list(plan.get("hard_constraints") or [])
    print(f"preprobe: resolve={target!r} constraints={constraints}")
    hits, pre_q = search_preprobe(
        state["client"],
        target,
        constraints=constraints,
    )
    info["query"] = pre_q
    info["hits"] = hits

    if not hits:
        info["resolved"] = {
            "phrase": target,
            "name": "",
            "confidence": 0.0,
            "found": False,
        }
        print(f"preprobe: empty recall for {target!r}; continue main retrieval")
        return {**state, "preprobe": info}

    resolved = extract_resolved_name(target, hits)
    info["resolved"] = resolved
    print(
        f"preprobe: hits={len(hits)} found={resolved.get('found')} "
        f"confidence={resolved.get('confidence')} name={resolved.get('name')!r}"
    )
    return {**state, "preprobe": info}


def _prepare(state: dict) -> dict:
    query = state["query"]
    plan = state.get("plan") or {}
    preprobe = state.get("preprobe") or _empty_preprobe()
    resolved = preprobe.get("resolved") or {}
    resolved_name = resolved.get("name") if resolved.get("found") else None

    # Time comes from plan; rewrite only builds a grounded rewritten_query.
    # time_preprobe_only: timestamp locates the resolve phrase; main search
    # then runs without time (related content elsewhere in the lecture).
    rewritten = rewrite(query, plan, resolved_name)
    if plan_resolve(plan) and plan.get("time_preprobe_only"):
        rewritten = {
            **rewritten,
            "time_mode": "none",
            "hard_constraints": [],
        }
    return {
        **state,
        "preprobe": preprobe,
        "rewritten": rewritten,
        "q": rewritten["rewritten_query"],
        "constraints": rewritten["hard_constraints"],
        "time_mode": resolve_time_mode(rewritten),
        "ts_value": timestamp_constraint_value(rewritten["hard_constraints"]),
        "ss_dense": [],
        "ss_bm25": [],
        "tr_dense": [],
        "tr_bm25": [],
        "final_hits": [],
    }


def _retrieve_now(state: dict) -> dict:
    client, constraints = state["client"], state["constraints"]
    ss = search_time_window(client, constraints, "screen_shot", limit=TIME_NEAR_TOP_K)
    tr = search_time_window(client, constraints, "transcript", limit=TIME_NEAR_TOP_K)
    center = ts_to_sec(state["ts_value"])
    final = pick_by_type_quota(merge_unique_hits(ss, tr), TIME_NEAR_TOP_K)
    final = sorted(final, key=lambda h: time_distance(center, h.payload))
    return {**state, "ss_dense": ss, "tr_dense": tr, "final_hits": final}


def _hybrid_recall(state: dict) -> dict:
    """Dense(short) + BM25(expanded) + anchor hits for both doc types."""
    client, query, q_short = state["client"], state["query"], state["q"]
    constraints = state["constraints"]
    q_expand, anchors = expand_query_with_anchors(client, query, q_short, constraints)
    ss_dense = search_dense(client, q_short, "screen_shot", constraints)
    ss_bm25 = search_bm25(client, q_expand, "screen_shot", constraints)
    tr_dense = search_dense(client, q_short, "transcript", constraints)
    tr_bm25 = search_bm25(client, q_expand, "transcript", constraints)
    return {
        **state,
        "q": q_expand,
        "ss_dense": ss_dense,
        "ss_bm25": ss_bm25,
        "tr_dense": tr_dense,
        "tr_bm25": tr_bm25,
        "anchors": anchors,
    }


def _finalize_time(state: dict) -> dict:
    center = ts_to_sec(state["ts_value"])
    tw_ss = merge_unique_hits(state["anchors"], state["ss_dense"], state["ss_bm25"])
    tw_tr = merge_unique_hits(state["tr_dense"], state["tr_bm25"])
    final = pick_by_type_quota(merge_unique_hits(tw_ss, tw_tr), TIME_NEAR_TOP_K)
    final = sorted(final, key=lambda h: time_distance(center, h.payload))
    return {**state, "final_hits": final}


def _finalize_rerank(state: dict) -> dict:
    candidates = merge_unique_hits(
        state["anchors"],
        state["ss_dense"],
        state["ss_bm25"],
        state["tr_dense"],
        state["tr_bm25"],
    )
    final = rerank_and_filter(f"{state['query']}\n{state['q']}", candidates)
    return {**state, "final_hits": final}


def _answer(state: dict) -> dict:
    rewritten = state["rewritten"]
    hits = state["final_hits"]
    course_general = bool(rewritten.get("course_general_knowledge"))
    prompt = build_prompt(
        state["query"],
        hits,
        course_general=course_general,
    )
    if hits or course_general:
        text = answer(prompt)
    else:
        text = no_hit_reply(now=state["time_mode"] == "now")
    return {**state, "prompt": prompt, "answer_text": text}


def _to_result(state: dict) -> QueryResult:
    return QueryResult(
        query=state["query"],
        rewritten=state["rewritten"],
        search_query=state["q"],
        time_mode=state["time_mode"],
        ts_value=state["ts_value"],
        ss_dense=state["ss_dense"],
        ss_bm25=state["ss_bm25"],
        tr_dense=state["tr_dense"],
        tr_bm25=state["tr_bm25"],
        final_hits=state["final_hits"],
        prompt=state["prompt"],
        answer_text=state["answer_text"],
        preprobe=state.get("preprobe") or _empty_preprobe(),
    )


# plan → optional resolve search → grounded rewrite → branch(retrieve) → answer → QueryResult
_CHAIN = (
    RunnableLambda(_plan)
    | RunnableLambda(_preprobe)
    | RunnableLambda(_prepare)
    | RunnableBranch(
        (lambda s: s["time_mode"] == "now", RunnableLambda(_retrieve_now)),
        (
            lambda s: bool(s["ts_value"]),
            RunnableLambda(_hybrid_recall) | RunnableLambda(_finalize_time),
        ),
        RunnableLambda(_hybrid_recall) | RunnableLambda(_finalize_rerank),
    )
    | RunnableLambda(_answer)
    | RunnableLambda(_to_result)
)


def answer_query(client, query: str) -> QueryResult:
    """Public entry: same contract as before for eval / callers."""
    # One Langfuse trace per question. Nested ollama_chat generations attach here.
    # (LangChain CallbackHandler is intentionally NOT used: it opens a second root.)
    with observation(
        name="answer_query",
        as_type="chain",
        input={"query": query},
    ) as span:
        result = _CHAIN.invoke({"client": client, "query": query})
        if span is not None:
            plan = (result.preprobe or {}).get("plan") or {}
            span.update(
                output={"answer": result.answer_text},
                metadata={
                    "time_mode": result.time_mode,
                    "ts_value": result.ts_value,
                    "resolve": plan.get("resolve"),
                    "time_preprobe_only": bool(plan.get("time_preprobe_only")),
                    "course_general": bool(
                        result.rewritten.get("course_general_knowledge")
                    ),
                    "final_hits": len(result.final_hits),
                    "search_query": result.search_query,
                },
            )
        return result
