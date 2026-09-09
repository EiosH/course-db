"""Core Q&A pipeline as a minimal LangChain LCEL chain."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.runnables import RunnableBranch, RunnableLambda

from answering import (
    answer,
    build_prompt,
    extract_entity_gloss,
    judge_unknown_entity,
    no_hit_answer,
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
from time_utils import resolve_time_mode, time_distance, timestamp_constraint_value, ts_to_sec


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
        "needs_preprobe": False,
        "unknown_entity": None,
        "reason": "",
        "query": None,
        "hits": [],
        "gloss": None,
        "entity_not_found": False,
    }


def _preprobe(state: dict) -> dict:
    """
    Stage 0: entity unfamiliarity → optional forced pre-probe.
    Pre-probe uses one query "What is {entity}", top 1–3, no rerank.
    """
    query = state["query"]
    judgment = judge_unknown_entity(query)
    info = _empty_preprobe()
    info["needs_preprobe"] = judgment["needs_preprobe"]
    info["unknown_entity"] = judgment["unknown_entity"]
    info["reason"] = judgment["reason"]

    if not judgment["needs_preprobe"]:
        print(f"preprobe: skip ({judgment.get('reason') or 'no unfamiliar entity'})")
        return {**state, "preprobe": info}

    entity = judgment["unknown_entity"]
    print(f"preprobe: force on entity={entity!r} ({judgment.get('reason')})")
    # Pre-probe has no time/homework constraints — definition lookup over the lecture
    hits, pre_q = search_preprobe(state["client"], entity, constraints=[])
    info["query"] = pre_q
    info["hits"] = hits

    if not hits:
        info["entity_not_found"] = True
        info["gloss"] = {
            "entity": entity,
            "definition": "",
            "confidence": 0.0,
            "found": False,
        }
        print(f"preprobe: empty recall for {entity!r}; continue main retrieval")
        return {**state, "preprobe": info}

    gloss = extract_entity_gloss(entity, hits)
    info["gloss"] = gloss
    info["entity_not_found"] = not gloss.get("found")
    print(
        f"preprobe: hits={len(hits)} found={gloss.get('found')} "
        f"confidence={gloss.get('confidence')} def={gloss.get('definition')!r}"
    )
    return {**state, "preprobe": info}


def _prepare(state: dict) -> dict:
    query = state["query"]
    preprobe = state.get("preprobe") or _empty_preprobe()
    entity_hints = []
    gloss = preprobe.get("gloss")
    if gloss and gloss.get("found"):
        entity_hints = [gloss]

    rewritten = rewrite(query, entity_hints=entity_hints or None)
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
    preprobe = state.get("preprobe") or _empty_preprobe()
    prompt = build_prompt(state["query"], state["final_hits"], preprobe=preprobe)
    if state["final_hits"]:
        text = answer(prompt)
    else:
        text = no_hit_answer(
            state["query"],
            state["rewritten"],
            now=state["time_mode"] == "now",
        )
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


# entity-judge → optional preprobe → rewrite → branch(retrieve) → answer → QueryResult
_CHAIN = (
    RunnableLambda(_preprobe)
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
    return _CHAIN.invoke({"client": client, "query": query})
