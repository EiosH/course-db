"""Core Q&A pipeline as a minimal LangChain LCEL chain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableBranch, RunnableLambda

from answering import (
    answer,
    assemble_plan,
    build_prompt,
    extract_resolved_name,
    no_hit_reply,
    plan_knowledge,
    plan_resolve,
    plan_resolve_part,
    plan_time,
    rewrite,
)
from config import TIME_NEAR_TOP_K
from course_route import route_course, session_course_ctx
from search import (
    expand_query_with_anchors,
    merge_unique_hits,
    pick_by_type_quota,
    rerank_and_filter,
    search_bm25,
    search_dense,
    search_preprobe,
    search_time_window,
    with_lecture_constraint,
)
from time_utils import (
    resolve_time_mode,
    time_distance,
    timestamp_constraint_value,
    ts_to_sec,
)
from tracing import hits_preview, observation, run_with_parent


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
    course_ctx: dict = field(default_factory=dict)


def _empty_preprobe() -> dict:
    return {
        "plan": None,
        "query": None,
        "hits": [],
        "resolved": None,
    }


def _plan_course_route(query: str) -> dict:
    with observation(
        name="course_route",
        as_type="span",
        require_parent=True,
        input={"query": query},
    ) as obs:
        course_ctx = route_course(query)
        print(
            f"course_route: {course_ctx['course_id']} / {course_ctx['quarter']} / "
            f"{course_ctx['lecturer']!r} lecture_ids={course_ctx.get('lecture_ids')} "
            f"({course_ctx.get('reason')})"
        )
        if obs is not None:
            obs.update(output=course_ctx)
        return course_ctx


def _plan_time_part(query: str, session_ctx: dict) -> dict:
    with observation(
        name="plan_time",
        as_type="span",
        require_parent=True,
        input={"query": query},
    ) as obs:
        part = plan_time(query, session_ctx)
        if obs is not None:
            obs.update(output=part)
        return part


def _plan_resolve_sibling(query: str, session_ctx: dict) -> dict:
    with observation(
        name="plan_resolve",
        as_type="span",
        require_parent=True,
        input={"query": query},
    ) as obs:
        part = plan_resolve_part(query, session_ctx)
        if obs is not None:
            obs.update(output=part)
        return part


def _plan_knowledge_part(query: str, session_ctx: dict) -> dict:
    with observation(
        name="plan_knowledge",
        as_type="span",
        require_parent=True,
        input={"query": query},
    ) as obs:
        part = plan_knowledge(query, session_ctx)
        if obs is not None:
            obs.update(output=part)
        return part


def _plan(state: dict) -> dict:
    """
    Stage 0: parallel siblings under plan, then assemble constraints.
    Independent: course_route | time | resolve | knowledge.
    Sequential after this: preprobe → rewrite → recall.
    """
    query = state["query"]
    session_ctx = session_course_ctx()
    with observation(
        name="plan",
        as_type="span",
        require_parent=True,
        input={"query": query, "course_id": session_ctx.get("course_id")},
    ) as obs:
        parent = obs
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_route = pool.submit(run_with_parent, parent, _plan_course_route, query)
            f_time = pool.submit(
                run_with_parent, parent, _plan_time_part, query, session_ctx
            )
            f_resolve = pool.submit(
                run_with_parent, parent, _plan_resolve_sibling, query, session_ctx
            )
            f_knowledge = pool.submit(
                run_with_parent, parent, _plan_knowledge_part, query, session_ctx
            )
            course_ctx = f_route.result()
            time_part = f_time.result()
            resolve_part = f_resolve.result()
            knowledge_part = f_knowledge.result()

        plan = assemble_plan(
            time_part=time_part,
            resolve_part=resolve_part,
            knowledge_part=knowledge_part,
        )
        plan = {
            **plan,
            "hard_constraints": with_lecture_constraint(
                plan.get("hard_constraints"), course_ctx
            ),
            "probe_hard_constraints": list(plan.get("probe_hard_constraints") or []),
        }
        print(
            f"plan: time_mode={plan.get('time_mode')} "
            f"course_general={bool(plan.get('course_general_knowledge'))} "
            f"resolve={plan.get('resolve')!r} "
            f"hard_cs={plan.get('hard_constraints')} "
            f"probe_cs={plan.get('probe_hard_constraints')} "
            f"({plan.get('reason')})"
        )
        if obs is not None:
            obs.update(output={"course_ctx": course_ctx, "plan": plan})
        return {**state, "course_ctx": course_ctx, "plan": plan}


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

    # Probe-only constraints if set; otherwise reuse main hard_constraints.
    probe_cs = list(plan.get("probe_hard_constraints") or [])
    main_cs = list(plan.get("hard_constraints") or [])
    constraints = (
        with_lecture_constraint(probe_cs, state["course_ctx"])
        if probe_cs
        else main_cs
    )
    print(f"preprobe: resolve={target!r} constraints={constraints}")

    with observation(
        name="resolve_retrieve",
        as_type="retriever",
        require_parent=True,
        input={"resolve": target, "constraints": constraints},
    ) as obs:
        hits, pre_q = search_preprobe(
            state["client"],
            target,
            state["course_ctx"],
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
            if obs is not None:
                obs.update(output={"query": pre_q, **hits_preview(hits)})
            return {**state, "preprobe": info}

        resolved = extract_resolved_name(target, hits)
        info["resolved"] = resolved
        print(
            f"preprobe: hits={len(hits)} found={resolved.get('found')} "
            f"confidence={resolved.get('confidence')} name={resolved.get('name')!r}"
        )
        if obs is not None:
            obs.update(
                output={
                    "query": pre_q,
                    "resolved": resolved,
                    **hits_preview(hits),
                }
            )
        return {**state, "preprobe": info}


def _prepare(state: dict) -> dict:
    query = state["query"]
    plan = state.get("plan") or {}
    preprobe = state.get("preprobe") or _empty_preprobe()
    resolved = preprobe.get("resolved") or {}
    resolved_name = resolved.get("name") if resolved.get("found") else None

    with observation(
        name="rewrite",
        as_type="span",
        require_parent=True,
        input={
            "query": query,
            "resolved_name": resolved_name,
            "probe_hard_constraints": plan.get("probe_hard_constraints"),
        },
    ) as obs:
        # Time / filters come from plan.hard_constraints (main retrieval).
        rewritten = rewrite(query, plan, state["course_ctx"], resolved_name)
        rewritten = {
            **rewritten,
            "hard_constraints": with_lecture_constraint(
                rewritten.get("hard_constraints"), state["course_ctx"]
            ),
        }
        out = {
            **state,
            "preprobe": preprobe,
            "rewritten": rewritten,
            "q": rewritten["rewritten_query"],
            "constraints": rewritten["hard_constraints"],
            "time_mode": resolve_time_mode(
                rewritten, course_ctx=state["course_ctx"]
            ),
            "ts_value": timestamp_constraint_value(
                rewritten["hard_constraints"], course_ctx=state["course_ctx"]
            ),
            "ss_dense": [],
            "ss_bm25": [],
            "tr_dense": [],
            "tr_bm25": [],
            "final_hits": [],
        }
        if obs is not None:
            obs.update(
                output={
                    "rewritten_query": rewritten.get("rewritten_query"),
                    "time_mode": out["time_mode"],
                    "ts_value": out["ts_value"],
                    "constraints": out["constraints"],
                }
            )
        return out


def _retrieve_now(state: dict) -> dict:
    with observation(
        name="retrieve_now",
        as_type="retriever",
        require_parent=True,
        input={
            "ts_value": state.get("ts_value"),
            "constraints": state.get("constraints"),
        },
    ) as obs:
        client, constraints = state["client"], state["constraints"]
        course_ctx = state["course_ctx"]
        ss = search_time_window(
            client, constraints, course_ctx, "screen_shot", limit=TIME_NEAR_TOP_K
        )
        tr = search_time_window(
            client, constraints, course_ctx, "transcript", limit=TIME_NEAR_TOP_K
        )
        center = ts_to_sec(
            state["ts_value"], lecture_max_ts=course_ctx.get("lecture_max_ts")
        )
        final = pick_by_type_quota(merge_unique_hits(ss, tr), TIME_NEAR_TOP_K)
        final = sorted(final, key=lambda h: time_distance(center, h.payload))
        out = {**state, "ss_dense": ss, "tr_dense": tr, "final_hits": final}
        if obs is not None:
            obs.update(
                output={
                    "ss": len(ss),
                    "tr": len(tr),
                    **hits_preview(final),
                }
            )
        return out


def _hybrid_recall(state: dict) -> dict:
    """Dense(short) + BM25(expanded) + anchor hits for both doc types."""
    constraints = state.get("constraints") or []
    course_ctx = state.get("course_ctx") or {}
    with observation(
        name="hybrid_recall",
        as_type="retriever",
        require_parent=True,
        input={
            "q": state.get("q"),
            "constraints": constraints,
            "course_id": course_ctx.get("course_id"),
            "lecture_ids": course_ctx.get("lecture_ids"),
        },
    ) as obs:
        client, query, q_short = state["client"], state["query"], state["q"]
        print(f"hybrid_recall: constraints={constraints}")
        q_expand, anchors = expand_query_with_anchors(
            client, query, q_short, constraints, course_ctx
        )
        ss_dense = search_dense(
            client, q_short, "screen_shot", constraints, course_ctx
        )
        ss_bm25 = search_bm25(
            client, q_expand, "screen_shot", constraints, course_ctx
        )
        tr_dense = search_dense(
            client, q_short, "transcript", constraints, course_ctx
        )
        tr_bm25 = search_bm25(
            client, q_expand, "transcript", constraints, course_ctx
        )
        out = {
            **state,
            "q": q_expand,
            "ss_dense": ss_dense,
            "ss_bm25": ss_bm25,
            "tr_dense": tr_dense,
            "tr_bm25": tr_bm25,
            "anchors": anchors,
        }
        if obs is not None:
            obs.update(
                output={
                    "q_expand": q_expand,
                    "anchors": len(anchors),
                    "ss_dense": len(ss_dense),
                    "ss_bm25": len(ss_bm25),
                    "tr_dense": len(tr_dense),
                    "tr_bm25": len(tr_bm25),
                }
            )
        return out


def _finalize_time(state: dict) -> dict:
    with observation(
        name="finalize_time",
        as_type="retriever",
        require_parent=True,
        input={"ts_value": state.get("ts_value")},
    ) as obs:
        center = ts_to_sec(
            state["ts_value"],
            lecture_max_ts=(state.get("course_ctx") or {}).get("lecture_max_ts"),
        )
        tw_ss = merge_unique_hits(
            state["anchors"], state["ss_dense"], state["ss_bm25"]
        )
        tw_tr = merge_unique_hits(state["tr_dense"], state["tr_bm25"])
        final = pick_by_type_quota(merge_unique_hits(tw_ss, tw_tr), TIME_NEAR_TOP_K)
        final = sorted(final, key=lambda h: time_distance(center, h.payload))
        if obs is not None:
            obs.update(output=hits_preview(final))
        return {**state, "final_hits": final}


def _finalize_rerank(state: dict) -> dict:
    with observation(
        name="finalize_rerank",
        as_type="retriever",
        require_parent=True,
        input={"query": state.get("query"), "q": state.get("q")},
    ) as obs:
        candidates = merge_unique_hits(
            state["anchors"],
            state["ss_dense"],
            state["ss_bm25"],
            state["tr_dense"],
            state["tr_bm25"],
        )
        # Rerank on topic query only — original question may contain lecture
        # deixis ("previous courses") that biases toward meta mentions.
        rerank_q = (state.get("q") or state.get("query") or "").strip()
        final = rerank_and_filter(rerank_q, candidates)
        if obs is not None:
            obs.update(
                output={
                    "candidates": len(candidates),
                    "rerank_q": rerank_q,
                    **hits_preview(final),
                }
            )
        return {**state, "final_hits": final}


def _answer(state: dict) -> dict:
    with observation(
        name="answer",
        as_type="span",
        require_parent=True,
        input={
            "query": state.get("query"),
            "hits": len(state.get("final_hits") or []),
            "course_general": bool(
                (state.get("rewritten") or {}).get("course_general_knowledge")
            ),
            "lecture_ids": (state.get("course_ctx") or {}).get("lecture_ids"),
        },
    ) as obs:
        rewritten = state["rewritten"]
        hits = state["final_hits"]
        course_general = bool(rewritten.get("course_general_knowledge"))
        prompt = build_prompt(
            state["query"],
            hits,
            course_general=course_general,
            course_ctx=state.get("course_ctx") or {},
        )
        if hits or course_general:
            text = answer(prompt)
        else:
            text = no_hit_reply(now=state["time_mode"] == "now")
        if obs is not None:
            obs.update(output={"answer": text})
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
        course_ctx=state.get("course_ctx") or {},
    )


# plan (parallel: course_route | time | resolve | knowledge) → preprobe → rewrite → retrieve → answer
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
            ctx = result.course_ctx or {}
            span.update(
                output={"answer": result.answer_text},
                metadata={
                    "time_mode": result.time_mode,
                    "ts_value": result.ts_value,
                    "course_id": ctx.get("course_id"),
                    "quarter": ctx.get("quarter"),
                    "lecture_id": ctx.get("lecture_id"),
                    "lecture_ids": ctx.get("lecture_ids"),
                    "resolve": plan.get("resolve"),
                    "probe_hard_constraints": plan.get("probe_hard_constraints"),
                    "course_general": bool(
                        result.rewritten.get("course_general_knowledge")
                    ),
                    "final_hits": len(result.final_hits),
                    "search_query": result.search_query,
                },
            )
            print(f"langfuse: trace_id={span.trace_id}")
        return result
