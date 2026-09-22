"""Lecture timestamp normalization and conversion helpers."""

from config import DEFAULT_LECTURE_MAX_TS, TIMESTAMP


def normalize_ts_hms(ts: str) -> str:
    """Lecture timestamp → canonical HH:MM:SS (accepts MM:SS or HH:MM:SS)."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        return f"00:{m:02d}:{s:02d}"
    if len(parts) == 3:
        # VTT may use HH:MM:SS.mmm — drop fractional seconds
        h, m, s = int(parts[0]), int(parts[1]), int(float(parts[2]))
        return f"{h:02d}:{m:02d}:{s:02d}"
    raise ValueError(f"invalid timestamp: {ts!r}")


def ts_hms_to_sec(h: int, m: int, s: int) -> float:
    return h * 3600 + m * 60 + float(s)


def canonicalize_anchor_timestamp(
    ts: str, *, lecture_max_ts: str | None = None
) -> str:
    """
    Normalize rewrite/output timestamps for index lookup.
    - MM:SS → 00:MM:SS
    - Fix common LLM mistake: 05:00 → 05:00:00 when user meant 00:05:00
    """
    max_ts = lecture_max_ts or DEFAULT_LECTURE_MAX_TS
    ts = normalize_ts_hms(ts)
    h, m, s = map(int, ts.split(":"))
    if ts_hms_to_sec(h, m, s) <= ts_hms_to_sec(*map(int, max_ts.split(":"))):
        return ts
    if h < 60 and m < 60:
        return f"00:{h:02d}:{m:02d}"
    return ts


def normalize_rewrite_timestamps(
    rewritten: dict, *, lecture_max_ts: str | None = None
) -> dict:
    constraints = []
    for c in rewritten.get("hard_constraints", []):
        if c.get("field") == "timestamp" and c.get("value"):
            c = {
                **c,
                "value": canonicalize_anchor_timestamp(
                    str(c["value"]), lecture_max_ts=lecture_max_ts
                ),
            }
        constraints.append(c)
    return {**rewritten, "hard_constraints": constraints}


def timestamp_constraint_value(constraints, *, course_ctx: dict | None = None):
    default_ts = (course_ctx or {}).get("timestamp") or TIMESTAMP
    lecture_max_ts = (course_ctx or {}).get("lecture_max_ts")
    for c in constraints:
        if c.get("field") == "timestamp":
            raw = c.get("value", default_ts)
            return canonicalize_anchor_timestamp(
                str(raw), lecture_max_ts=lecture_max_ts
            )
    return None


def resolve_time_mode(rewritten: dict, *, course_ctx: dict | None = None) -> str:
    """Route retrieval from rewrite LLM output (no regex on user text)."""
    mode = rewritten.get("time_mode")
    if mode in ("now", "anchor", "none"):
        return mode
    ts = timestamp_constraint_value(
        rewritten.get("hard_constraints", []), course_ctx=course_ctx
    )
    if ts is None:
        return "none"
    now_ts = (course_ctx or {}).get("timestamp") or TIMESTAMP
    if ts == now_ts:
        return "now"
    return "anchor"


def ts_to_sec(ts: str, *, lecture_max_ts: str | None = None) -> float:
    h, m, s = map(
        int,
        canonicalize_anchor_timestamp(ts, lecture_max_ts=lecture_max_ts).split(":"),
    )
    return ts_hms_to_sec(h, m, s)


def time_distance(center: float, payload: dict) -> float:
    start = payload.get("start_sec")
    end = payload.get("end_sec")
    if start is not None and end is not None:
        if start <= center <= end:
            return 0.0
        return min(abs(start - center), abs(end - center))
    ts = payload.get("timestamp", "")
    if ts and "-->" not in ts:
        try:
            return abs(ts_to_sec(ts) - center)
        except (ValueError, AttributeError):
            pass
    return float("inf")
