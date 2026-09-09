"""Lecture timestamp normalization and conversion helpers."""

from config import LECTURE_MAX_TS, TIMESTAMP


def normalize_ts_hms(ts: str) -> str:
    """Lecture timestamp → canonical HH:MM:SS (accepts MM:SS or HH:MM:SS)."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        return f"00:{m:02d}:{s:02d}"
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{h:02d}:{m:02d}:{s:02d}"
    raise ValueError(f"invalid timestamp: {ts!r}")


def ts_hms_to_sec(h: int, m: int, s: int) -> float:
    return h * 3600 + m * 60 + float(s)


def canonicalize_anchor_timestamp(ts: str) -> str:
    """
    Normalize rewrite/output timestamps for index lookup.
    - MM:SS → 00:MM:SS
    - Fix common LLM mistake: 05:00 → 05:00:00 when user meant 00:05:00
    """
    ts = normalize_ts_hms(ts)
    h, m, s = map(int, ts.split(":"))
    if ts_hms_to_sec(h, m, s) <= ts_hms_to_sec(
        *map(int, LECTURE_MAX_TS.split(":"))
    ):
        return ts
    if h < 60 and m < 60:
        return f"00:{h:02d}:{m:02d}"
    return ts


def normalize_rewrite_timestamps(rewritten: dict) -> dict:
    constraints = []
    for c in rewritten.get("hard_constraints", []):
        if c.get("field") == "timestamp" and c.get("value"):
            c = {**c, "value": canonicalize_anchor_timestamp(str(c["value"]))}
        constraints.append(c)
    return {**rewritten, "hard_constraints": constraints}


def timestamp_constraint_value(constraints):
    for c in constraints:
        if c.get("field") == "timestamp":
            raw = c.get("value", TIMESTAMP)
            return canonicalize_anchor_timestamp(str(raw))
    return None


def resolve_time_mode(rewritten: dict) -> str:
    """Route retrieval from rewrite LLM output (no regex on user text)."""
    mode = rewritten.get("time_mode")
    if mode in ("now", "anchor", "none"):
        return mode
    ts = timestamp_constraint_value(rewritten.get("hard_constraints", []))
    if ts is None:
        return "none"
    if ts == TIMESTAMP:
        return "now"
    return "anchor"


def ts_to_sec(ts: str) -> float:
    h, m, s = map(int, canonicalize_anchor_timestamp(ts).split(":"))
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
