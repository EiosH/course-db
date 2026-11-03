"""Timestamp helpers needed by lecture loaders."""

from config import DEFAULT_LECTURE_MAX_TS


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
    """Normalize timestamps; also fixes common MM:SS mis-parsed as HH:MM:SS."""
    max_ts = lecture_max_ts or DEFAULT_LECTURE_MAX_TS
    ts = normalize_ts_hms(ts)
    h, m, s = map(int, ts.split(":"))
    if ts_hms_to_sec(h, m, s) <= ts_hms_to_sec(*map(int, max_ts.split(":"))):
        return ts
    if h < 60 and m < 60:
        return f"00:{h:02d}:{m:02d}"
    return ts


def ts_to_sec(ts: str, *, lecture_max_ts: str | None = None) -> float:
    h, m, s = map(
        int,
        canonicalize_anchor_timestamp(ts, lecture_max_ts=lecture_max_ts).split(":"),
    )
    return ts_hms_to_sec(h, m, s)
