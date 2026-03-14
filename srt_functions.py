import json
import re
from typing import Any, Dict, List, Sequence

_TS_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$"
)
_TIMING_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
    r"(?:\s+(?P<settings>.*))?$"
)


def _ts_to_ms(ts: str) -> int:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts!r}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms = int(m.group("ms"))
    return (((h * 60 + mi) * 60) + s) * 1000 + ms


def _ms_to_ts(total_ms: int) -> str:
    if total_ms < 0:
        raise ValueError("Timestamp cannot be negative")
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h > 99:
        raise ValueError("SRT hours exceed 99 (unsupported by this formatter)")
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_to_json(srt_text: str, *, as_string: bool = False) -> List[Dict[str, Any]] | str:
    text = srt_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.split("\n") if ln.strip() != "" or True]
        if len(lines) < 2:
            continue

        idx_line = lines[0].strip()
        if not idx_line.isdigit():
            raise ValueError(f"Invalid SRT index line: {idx_line!r}")
        cue_id = int(idx_line)

        timing_line = lines[1].strip()
        tm = _TIMING_RE.match(timing_line)
        if not tm:
            raise ValueError(f"Invalid SRT timing line: {timing_line!r}")

        start = tm.group("start")
        end = tm.group("end")
        settings = tm.group("settings")
        cue_text = "\n".join(lines[2:]).rstrip()

        out.append(
            {
                "id": cue_id,
                "start": start,
                "end": end,
                "start_ms": _ts_to_ms(start),
                "end_ms": _ts_to_ms(end),
                "settings": settings if settings else None,
                "text": cue_text,
            }
        )

    out.sort(key=lambda d: d["id"])

    if as_string:
        return json.dumps(out, ensure_ascii=False, indent=2)
    return out


def json_to_srt(items: Sequence[Dict[str, Any]] | str) -> str:
    if isinstance(items, str):
        items = json.loads(items)

    cues: List[Dict[str, Any]] = []
    for it in items:  # type: ignore[assignment]
        if not isinstance(it, dict):
            raise ValueError("Each JSON item must be an object/dict")

        if "id" not in it or "text" not in it:
            raise ValueError("Each item must contain 'id' and 'text'")

        cue_id = int(it["id"])
        cue_text = "" if it["text"] is None else str(it["text"])

        settings = it.get("settings")
        settings_str = f" {settings}".rstrip() if settings else ""

        if "start" in it and "end" in it:
            start = str(it["start"]).strip()
            end = str(it["end"]).strip()
            _ = _ts_to_ms(start)
            _ = _ts_to_ms(end)
        elif "start_ms" in it and "end_ms" in it:
            start = _ms_to_ts(int(it["start_ms"]))
            end = _ms_to_ts(int(it["end_ms"]))
        else:
            raise ValueError("Each item must contain either ('start','end') or ('start_ms','end_ms')")

        cues.append(
            {
                "id": cue_id,
                "start": start,
                "end": end,
                "settings_str": settings_str,
                "text": cue_text,
            }
        )

    cues.sort(key=lambda d: d["id"])

    parts: List[str] = []
    for cue in cues:
        parts.append(str(cue["id"]))
        parts.append(f"{cue['start']} --> {cue['end']}{cue['settings_str']}".rstrip())
        parts.append(cue["text"].rstrip("\n"))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
