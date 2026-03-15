import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

def split_for_translation(
    chunks: List[Dict[str, Any]],
    min_group_size: int = 10,
    lookahead: int = 5,
    gap_threshold_ms: int = 2000,
    end_punct: Tuple[str, ...] = (".", "?", "!"),
) -> List[List[Dict[str, Any]]]:
    """
    Split subtitle chunks into groups using the user-specified policy:

    - Each group has at least `min_group_size` chunks, unless we run out at the end.
    - Once `min_group_size` is reached, inspect the following `lookahead` chunks:
        1) If a gap between consecutive cues is > gap_threshold_ms, break *before* the chunk after the gap.
        2) Else, if any of those lookahead chunks ends with '.', '?', or '!', break *at* that chunk.
        3) Else, break after those lookahead chunks (i.e., group size = min_group_size + lookahead),
           or earlier if the file ends.

    Returns: list of groups, each group is a list of SRTChunk.
    """
    n = len(chunks)
    if n == 0:
        return []

    groups: List[List[Dict[str, Any]]] = []
    i = 0

    while i < n:
        # If not enough remaining to reach min_group_size, just take the rest.
        if i + min_group_size >= n:
            groups.append(chunks[i:])
            break

        base_end = i + min_group_size - 1  # inclusive index of the 10th chunk in the group
        look_start = base_end + 1
        look_end = min(n - 1, base_end + lookahead)  # inclusive

        break_at: Optional[int] = None  # inclusive index where group ends

        # 1) Gap rule: find first gap > threshold among boundary pairs inside the lookahead window
        # We need to check gaps between consecutive cues; the earliest relevant pair could be
        # (look_start-1, look_start), then (look_start, look_start+1), ... up to look_end.
        for k in range(look_start, look_end + 1):
            prev = chunks[k - 1]
            curr = chunks[k]
            gap = curr["start_ms"] - prev["end_ms"]
            if gap > gap_threshold_ms:
                # break before curr => group ends at k-1
                break_at = k - 1
                break

        # 2) Punctuation rule: within the same lookahead chunks
        if break_at is None:
            for k in range(look_start, look_end + 1):
                if chunks[k]["text"].rstrip().endswith(end_punct):
                    break_at = k
                    break

        # 3) Fallback: split after lookahead (i.e., after total min_group_size + lookahead),
        # or at end if fewer remain.
        if break_at is None:
            break_at = look_end

        groups.append(chunks[i : break_at + 1])
        i = break_at + 1

    return groups
