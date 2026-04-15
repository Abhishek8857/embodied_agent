"""
memory_summarizer.py  (navigation stack)
-----------------------------------------
Reads multiple episodic memory session JSON files and produces a two-layer
memory file optimised for injecting into a wheeled navigation agent's context.

Layer 1 – High-signal, injected every call (small):
  world_state   current pose and last known location derived from recent episodes
  procedures    canonical tool chains mined from successful episodes

Layer 2 – Recent context, injected for novel/recovery situations:
  recent_episodes  last N *meaningful* episodes (trivial info queries stripped)
  stats            tool frequency, success rates

Output
------
  memory.json       full two-layer structure
  memory.txt        ultra-compact one-liner-per-episode version (--txt)

Usage
-----
    python memory_summarizer.py                          # reads *.json in ./memory/
    python memory_summarizer.py --dir /path/to/sessions  # custom folder
    python memory_summarizer.py --files s1.json s2.json  # specific files
    python memory_summarizer.py --top-k 20               # keep only 20 recent episodes
    python memory_summarizer.py --txt                    # also write .txt version
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


_BLOCKLIST_KEYS = {
    "mask", "mask_array", "feedback", "sent_data",
    "rgb_shape", "depth_shape",
}

# Keys to keep from each tool's output — everything else is dropped to save tokens.
_SLIM_OUTPUT_KEYS: dict[str, list[str]] = {
    "navigate_to_location": ["success", "status", "message"],
    "get_current_pose":     ["success", "x", "y", "yaw_degrees"],
    "relative_move":        ["success", "x", "y", "yaw_degrees"],
    "save_location":        ["success", "name"],
    "delete_location":      ["success", "name"],
    "list_locations":       ["success", "locations"],
    "cancel_navigation":    ["success"],
}

# Queries that are operationally trivial and low-signal for future planning.
# These are kept in stats but excluded from recent_episodes injected to the agent.
_TRIVIAL_QUERY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*what\s+can\s+you\s+do\s*$",
        r"^\s*help\s*$",
        r"^\s*what\s+location(s)?\s+(do\s+you\s+know|are\s+(there|available))\s*$",
        r"^\s*list\s+location(s)?\s*$",
    ]
]

# Task-type classifiers  (query → procedure key)
_TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"go\s+back\s+to\s+home|go\s+home|home\s+pose|return\s+(to\s+)?home", re.IGNORECASE), "home"),
    (re.compile(r"navigate|go\s+to|take\s+me|move\s+to|drive\s+to",                   re.IGNORECASE), "navigate"),
    (re.compile(r"move\s+forward|move\s+back(ward)?|go\s+(left|right|straight)",       re.IGNORECASE), "relative_move"),
    (re.compile(r"turn\s+(left|right)|rotate|spin",                                    re.IGNORECASE), "rotate"),
    (re.compile(r"save\s+(this|location|spot|pose|here)|remember\s+this",             re.IGNORECASE), "save_location"),
    (re.compile(r"delete|remove\s+(location|spot|waypoint)",                           re.IGNORECASE), "delete_location"),
    (re.compile(r"stop|cancel|abort|halt",                                             re.IGNORECASE), "stop"),
    (re.compile(r"where\s+am\s+i|current\s+pose|current\s+position",                  re.IGNORECASE), "query_pose"),
]



def _classify_task(query: str) -> str:
    for pattern, label in _TASK_PATTERNS:
        if pattern.search(query):
            return label
    return "other"


def _is_trivial(query: str) -> bool:
    return any(p.match(query) for p in _TRIVIAL_QUERY_PATTERNS)


def _slim_output(tool_name: str, raw_output: str) -> str:
    spec = _SLIM_OUTPUT_KEYS.get(tool_name)

    try:
        obj = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return raw_output[:200]

    if not isinstance(obj, dict):
        return str(obj)[:200]

    obj = {k: v for k, v in obj.items() if k not in _BLOCKLIST_KEYS}

    if spec:
        slim = {k: obj[k] for k in spec if k in obj}
        return json.dumps(slim)

    return json.dumps(obj)


def _slim_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in _BLOCKLIST_KEYS}


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    text = text.strip().splitlines()[0]
    m = re.search(r'[.!?]', text)
    return text[: m.start() + 1] if m else text[:150]


def _extract_outcome_fact(ep: dict) -> str:
    """
    Build a terse factual outcome sentence from the structured episode data.
    Pulls the "Result:" line from the final_response where available.
    """
    query    = ep.get("query", "")
    outcome  = ep.get("outcome", "unknown")
    response = ep.get("final_response", "")
    task     = _classify_task(query)
    key_args = _extract_key_args(ep)

    # Try to extract a "Result:" line from the agent's response JSON
    try:
        resp_data = json.loads(response)
        response_text = resp_data.get("response", response)
    except (json.JSONDecodeError, TypeError):
        response_text = response

    result_match = re.search(r"Result:\s*([^\n]+)", response_text, re.IGNORECASE)
    result_str   = result_match.group(1).strip() if result_match else outcome.upper()

    location = key_args.get("location_name", "")

    if task == "navigate":
        loc_str = f" to {location}" if location else ""
        return f"Navigate{loc_str}. {result_str}."

    if task == "home":
        return f"Return to home. {result_str}."

    if task == "relative_move":
        dist_x = key_args.get("distance_x", "")
        dist_y = key_args.get("distance_y", "")
        move_str = ""
        if dist_x:
            move_str += f" forward {dist_x}m"
        if dist_y:
            move_str += f" lateral {dist_y}m"
        return f"Relative move{move_str}. {result_str}."

    if task == "rotate":
        angle = key_args.get("angle_degrees", "")
        angle_str = f" {angle}°" if angle else ""
        return f"Rotate{angle_str}. {result_str}."

    if task == "save_location":
        return f"Saved location '{location}'. {result_str}."

    if task == "delete_location":
        return f"Deleted location '{location}'. {result_str}."

    # Generic fallback
    return f"{query.strip().capitalize()}. {result_str}."


def _extract_key_args(ep: dict) -> dict:
    """Extract navigation-relevant arguments from tool calls."""
    key_args: dict = {}
    tool_calls = ep.get("tool_calls", [])

    for tc in tool_calls:
        args = tc.get("args", {})
        tool = tc.get("tool", "")

        if tool == "navigate_to_location":
            if "location_name" in args:
                key_args["location_name"] = args["location_name"]

        elif tool == "relative_move":
            if "distance_x" in args:
                key_args["distance_x"] = args["distance_x"]
            if "distance_y" in args:
                key_args["distance_y"] = args["distance_y"]

        elif tool in ("save_location", "delete_location"):
            if "name" in args:
                key_args["location_name"] = args["name"]
            elif "location_name" in args:
                key_args["location_name"] = args["location_name"]

        elif tool == "rotate":
            if "angle_degrees" in args:
                key_args["angle_degrees"] = args["angle_degrees"]

    return key_args



def summarise_episode(ep: dict) -> dict:
    """Convert one full episode dict into a compact summary dict."""
    # Deduplicate tool list while preserving order
    seen: list[str] = []
    for tc in ep.get("tool_calls", []):
        if tc["tool"] not in seen:
            seen.append(tc["tool"])

    key_args = _extract_key_args(ep)
    task     = _classify_task(ep.get("query", ""))

    entry: dict = {
        "time":         ep.get("timestamp_start", "")[:19],
        "duration_s":   ep.get("duration_s"),
        "query":        ep.get("query", ""),
        "task_type":    task,
        "outcome":      ep.get("outcome", "unknown"),
        "outcome_fact": _extract_outcome_fact(ep),
        "tools":        seen,
    }
    if key_args:
        entry["key_args"] = key_args
    if ep.get("error"):
        entry["error"] = ep["error"]
    return entry


def summarise_session(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = {
        "session_id":    data.get("session_id", path.stem),
        "created_at":    data.get("created_at", ""),
        "source_file":   str(path),
        "episode_count": len(data.get("episodes", [])),
    }
    episodes = [summarise_episode(ep) for ep in data.get("episodes", [])]
    return meta, episodes



def derive_world_state(episodes: list[dict]) -> dict:
    """
    Infer the current navigation world state from the ordered episode list.

    Tracks:
      - current_pose: last known (x, y, yaw_degrees) from get_current_pose
      - last_location: name of the last successfully navigated-to location
      - known_locations: all location names ever successfully navigated to or saved
      - notes: any extra context
    """
    state: dict = {
        "last_updated":   "",
        "current_pose":   {"x": None, "y": None, "yaw_degrees": None},
        "last_location":  None,
        "known_locations": [],
        "notes":          [],
    }

    seen_locations: set[str] = set()

    # Forward pass — collect all known locations from successful navigate/save episodes
    for ep in episodes:
        task  = ep.get("task_type")
        kargs = ep.get("key_args", {})
        loc   = kargs.get("location_name", "")

        if loc and ep["outcome"] == "success" and task in ("navigate", "home", "save_location"):
            seen_locations.add(loc)

    # Reverse pass — most recent episode wins for pose and last_location
    for ep in reversed(episodes):
        if not state["last_updated"]:
            state["last_updated"] = ep.get("time", "")

        task  = ep.get("task_type")
        kargs = ep.get("key_args", {})

        # Extract pose from get_current_pose tool output (most recent successful nav)
        if state["current_pose"]["x"] is None and ep["outcome"] == "success":
            for tc in ep.get("tool_calls", []) if False else []:
                pass  # handled below via key_args fallback

            # Pull pose directly from tool call outputs in the episode
            if hasattr(ep, "_raw_tool_calls"):
                for tc in ep["_raw_tool_calls"]:
                    if tc.get("tool") == "get_current_pose":
                        try:
                            out = json.loads(tc.get("output", "{}"))
                            if out.get("success"):
                                state["current_pose"] = {
                                    "x":           out.get("x"),
                                    "y":           out.get("y"),
                                    "yaw_degrees": out.get("yaw_degrees"),
                                }
                        except (json.JSONDecodeError, TypeError):
                            pass

        # Last navigated location
        if state["last_location"] is None and ep["outcome"] == "success":
            if task in ("navigate", "home") and kargs.get("location_name"):
                state["last_location"] = kargs["location_name"]

        if state["last_location"] is not None and state["current_pose"]["x"] is not None:
            break

    state["known_locations"] = sorted(seen_locations)
    return state


def derive_world_state_from_raw(all_raw_episodes: list[dict]) -> dict:
    """
    Richer world state extraction that reads tool call outputs directly
    from the raw session episode dicts
    """
    state: dict = {
        "last_updated":    "",
        "current_pose":    {"x": None, "y": None, "yaw_degrees": None},
        "last_location":   None,
        "known_locations": [],
        "notes":           [],
    }

    seen_locations: set[str] = set()

    for ep in all_raw_episodes:
        task  = _classify_task(ep.get("query", ""))
        loc   = ""
        for tc in ep.get("tool_calls", []):
            if tc.get("tool") == "navigate_to_location":
                loc = tc.get("args", {}).get("location_name", "")
        if loc and ep.get("outcome") == "success" and task in ("navigate", "home"):
            seen_locations.add(loc)
        # save_location
        for tc in ep.get("tool_calls", []):
            if tc.get("tool") == "save_location" and ep.get("outcome") == "success":
                name = tc.get("args", {}).get("name", tc.get("args", {}).get("location_name", ""))
                if name:
                    seen_locations.add(name)
            if tc.get("tool") == "delete_location" and ep.get("outcome") == "success":
                name = tc.get("args", {}).get("name", tc.get("args", {}).get("location_name", ""))
                seen_locations.discard(name)

    # Reverse pass for current pose and last location
    for ep in reversed(all_raw_episodes):
        if not state["last_updated"]:
            state["last_updated"] = ep.get("timestamp_start", ep.get("time", ""))[:19]

        if state["current_pose"]["x"] is None and ep.get("outcome") == "success":
            for tc in ep.get("tool_calls", []):
                if tc.get("tool") == "get_current_pose":
                    try:
                        out = json.loads(tc.get("output", "{}"))
                        if out.get("success"):
                            state["current_pose"] = {
                                "x":           out.get("x"),
                                "y":           out.get("y"),
                                "yaw_degrees": out.get("yaw_degrees"),
                            }
                    except (json.JSONDecodeError, TypeError):
                        pass

        if state["last_location"] is None and ep.get("outcome") == "success":
            task = _classify_task(ep.get("query", ""))
            if task in ("navigate", "home"):
                for tc in ep.get("tool_calls", []):
                    if tc.get("tool") == "navigate_to_location":
                        loc = tc.get("args", {}).get("location_name", "")
                        if loc:
                            state["last_location"] = loc
                            break

        if state["last_location"] is not None and state["current_pose"]["x"] is not None:
            break

    state["known_locations"] = sorted(seen_locations)
    return state


def mine_procedures(episodes: list[dict]) -> dict:
    """
    Derive canonical tool-call sequences from successful episodes
    grouped by task type.
    """
    from collections import Counter

    buckets: dict[str, list[tuple[str, ...]]] = {}
    for ep in episodes:
        if ep["outcome"] != "success":
            continue
        task = ep.get("task_type", "other")
        seq  = tuple(ep.get("tools", []))
        buckets.setdefault(task, []).append(seq)

    procedures: dict = {}
    for task, seqs in buckets.items():
        if not seqs:
            continue
        modal_seq, count = Counter(seqs).most_common(1)[0]
        durations = [
            ep["duration_s"] for ep in episodes
            if ep.get("task_type") == task
            and ep["outcome"] == "success"
            and ep["duration_s"] is not None
        ]
        avg_dur = round(sum(durations) / len(durations), 1) if durations else None

        procedures[task] = {
            "tool_sequence":  list(modal_seq),
            "observed_count": len(seqs),
            "modal_count":    count,
            "avg_duration_s": avg_dur,
        }

    return procedures



def build_summary(session_files: list[Path], top_k: int | None = None) -> dict:
    all_raw_episodes: list[dict] = []
    all_episodes:     list[dict] = []

    for path in sorted(session_files):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw_eps = data.get("episodes", [])
        all_raw_episodes.extend(raw_eps)
        _, summarised = summarise_session(path)
        all_episodes.extend(summarised)

    # Sort chronologically
    all_episodes.sort(key=lambda e: e["time"])
    all_raw_episodes.sort(key=lambda e: e.get("timestamp_start", ""))

    if not all_episodes:
        return {}

    world_state = derive_world_state_from_raw(all_raw_episodes)
    procedures  = mine_procedures(all_episodes)

    meaningful = [ep for ep in all_episodes if not _is_trivial(ep["query"])]

    if top_k and len(meaningful) > top_k:
        meaningful = meaningful[-top_k:]

    success  = sum(1 for e in all_episodes if e["outcome"] == "success")
    errors   = sum(1 for e in all_episodes if e["outcome"] == "error")
    avg_dur  = (sum(e["duration_s"] or 0 for e in all_episodes) / len(all_episodes)
                if all_episodes else 0)

    tool_freq: dict[str, int] = {}
    for ep in all_episodes:
        for t in ep["tools"]:
            tool_freq[t] = tool_freq.get(t, 0) + 1
    tool_freq = dict(sorted(tool_freq.items(), key=lambda x: -x[1]))

    task_freq: dict[str, int] = {}
    for ep in all_episodes:
        tt = ep.get("task_type", "other")
        task_freq[tt] = task_freq.get(tt, 0) + 1

    return {
        "generated_at": datetime.now().isoformat()[:19],
        "world_state":      world_state,
        "procedures":       procedures,
        "recent_episodes":  meaningful,
        "stats": {
            "total_episodes":          len(all_episodes),
            "meaningful_episodes":     len(meaningful),
            "trivial_episodes_pruned": len(all_episodes) - len(meaningful),
            "successful":              success,
            "errors":                  errors,
            "average_duration_s":      round(avg_dur, 2),
            "task_frequency":          task_freq,
            "tool_frequency":          tool_freq,
        },
    }



def to_text(summary: dict) -> str:
    ws = summary.get("world_state", {})
    pose = ws.get("current_pose", {})
    pose_str = (f"x={pose.get('x')}, y={pose.get('y')}, yaw={pose.get('yaw_degrees')}°"
                if pose.get("x") is not None else "unknown")
    lines = [
        f"# Agent Memory — generated {summary.get('generated_at', '')}",
        f"# pose={pose_str}",
        f"# last_location={ws.get('last_location', '?')}",
        f"# known_locations: {', '.join(ws.get('known_locations', [])) or '—'}",
        "",
        "# PROCEDURES",
    ]
    for task, proc in summary.get("procedures", {}).items():
        lines.append(f"  {task}: {' → '.join(proc['tool_sequence'])}  "
                     f"(n={proc['observed_count']}, avg {proc['avg_duration_s']}s)")
    lines.append("")
    lines.append("# RECENT MEANINGFUL EPISODES")
    for ep in summary.get("recent_episodes", []):
        status = "✓" if ep["outcome"] == "success" else ("✗" if ep["outcome"] == "error" else "?")
        dur    = f"{ep['duration_s']:.1f}s" if ep["duration_s"] is not None else "?"
        lines.append(
            f"[{ep['time']}] {status} {dur} [{ep['task_type']}] "
            f"Q: {ep['query']} | FACT: {ep['outcome_fact']}"
        )
    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(description="Summarise episodic memory session files.")
    parser.add_argument("--dir",   default="memory", help="Directory containing session JSON files")
    parser.add_argument("--files", nargs="+",        help="Explicit list of session JSON files")
    parser.add_argument("--out",   default="memory.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the N most recent meaningful episodes")
    parser.add_argument("--txt",   action="store_true",
                        help="Also write a compact .txt version")
    args = parser.parse_args()

    if args.files:
        session_files = [Path(f) for f in args.files]
    else:
        session_files = [p for p in Path(args.dir).glob("*.json")
                         if not p.name.startswith("memory")]

    if not session_files:
        print("No session files found.")
        return

    print(f"Processing {len(session_files)} session file(s)…")
    summary = build_summary(session_files, top_k=args.top_k)

    if not summary:
        print("No episodes found.")
        return

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON summary → {out_path}  ({out_path.stat().st_size // 1024} KB)")

    if args.txt:
        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text(to_text(summary), encoding="utf-8")
        print(f"Text summary → {txt_path}")

    s  = summary["stats"]
    ws = summary["world_state"]
    pose = ws.get("current_pose", {})
    print(f"\nStats: {s['total_episodes']} total episodes | "
          f"{s['meaningful_episodes']} meaningful | "
          f"{s['trivial_episodes_pruned']} trivial pruned")
    print(f"       {s['successful']} success | {s['errors']} errors | "
          f"avg {s['average_duration_s']}s/episode")
    print(f"Last location: {ws.get('last_location', 'unknown')} | "
          f"Pose: x={pose.get('x')}, y={pose.get('y')}, yaw={pose.get('yaw_degrees')}°")
    print(f"Known locations: {', '.join(ws.get('known_locations', [])) or '—'}")
    print("Procedures mined:", ", ".join(summary["procedures"].keys()))


if __name__ == "__main__":
    main()