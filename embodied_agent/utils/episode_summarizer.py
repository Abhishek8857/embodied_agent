"""
memory_summarizer.py  (v2)
--------------------------
Reads multiple episodic memory session JSON files and produces a two-layer
memory file optimised for injecting into an agent's context window.

Layer 1 – High-signal, injected every call (small):
  world_state   current scene facts derived from the most recent episodes
  procedures    canonical tool chains mined from successful episodes

Layer 2 – Recent context, injected for novel/recovery situations:
  recent_episodes  last N *meaningful* episodes (trivial homing stripped out)
  stats            tool frequency, success rates

Output
------
  memory_summary.json   full two-layer structure
  memory_summary.txt    ultra-compact one-liner-per-episode version (--txt)

Usage
-----
    python memory_summarizer.py                          # reads *.json in ./memory/
    python memory_summarizer.py --dir /path/to/sessions  # custom folder
    python memory_summarizer.py --files s1.json s2.json  # specific files
    python memory_summarizer.py --top-k 20               # keep only 20 recent episodes
    python memory_summarizer.py --txt                    # also write .txt version
"""

import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

_BLOCKLIST_KEYS = {
    "mask", "mask_array", "feedback", "sent_data",
    "rgb_shape", "depth_shape", "segmap_shape", "segmap_unique",
    "visualizations", "individual_masks",
}

_SLIM_OUTPUT_KEYS: dict[str, list[str]] = {
    "segment_objects":                    ["count", "objects.label", "objects.grasp_center_3d"],
    "get_latest_grasp_pose":              ["success", "x", "y", "z"],
    "get_place_pose":                     ["success", "x", "y", "z", "object_label"],
    "grasp_object":                       ["success", "final_status"],
    "place_object":                       ["success", "final_status"],
    "move_to_home_pose":                  ["success", "final_status", "elapsed_s"],
    "move_to_pose":                       ["success", "final_status"],
    "get_current_joint_states":           ["success"],
    "capture_rgbd":                       ["success", "path"],
    "capture_only_rgb_image":             ["success", "path"],
    "save_segmentation_for_graspnet":     ["success", "num_objects"],
    "describe_what_you_see":              ["__text_truncate_200__"],
    "describe_environment":               ["__text_truncate_200__"],
}

# Queries that are operationally trivial and low-signal for future planning.
# These are kept in stats but excluded from recent_episodes injected to the agent.
_TRIVIAL_QUERY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*go\s+home\s*$",
        r"^\s*home\s*$",
        r"^\s*return\s+(to\s+)?home\s*$",
        r"^\s*open\s+gripper\s*$",
        r"^\s*close\s+gripper\s*$",
    ]
]

# Task-type classifiers  (query → procedure key)
_TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pick.*place|place.*on|stack|put.*on", re.IGNORECASE), "pick_and_place"),
    (re.compile(r"pick\s+up|grasp|grab",               re.IGNORECASE), "pick_only"),
    (re.compile(r"go\s+home|home\s+pose|return\s+home", re.IGNORECASE), "home"),
    (re.compile(r"open\s+gripper",                      re.IGNORECASE), "open_gripper"),
    (re.compile(r"close\s+gripper",                     re.IGNORECASE), "close_gripper"),
    (re.compile(r"describe|look|see|observe|what.+see", re.IGNORECASE), "observe"),
]


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _classify_task(query: str) -> str:
    for pattern, label in _TASK_PATTERNS:
        if pattern.search(query):
            return label
    return "other"


def _is_trivial(query: str) -> bool:
    return any(p.fullmatch(query) for p in _TRIVIAL_QUERY_PATTERNS)


def _slim_output(tool_name: str, raw_output: str) -> str:
    spec = _SLIM_OUTPUT_KEYS.get(tool_name)

    if spec == ["__text_truncate_200__"]:
        text = raw_output.strip()
        return text[:200] + ("…" if len(text) > 200 else "")

    try:
        obj = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        try:
            cleaned = re.sub(r"array\([^)]*\)", '"<array>"', raw_output, flags=re.DOTALL)
            obj = ast.literal_eval(cleaned)
        except Exception:
            return raw_output[:200]

    if not isinstance(obj, dict):
        return str(obj)[:200]

    obj = {k: v for k, v in obj.items() if k not in _BLOCKLIST_KEYS}

    if spec:
        slim = {}
        for key in spec:
            if "." in key:
                parent, child = key.split(".", 1)
                parent_val = obj.get(parent, [])
                if isinstance(parent_val, list):
                    slim[key] = [item.get(child) for item in parent_val if isinstance(item, dict)]
                else:
                    slim[key] = parent_val
            elif key in obj:
                slim[key] = obj[key]
        return json.dumps(slim)

    return json.dumps(obj)


def _slim_args(args: dict) -> dict:
    return {k: v for k, v in args.items()
            if k not in _BLOCKLIST_KEYS and (not isinstance(v, list) or k == "query")}


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    text = text.strip().splitlines()[0]
    m = re.search(r'[.!?]', text)
    return text[: m.start() + 1] if m else text[:150]


def _extract_outcome_fact(ep: dict) -> str:
    """
    Build a terse factual outcome sentence from the structured episode data
    rather than parroting the agent's own verbose narration.

    Priority: use verification lines from final_response if present,
    otherwise synthesise from key_args and outcome.
    """
    query    = ep.get("query", "")
    outcome  = ep.get("outcome", "unknown")
    response = ep.get("final_response", "")
    task     = _classify_task(query)

    # Try to extract a "Result: X" line from the response
    result_match = re.search(r"Result:\s*([^\n]+)", response, re.IGNORECASE)
    result_str   = result_match.group(1).strip() if result_match else outcome.upper()

    if task == "pick_and_place":
        key_args = _extract_key_args(ep)
        placements = key_args.get("placements", [])
        retries    = key_args.get("retries", 0)
        retry_s    = f" ({retries} retr{'ies' if retries != 1 else 'y'})" if retries else ""
        if len(placements) > 1:
            pairs = ", ".join(f"{p['picked']} → {p['placed_on']}" for p in placements)
            return f"Multi-place{retry_s}: {pairs}. {result_str}."
        obj    = key_args.get("picked_object", "object")
        target = key_args.get("placed_on", "target")
        return f"{obj} picked and placed on {target}{retry_s}. {result_str}."

    if task == "pick_only":
        key_args = _extract_key_args(ep)
        obj    = key_args.get("picked_object", "object")
        retries = key_args.get("retries", 0)
        retry_s = f" ({retries} retr{'ies' if retries != 1 else 'y'})" if retries else ""
        return f"{obj} grasped{retry_s}. {result_str}."

    if task == "home":
        return f"Arm returned to home pose. {result_str}."

    if task in ("open_gripper", "close_gripper"):
        action = "opened" if task == "open_gripper" else "closed"
        return f"Gripper {action}. {result_str}."

    # Generic fallback
    return f"{query.strip().capitalize()}. {result_str}."


def _extract_seen_objects(ep: dict) -> list[str]:
    """
    Extract object labels seen during an episode from segment_objects outputs.
    These are the ground-truth labels used by the agent, so they're more
    reliable for known_objects than free-text from describe_environment.
    """
    labels: list[str] = []
    for tc in ep.get("tool_calls", []):
        if tc.get("tool") != "segment_objects":
            continue
        raw = tc.get("output", "")
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            try:
                cleaned = re.sub(r"array\([^)]*\)", '"<array>"', raw, flags=re.DOTALL)
                obj = ast.literal_eval(cleaned)
            except Exception:
                continue
        for item in obj.get("objects", []):
            label = item.get("label")
            if label and label not in labels:
                labels.append(label)
    return labels


def _extract_key_args(ep: dict) -> dict:
    """
    Extract pick/place pairs and retry count from tool call sequence.

    Uses a state machine to correctly pair each pick_up_object with the
    segment query that preceded it (= what was picked) and the
    get_place_pose target that follows it (= where it was placed).

    This handles multi-placement episodes correctly, where the naive approach
    of taking first-segment / last-target would cross the pairs.

    Returns:
        placements  - list of {picked, placed_on} dicts, one per placement
        picked_object / placed_on - first placement, kept for backward compat
        retries     - number of extra pick_up_object calls beyond the first
    """
    key_args: dict = {}
    tool_calls = ep.get("tool_calls", [])

    placements: list[dict] = []
    candidate_pick: str | None = None   # last segment label before a pick
    current_held:   str | None = None   # object currently in gripper
    current_target: str | None = None   # placement target from get_place_pose
    in_pick_phase = True                # True = scanning for next pick object
    pick_attempts = 0

    for tc in tool_calls:
        tool = tc.get("tool", "")
        args = tc.get("args", {})

        # Parse success from output (default True so unknown = assume ok)
        try:
            out = json.loads(tc.get("output", "{}"))
            success = out.get("success", True)
        except Exception:
            success = True

        if tool == "segment_objects":
            q = args.get("query", "")
            # Non-question segment queries in the pick phase identify the target object
            if in_pick_phase and q and not re.search(r'\?', q):
                candidate_pick = q

        elif tool == "pick_up_object":
            pick_attempts += 1
            if success:
                current_held = candidate_pick
                current_target = None
                in_pick_phase = False   # now scanning for place target

        elif tool == "get_place_pose":
            current_target = args.get("target_object_label")

        elif tool == "place_object":
            if success and current_held:
                placements.append({
                    "picked":    current_held,
                    "placed_on": current_target,
                })
            # Reset for next pick-place cycle regardless of success
            current_held = None
            current_target = None
            candidate_pick = None
            in_pick_phase = True

    if placements:
        key_args["placements"]    = placements
        # Backward-compat single fields (first placement)
        key_args["picked_object"] = placements[0]["picked"]
        key_args["placed_on"]     = placements[0]["placed_on"]

    # Retries = extra pick_up_object calls beyond the expected one-per-placement
    total_picks = len(placements) if placements else 1
    if pick_attempts > total_picks:
        key_args["retries"] = pick_attempts - total_picks

    return key_args


# ── Episode summarisation ──────────────────────────────────────────────────────

def summarise_episode(ep: dict) -> dict:
    """Convert one full episode dict into a compact summary dict."""
    seen: list[str] = []
    for tc in ep.get("tool_calls", []):
        if tc["tool"] not in seen:
            seen.append(tc["tool"])

    key_args     = _extract_key_args(ep)
    task         = _classify_task(ep.get("query", ""))
    seen_objects = _extract_seen_objects(ep)

    # ── ADD THIS BLOCK ──────────────────────────────────────────────────────
    # Pull structured retry data written by episode_recorder.record_retry().
    # _extract_key_args() already counts retries from tool calls (pick attempts),
    # but that only works for pick/place tasks. This covers ALL task types and
    # also captures the failure reason and hint that was used.
    recorder_retries = ep.get("retries", {})
    retry_count = recorder_retries.get("count", 0)

    # Only override if _extract_key_args didn't already find a higher number
    # (the tool-call counter is more precise for pick/place; use whichever is larger)
    if retry_count > key_args.get("retries", 0):
        key_args["retries"] = retry_count

    # Capture the last failure reason for outcome_fact on failed/retried episodes
    retry_attempts = recorder_retries.get("attempts", [])
    last_failure_reason = (
        retry_attempts[-1].get("failure_reason", "") if retry_attempts else ""
    )
    # ── END OF ADDED BLOCK ──────────────────────────────────────────────────

    entry: dict = {
        "time":         ep.get("timestamp_start", "")[:19],
        "duration_s":   ep.get("duration_s"),
        "query":        ep.get("query", ""),
        "task_type":    task,
        "outcome":      ep.get("outcome", "unknown"),
        "outcome_fact": _extract_outcome_fact(ep),
        "tools":        seen,
    }
    if seen_objects:
        entry["seen_objects"] = seen_objects
    if key_args:
        entry["key_args"] = key_args
    if ep.get("error"):
        entry["error"] = ep["error"]

    # ── ADD THIS LINE ───────────────────────────────────────────────────────
    # Store the failure reason so _compute_failure_stats() in memory_context.py
    # can read it as outcome_fact on failed/retried episodes
    if last_failure_reason and entry["outcome"] != "success":
        entry["outcome_fact"] = last_failure_reason
    # ── END ─────────────────────────────────────────────────────────────────

    return entry


def summarise_session(path: Path) -> tuple[dict, list[dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: skipping {path} — {e}")
        return {}, []

    meta = {
        "session_id":    data.get("session_id", path.stem),
        "created_at":    data.get("created_at", ""),
        "source_file":   str(path),
        "episode_count": len(data.get("episodes", [])),
    }
    episodes = [summarise_episode(ep) for ep in data.get("episodes", [])]
    return meta, episodes


# ── World-state extractor ──────────────────────────────────────────────────────

def derive_world_state(episodes: list[dict]) -> dict:
    """
    Infer the current world state from the ordered episode list.

    Placement tracking uses a graph approach:
      - sitting_on[obj] = base  means obj is currently resting on base
      - When obj is picked up it is removed from the graph (it left its base)
      - This correctly handles multiple independent stacks and rebuilds

    The final stacks list contains one entry per independent tower, each
    formatted bottom→top, e.g.:
      [["red cube", "green cube"], ["yellow cube", "blue cube"]]

    known_objects is populated from:
      1. segment_objects outputs (structured, ground-truth labels)
      2. pick/place key_args (placement targets also reveal object names)
    """
    state: dict = {
        "last_updated":  "",
        "arm_pose":      "unknown",
        "gripper":       "unknown",
        "known_objects": [],
        "stacks":        [],   # list of towers, each tower is [bottom, ..., top]
        "notes":         [],
    }

    seen_objects: set[str] = set()

    # sitting_on[obj] = base — tracks where each object currently rests.
    # Forward pass: process episodes in chronological order so later actions
    # overwrite earlier ones (pick removes, place adds).
    sitting_on: dict[str, str] = {}

    for ep in episodes:
        t     = ep.get("task_type")
        kargs = ep.get("key_args", {})

        # Accumulate known objects from segment_objects outputs
        for label in ep.get("seen_objects", []):
            seen_objects.add(label)

        # Also capture placement targets (may not have been segmented directly)
        for p in kargs.get("placements", []):
            if p.get("picked"):
                seen_objects.add(p["picked"])
            if p.get("placed_on"):
                seen_objects.add(p["placed_on"])

        if ep["outcome"] != "success":
            continue

        # Apply each pick→place pair to the sitting_on graph in order
        placements = kargs.get("placements", [])
        if not placements and t in ("pick_and_place", "pick_only"):
            # Fallback for episodes without placements list (e.g. legacy data)
            obj  = kargs.get("picked_object", "")
            onto = kargs.get("placed_on", "")
            if obj:
                placements = [{"picked": obj, "placed_on": onto or None}]

        for p in placements:
            obj  = p.get("picked", "")
            onto = p.get("placed_on")
            if obj:
                sitting_on.pop(obj, None)   # object was lifted off its previous base
            if obj and onto:
                sitting_on[obj] = onto      # object now rests on onto

    # ── Build independent stacks from the sitting_on graph ────────────────────
    # For each object that has nothing sitting on top of it (a "top" object),
    # walk down the sitting_on chain to reconstruct the full tower.
    all_tops = set(sitting_on.keys())          # objects that are ON something
    all_bases = set(sitting_on.values())       # objects that have something ON them
    top_objects = all_tops - all_bases         # objects with nothing on top = tower tops

    stacks = []
    for top in sorted(top_objects):           # sorted for deterministic output
        tower = [top]
        current = top
        visited = {top}
        while current in sitting_on:
            base = sitting_on[current]
            if base in visited:
                break                          # cycle guard (shouldn't happen)
            tower.append(base)
            visited.add(base)
            current = base
        tower.reverse()                        # now bottom → top
        stacks.append(tower)

    # ── Reverse pass for arm / gripper state (most recent wins) ───────────────
    for ep in reversed(episodes):
        if not state["last_updated"]:
            state["last_updated"] = ep.get("time", "")

        t = ep.get("task_type")
        if state["arm_pose"] == "unknown":
            if "move_to_home_pose" in ep.get("tools", []) and ep["outcome"] == "success":
                state["arm_pose"] = "home"

        if state["gripper"] == "unknown":
            if t == "open_gripper" and ep["outcome"] == "success":
                state["gripper"] = "open"
            elif t == "close_gripper" and ep["outcome"] == "success":
                state["gripper"] = "closed"
            elif t == "pick_and_place" and ep["outcome"] == "success":
                state["gripper"] = "open"    # object released after place
            elif t == "pick_only" and ep["outcome"] == "success":
                state["gripper"] = "closed"  # holding object

        if state["arm_pose"] != "unknown" and state["gripper"] != "unknown":
            break

    state["known_objects"] = sorted(seen_objects)
    state["stacks"]        = stacks

    return state


# ── Procedure miner ────────────────────────────────────────────────────────────

def mine_procedures(episodes: list[dict]) -> dict:
    """
    Derive canonical tool-call sequences from successful episodes
    grouped by task type.  For each task type, keep the most common
    (modal) sequence — that's the procedure the agent should follow.
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
        # Modal (most common) sequence
        modal_seq, count = Counter(seqs).most_common(1)[0]
        # Compute an avg duration for this task type
        durations = [
            ep["duration_s"] for ep in episodes
            if ep.get("task_type") == task
            and ep["outcome"] == "success"
            and ep["duration_s"] is not None
        ]
        avg_dur = round(sum(durations) / len(durations), 1) if durations else None

        procedures[task] = {
            "tool_sequence":    list(modal_seq),
            "observed_count":   len(seqs),
            "modal_count":      count,
            "avg_duration_s":   avg_dur,
        }

    return procedures


# ── Main aggregator ────────────────────────────────────────────────────────────

def build_summary(session_files: list[Path], top_k: int | None = None) -> dict:
    all_episodes: list[dict] = []

    for path in sorted(session_files):
        _, episodes = summarise_session(path)
        all_episodes.extend(episodes)

    # Sort chronologically
    all_episodes.sort(key=lambda e: e["time"])

    if not all_episodes:
        return {}

    # ── Layer 1: world state & procedures (derived from ALL episodes) ──────────
    world_state = derive_world_state(all_episodes)
    procedures  = mine_procedures(all_episodes)

    # ── Layer 2: recent meaningful episodes ───────────────────────────────────
    meaningful = [ep for ep in all_episodes if not _is_trivial(ep["query"])]

    if top_k and len(meaningful) > top_k:
        meaningful = meaningful[-top_k:]

    # ── Stats (over all episodes, including trivial) ──────────────────────────
    success  = sum(1 for e in all_episodes if e["outcome"] == "success")
    errors   = sum(1 for e in all_episodes if e["outcome"] == "error")
    valid_durations = [e["duration_s"] for e in all_episodes if e["duration_s"] is not None]
    avg_dur = sum(valid_durations) / len(valid_durations) if valid_durations else 0

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
        # ── Layer 1 ──────────────────────────────────────────────────────────
        "world_state": world_state,
        "procedures":  procedures,
        # ── Layer 2 ──────────────────────────────────────────────────────────
        "recent_episodes": meaningful,
        # ── Meta ─────────────────────────────────────────────────────────────
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


# ── Text formatter ─────────────────────────────────────────────────────────────

def to_text(summary: dict) -> str:
    ws = summary.get("world_state", {})
    lines = [
        f"# Agent Memory — generated {summary.get('generated_at', '')}",
        f"# arm={ws.get('arm_pose','?')}  gripper={ws.get('gripper','?')}",
        f"# stacks: {' | '.join(' > '.join(t) for t in ws.get('stacks', [])) or '—'}",
        f"# known objects: {', '.join(ws.get('known_objects', [])) or '—'}",
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarise episodic memory session files.")
    parser.add_argument("--dir",   default="memory", help="Directory containing session JSON files")
    parser.add_argument("--files", nargs="+",        help="Explicit list of session JSON files")
    parser.add_argument("--out",   default="memory_summary.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the N most recent meaningful episodes")
    parser.add_argument("--txt",   action="store_true",
                        help="Also write a compact .txt version")
    args = parser.parse_args()

    if args.files:
        session_files = [Path(f) for f in args.files]
    else:
        session_files = [p for p in Path(args.dir).glob("*.json")
                         if not p.name.startswith("memory_summary")]

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

    s = summary["stats"]
    ws = summary["world_state"]
    print(f"\nStats: {s['total_episodes']} total episodes | "
          f"{s['meaningful_episodes']} meaningful | "
          f"{s['trivial_episodes_pruned']} trivial pruned")
    print(f"       {s['successful']} success | {s['errors']} errors | "
          f"avg {s['average_duration_s']}s/episode")
    print(f"World state: arm={ws['arm_pose']} | gripper={ws['gripper']}")
    if ws.get("stacks"):
        for i, tower in enumerate(ws["stacks"], 1):
            print(f"Stack {i} (bottom→top): {' > '.join(tower)}")
    print("Procedures mined:", ", ".join(summary["procedures"].keys()))


if __name__ == "__main__":
    main()