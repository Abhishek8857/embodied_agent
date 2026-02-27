"""
memory_summarizer.py
--------------------
Reads multiple episodic memory session JSON files and produces a single
compact summary file optimised for injecting into an agent's context window.

For each episode the full tool I/O (which can be huge) is stripped down to
just the essential signal:
  - what was asked
  - which tools were called (names + key args only, no raw sensor blobs)
  - whether it succeeded
  - how long it took
  - the agent's own final summary sentence

The output is a lean JSON file:
  memory_summary.json
and an optional ultra-compact plain-text version:
  memory_summary.txt   (one line per episode, good for small context windows)

Usage
-----
    python memory_summarizer.py                          # reads *.json in ./memory/
    python memory_summarizer.py --dir /path/to/sessions  # custom folder
    python memory_summarizer.py --files s1.json s2.json  # specific files
    python memory_summarizer.py --top-k 20               # keep only 20 most recent episodes
    python memory_summarizer.py --txt                    # also write .txt version
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

# Tool output fields that are too large / not useful for memory
_BLOCKLIST_KEYS = {
    "mask", "mask_array", "feedback", "sent_data",
    "rgb_shape", "depth_shape", "segmap_shape", "segmap_unique",
    "visualizations", "individual_masks",
}

# For these tools keep only these specific fields from the output
_SLIM_OUTPUT_KEYS: dict[str, list[str]] = {
    "segment_objects":          ["count", "objects.label", "objects.grasp_center_3d"],
    "get_latest_grasp_pose":    ["success", "x", "y", "z"],
    "get_place_pose":           ["success", "x", "y", "z", "object_label"],
    "grasp_object":             ["success", "final_status"],
    "place_object":             ["success", "final_status"],
    "move_to_home_pose":        ["success", "final_status", "elapsed_s"],
    "move_to_pose":             ["success", "final_status"],
    "get_current_joint_states": ["success"],
    "capture_rgbd":             ["success", "path"],
    "capture_only_rgb_image":   ["success", "path"],
    "save_segmentation_for_graspnet": ["success", "num_objects"],
    "describe_what_you_see":    ["__text_truncate_200__"],
}


def _slim_output(tool_name: str, raw_output: str) -> str:
    """Return a compact representation of a tool's output string."""
    spec = _SLIM_OUTPUT_KEYS.get(tool_name)

    # --- plain-text output (describe_what_you_see etc.) -----------------------
    if spec == ["__text_truncate_200__"]:
        text = raw_output.strip()
        return text[:200] + ("…" if len(text) > 200 else "")

    # --- try to parse as JSON / dict -----------------------------------------
    try:
        # Python repr dicts use single quotes; json needs double
        obj = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        try:
            # Strip numpy array(...) calls before eval to avoid truncated repr leaking through
            cleaned = re.sub(r"array\([^)]*\)", '"<array>"', raw_output, flags=re.DOTALL)
            obj = eval(cleaned, {"__builtins__": {}},   # safe-ish eval for repr dicts
                       {"False": False, "True": True, "None": None})
        except Exception:
            # fallback: just truncate
            return raw_output[:200]

    if not isinstance(obj, dict):
        return str(obj)[:200]

    # Remove big blocklisted keys
    obj = {k: v for k, v in obj.items() if k not in _BLOCKLIST_KEYS}

    if spec:
        slim = {}
        for key in spec:
            if "." in key:
                # e.g. "objects.label" → extract from list of dicts
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
    """Strip mask / array data from tool args too (usually clean, but just in case)."""
    return {k: v for k, v in args.items() if k not in _BLOCKLIST_KEYS and not isinstance(v, list) or k == "query"}


def _first_sentence(text: str) -> str:
    """Return roughly the first sentence of a string."""
    if not text:
        return ""
    text = text.strip().splitlines()[0]  # first non-empty line
    m = re.search(r'[.!?]', text)
    return text[: m.start() + 1] if m else text[:150]


# ── Core summarisation ─────────────────────────────────────────────────────────

def summarise_episode(ep: dict) -> dict:
    """Convert one full episode dict into a compact summary dict."""
    # Deduplicate tools while preserving order
    seen: list[str] = []
    for tc in ep.get("tool_calls", []):
        if tc["tool"] not in seen:
            seen.append(tc["tool"])

    # Extract only the key args that add meaning (segment targets, place target)
    key_args: dict = {}
    for tc in ep.get("tool_calls", []):
        args = tc.get("args", {})
        if "query" in args:
            key_args.setdefault("segmented", [])
            if args["query"] not in key_args["segmented"]:
                key_args["segmented"].append(args["query"])
        if "target_object_label" in args:
            key_args["placed_on"] = args["target_object_label"]

    entry: dict = {
        "time":       ep.get("timestamp_start", "")[:19],
        "duration_s": ep.get("duration_s"),
        "query":      ep.get("query", ""),
        "outcome":    ep.get("outcome", "unknown"),
        "response":   _first_sentence(ep.get("final_response", "")),
        "tools":      seen,
    }
    if key_args:
        entry["key_args"] = key_args
    if ep.get("error"):
        entry["error"] = ep["error"]
    return entry


def summarise_session(path: Path) -> tuple[dict, list[dict]]:
    """Load one session file and return (session_meta, [summarised_episodes])."""
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


def build_summary(session_files: list[Path], top_k: int | None = None) -> dict:
    """Aggregate multiple session files into one summary structure."""
    all_episodes = []
    session_count = 0

    for path in sorted(session_files):
        meta, episodes = summarise_session(path)
        if not episodes:
            continue  # skip empty sessions entirely
        session_count += 1
        all_episodes.extend(episodes)

    # Sort by time, newest last
    all_episodes.sort(key=lambda e: e["time"])

    # Keep only the most recent top_k episodes if requested
    if top_k and len(all_episodes) > top_k:
        all_episodes = all_episodes[-top_k:]

    success = sum(1 for e in all_episodes if e["outcome"] == "success")
    errors  = sum(1 for e in all_episodes if e["outcome"] == "error")
    avg_dur = (sum(e["duration_s"] or 0 for e in all_episodes) / len(all_episodes)
               if all_episodes else 0)

    # Tool frequency table
    tool_freq: dict[str, int] = {}
    for ep in all_episodes:
        for t in ep["tools"]:
            tool_freq[t] = tool_freq.get(t, 0) + 1
    tool_freq = dict(sorted(tool_freq.items(), key=lambda x: -x[1]))

    return {
        "generated_at": datetime.now().isoformat()[:19],
        "sessions": session_count,
        "stats": {
            "total_episodes":     len(all_episodes),
            "successful":         success,
            "errors":             errors,
            "average_duration_s": round(avg_dur, 2),
            "tool_frequency":     tool_freq,
        },
        "episodes": all_episodes,
    }


# ── Text formatter (ultra-compact) ─────────────────────────────────────────────

def to_text(summary: dict) -> str:
    lines = [
        f"# Agent Episode Summary — generated {summary['generated_at']}",
        f"# {summary['sessions']} session(s) | "
        f"{summary['stats']['total_episodes']} episodes | "
        f"{summary['stats']['successful']} OK / {summary['stats']['errors']} errors",
        "",
    ]
    for ep in summary["episodes"]:
        status = "✓" if ep["outcome"] == "success" else ("✗" if ep["outcome"] == "error" else "?")
        tools  = " → ".join(ep["tools"]) if ep["tools"] else "—"
        dur    = f"{ep['duration_s']:.1f}s" if ep["duration_s"] is not None else "?"
        lines.append(
            f"[{ep['time']}] {status} {dur} | Q: {ep['query']} | "
            f"TOOLS: {tools} | A: {ep['response']}"
        )
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarise episodic memory session files.")
    parser.add_argument("--dir",   default="memory", help="Directory containing session JSON files")
    parser.add_argument("--files", nargs="+",        help="Explicit list of session JSON files")
    parser.add_argument("--out",   default="memory_summary.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the N most recent episodes")
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

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON summary → {out_path}  ({out_path.stat().st_size // 1024} KB)")

    if args.txt:
        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text(to_text(summary), encoding="utf-8")
        print(f"Text summary → {txt_path}")

    s = summary["stats"]
    print(f"\nStats: {s['total_episodes']} episodes | "
          f"{s['successful']} success | {s['errors']} errors | "
          f"avg {s['average_duration_s']}s/episode")
    print("Top tools:", ", ".join(list(s["tool_frequency"].keys())[:5]))


if __name__ == "__main__":
    main()