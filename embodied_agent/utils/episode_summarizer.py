"""
memory_summarizer.py  (v3 - hardened)
--------------------------------------
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
    "move_to_named_pose":                 ["success", "final_status", "elapsed_s"],
    "move_to_pose":                       ["success", "final_status"],
    "get_current_joint_states":           ["success"],
    "capture_rgbd":                       ["success", "path"],
    "capture_only_rgb_image":             ["success", "path"],
    "save_segmentation_for_graspnet":     ["success", "num_objects"],
    "describe_what_you_see":              ["__text_truncate_200__"],
    "describe_environment":               ["__text_truncate_200__"],
}

# Queries that are operationally trivial and low-signal for future planning.
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

_GRASP_VERIFY_PATTERN = re.compile(
    r"\b(grasped|grasping|assess\s+whether|likely\s+grasped|being\s+held|confirm\s+if"
    r"|is\s+(it\s+)?grasped|visual\s+assessment)\b",
    re.IGNORECASE,
)

# Task-type classifiers (query → procedure key)
_TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pick.*place|place.*on|stack|put.*on",              re.IGNORECASE), "pick_and_place"),
    (re.compile(r"pick\s+up|grasp|grab",                             re.IGNORECASE), "pick_only"),
    (re.compile(r"go\s+home|home\s+pose|return\s+home",              re.IGNORECASE), "home"),
    (re.compile(r"open\s+gripper",                                   re.IGNORECASE), "open_gripper"),
    (re.compile(r"close\s+gripper",                                  re.IGNORECASE), "close_gripper"),
    (re.compile(r"describe|look|see|observe|what.+see",              re.IGNORECASE), "observe"),
    (re.compile(r"\b(go\s+to|move\s+to|navigate\s+to)\s+\w",        re.IGNORECASE), "named_pose_move"),
    (re.compile(r"\bmove\s+(forward|backward|left|right|up|down)\b", re.IGNORECASE), "cartesian_move"),
    (re.compile(r"\bsave\s+(this|current|the)?\s*(pose|position)\b", re.IGNORECASE), "save_pose"),
    (re.compile(r"\bdelete\s+\w+\s+pose\b|\brename\s+\w+",           re.IGNORECASE), "pose_management"),
]

_SHAPE_SYNONYMS: dict[str, str] = {
    "block": "cube",
}

_NVIDIA_ALIAS_RE = re.compile(
    r"nvidia|black\s+rectangular\s+box",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _normalise_object_label(label: str) -> str:
    """Return a canonical, lowercase form of an object label."""
    if not label or not isinstance(label, str):
        return ""

    norm = " ".join(label.lower().split())
    norm = _ARTICLE_RE.sub("", norm).strip()

    if _NVIDIA_ALIAS_RE.search(norm):
        return "nvidia box"

    for synonym, canonical in _SHAPE_SYNONYMS.items():
        norm = re.sub(rf"\b{re.escape(synonym)}\b", canonical, norm)

    return norm


def _classify_task(query: str) -> str:
    if not query or not isinstance(query, str):
        return "other"
    for pattern, label in _TASK_PATTERNS:
        if pattern.search(query):
            return label
    return "other"


def _is_trivial(query: str) -> bool:
    if not query or not isinstance(query, str) or not query.strip():
        return True
    return any(p.fullmatch(query.strip()) for p in _TRIVIAL_QUERY_PATTERNS)


def _slim_output(tool_name: str, raw_output: str) -> str:
    spec = _SLIM_OUTPUT_KEYS.get(tool_name)

    if spec == ["__text_truncate_200__"]:
        text = str(raw_output).strip()
        return text[:200] + ("…" if len(text) > 200 else "")

    try:
        obj = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        try:
            cleaned = re.sub(r"array\([^)]*\)", '"<array>"', str(raw_output), flags=re.DOTALL)
            obj = ast.literal_eval(cleaned)
        except Exception:
            return str(raw_output)[:200]

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
    if not isinstance(args, dict):
        return {}
    return {k: v for k, v in args.items()
            if k not in _BLOCKLIST_KEYS and (not isinstance(v, list) or k == "query")}


def _first_sentence(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip().splitlines()[0]
    m = re.search(r'[.!?]', text)
    return text[: m.start() + 1] if m else text[:150]


def _extract_outcome_fact(ep: dict) -> str:
    if not isinstance(ep, dict):
        return ""

    query    = ep.get("query", "")
    outcome  = ep.get("outcome", "unknown")
    response = ep.get("final_response", "")
    task     = _classify_task(query)

    result_match = re.search(r"Result:\s*([^\n]+)", response, re.IGNORECASE)
    result_str   = result_match.group(1).strip() if result_match else str(outcome).upper()

    recorder_retries = ep.get("retries", {})
    if isinstance(recorder_retries, dict):
        retry_count = recorder_retries.get("count", 0)
    elif isinstance(recorder_retries, int):
        retry_count = recorder_retries
    else:
        retry_count = 0

    key_args_retries = ep.get("_key_args_retries", 0)
    total_retries = max(retry_count, key_args_retries)
    retry_s = (
        f" (succeeded after {total_retries} retr{'ies' if total_retries != 1 else 'y'})"
        if total_retries > 0 and outcome == "success"
        else ""
    )

    if task == "pick_and_place":
        key_args = _extract_key_args(ep)
        placements = key_args.get("placements", [])
        if len(placements) > 1:
            pairs = ", ".join(f"{p['picked']} → {p['placed_on']}" for p in placements)
            return f"Multi-place{retry_s}: {pairs}. {result_str}."
        obj    = key_args.get("picked_object", "object")
        target = key_args.get("placed_on", "target")
        return f"{obj} picked and placed on {target}{retry_s}. {result_str}."

    if task == "pick_only":
        key_args = _extract_key_args(ep)
        obj = key_args.get("picked_object", "object")
        base = f"{obj} grasped{retry_s}. {result_str}."
        grasp_assessment = ep.get("_grasp_assessment", "")
        if grasp_assessment:
            first_line = grasp_assessment.splitlines()[0]
            base += f" Visual check: {first_line}"
        return base

    if task == "home":
        return f"Arm returned to home pose{retry_s}. {result_str}."

    if task in ("open_gripper", "close_gripper"):
        action = "opened" if task == "open_gripper" else "closed"
        return f"Gripper {action}{retry_s}. {result_str}."

    if task == "named_pose_move":
        m = re.search(r"\b(?:go\s+to|move\s+to|navigate\s+to)\s+(\w[\w\s]*)", query, re.IGNORECASE)
        pose_name = m.group(1).strip() if m else "named pose"
        return f"Moved to '{pose_name}'{retry_s}. {result_str}."

    if task == "cartesian_move":
        return f"{query.strip().capitalize()}{retry_s}. {result_str}."

    if task == "save_pose":
        m = re.search(r"as\s+([\w\s]+?)(?:\s*$|\s*\.)", query, re.IGNORECASE)
        pose_name = m.group(1).strip() if m else "pose"
        return f"Saved pose '{pose_name}'{retry_s}. {result_str}."

    if task == "pose_management":
        return f"{query.strip().capitalize()}{retry_s}. {result_str}."

    return f"{query.strip().capitalize()}{retry_s}. {result_str}."


def _extract_seen_objects(ep: dict) -> list[str]:
    labels: list[str] = []
    tool_calls = ep.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return labels

    for tc in tool_calls:
        if not isinstance(tc, dict) or tc.get("tool") != "segment_objects":
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
        if isinstance(obj, dict):
            for item in obj.get("objects", []):
                if isinstance(item, dict):
                    label = _normalise_object_label(item.get("label", ""))
                    if label and label not in labels:
                        labels.append(label)
    return labels


def _extract_scene_object_labels(scene_text: str) -> list[str]:
    labels: list[str] = []
    if not scene_text or not isinstance(scene_text, str):
        return labels

    list_item_re = re.compile(
        r"^\s*\d+\.\s+[Aa]n?\s+(.+?)(?:\s+(?:located|positioned|placed|sitting|resting)\b|[,.]|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in list_item_re.finditer(scene_text):
        label = _normalise_object_label(m.group(1).strip())
        if label and label not in labels:
            labels.append(label)

    if not labels:
        prose_re = re.compile(r"\ban?\s+([a-z][a-z\s]{2,30}?)(?=\s*(?:,|and\b|\.|$))", re.IGNORECASE)
        for m in prose_re.finditer(scene_text):
            label = _normalise_object_label(m.group(1).strip())
            if 2 <= len(label.split()) <= 4 and label not in labels:
                labels.append(label)

    return labels


def _extract_scene_description(ep: dict) -> tuple[str, str]:
    last_scene = ""
    last_grasp = ""
    after_pick = False

    tool_calls = ep.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return last_scene, last_grasp

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tool = tc.get("tool", "")

        if tool in ("grasp_object", "place_object", "pick_up_object"):
            after_pick = True
            continue

        if tool != "describe_environment":
            continue

        args = tc.get("args", {})
        query = args.get("query", "") if isinstance(args, dict) else ""
        text = str(tc.get("output", "")).strip()
        if not text:
            continue

        if after_pick or _GRASP_VERIFY_PATTERN.search(query):
            last_grasp = text[:200] + ("..." if len(text) > 200 else "")
        else:
            last_scene = text[:300] + ("..." if len(text) > 300 else "")

    return last_scene, last_grasp


def _extract_pose_mutations(ep: dict) -> list[dict]:
    mutations = []
    tool_calls = ep.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return mutations

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tool = tc.get("tool", "")
        args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
        out_raw = tc.get("output", "{}")
        try:
            out = json.loads(out_raw) if isinstance(out_raw, str) else out_raw
            if not isinstance(out, dict):
                out = {}
        except Exception:
            out = {}

        if tool == "save_current_pose" and out.get("success"):
            mutations.append({
                "op":          "save",
                "name":        out.get("name") or args.get("name", ""),
                "description": args.get("description", ""),
            })
        elif tool == "delete_saved_pose" and out.get("success"):
            mutations.append({
                "op":   "delete",
                "name": out.get("deleted") or args.get("name", ""),
            })
        elif tool == "rename_saved_pose" and out.get("success"):
            parts = str(out.get("renamed", " -> ")).split(" -> ")
            if len(parts) == 2:
                mutations.append({
                    "op":      "rename",
                    "old":     parts[0].strip(),
                    "new":     parts[1].strip(),
                })
        elif tool == "list_saved_poses" and out.get("success"):
            poses = out.get("poses", {})
            mutations.append({
                "op":    "list_snapshot",
                "poses": {
                    name: (data.get("description", "") if isinstance(data, dict) else str(data))
                    for name, data in poses.items()
                } if isinstance(poses, dict) else {},
            })
    return mutations


def _extract_key_args(ep: dict) -> dict:
    key_args: dict = {}
    tool_calls = ep.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return key_args

    placements: list[dict] = []
    candidate_pick: str | None = None
    current_held:   str | None = None
    current_target: str | None = None
    in_pick_phase = True
    pick_attempts = 0

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tool = tc.get("tool", "")
        args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}

        try:
            out = json.loads(tc.get("output", "{}")) if isinstance(tc.get("output"), str) else tc.get("output", {})
            success = out.get("success", True) if isinstance(out, dict) else True
        except Exception:
            success = True

        if tool == "segment_objects":
            q = args.get("query", "")
            if in_pick_phase and q and not re.search(r'\?', q):
                candidate_pick = _normalise_object_label(q)

        elif tool == "pick_up_object":
            pick_attempts += 1
            if success:
                current_held = candidate_pick
                current_target = None
                in_pick_phase = False

        elif tool == "get_place_pose":
            raw_target = args.get("target_object_label")
            current_target = _normalise_object_label(raw_target) if raw_target else None

        elif tool == "place_object":
            if success and current_held:
                placements.append({
                    "picked":    current_held,
                    "placed_on": current_target,
                })
            current_held = None
            current_target = None
            candidate_pick = None
            in_pick_phase = True

    if placements:
        key_args["placements"]    = placements
        key_args["picked_object"] = placements[0]["picked"]
        key_args["placed_on"]     = placements[0]["placed_on"]

    total_picks = len(placements) if placements else 1
    if pick_attempts > total_picks:
        key_args["retries"] = pick_attempts - total_picks

    return key_args


# ── Episode summarisation ──────────────────────────────────────────────────────

def summarise_episode(ep: dict) -> dict:
    if not isinstance(ep, dict):
        return {}

    seen: list[str] = []
    tool_calls = ep.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict) and "tool" in tc:
                if tc["tool"] not in seen:
                    seen.append(tc["tool"])
            elif isinstance(tc, (list, tuple)) and len(tc) > 0:
                name = str(tc[0])
                if name not in seen:
                    seen.append(name)

    key_args     = _extract_key_args(ep)
    task         = _classify_task(ep.get("query", ""))
    seen_objects = _extract_seen_objects(ep)
    scene_desc, grasp_assessment = _extract_scene_description(ep)
    pose_muts    = _extract_pose_mutations(ep)

    recorder_retries = ep.get("retries", {})
    if isinstance(recorder_retries, dict):
        retry_count = recorder_retries.get("count", 0)
        retry_attempts = recorder_retries.get("attempts", [])
    elif isinstance(recorder_retries, int):
        retry_count = recorder_retries
        retry_attempts = []
    else:
        retry_count = 0
        retry_attempts = []

    if retry_count > key_args.get("retries", 0):
        key_args["retries"] = retry_count

    ep["_key_args_retries"] = key_args.get("retries", 0)
    ep["_grasp_assessment"] = grasp_assessment

    last_failure_reason = ""
    if retry_attempts and isinstance(retry_attempts, list) and isinstance(retry_attempts[-1], dict):
        last_failure_reason = retry_attempts[-1].get("failure_reason", "")

    entry: dict = {
        "time":         str(ep.get("timestamp_start", ""))[:19],
        "duration_s":   ep.get("duration_s"),
        "query":        ep.get("query", ""),
        "task_type":    task,
        "outcome":      ep.get("outcome", "unknown"),
        "outcome_fact": _extract_outcome_fact(ep),
        "tools":        seen,
        "_pose_mutations": pose_muts,
    }

    if seen_objects:
        entry["seen_objects"] = seen_objects
    if key_args:
        entry["key_args"] = key_args
    if scene_desc:
        entry["scene_description"] = scene_desc
    if grasp_assessment:
        entry["grasp_assessment"] = grasp_assessment
    if ep.get("error"):
        entry["error"] = str(ep["error"])

    if last_failure_reason and entry["outcome"] != "success":
        entry["outcome_fact"] = last_failure_reason

    return entry


def summarise_session(path: Path) -> tuple[dict, list[dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: skipping {path} — {e}")
        return {}, []

    if isinstance(data, list):
        meta = {
            "session_id":    path.stem,
            "created_at":    "",
            "source_file":   str(path),
            "episode_count": len(data),
        }
        episodes = [summarise_episode(ep) for ep in data if isinstance(ep, dict)]
        return meta, episodes

    if not isinstance(data, dict):
        return {}, []

    raw_episodes = data.get("episodes", [])
    if not isinstance(raw_episodes, list):
        raw_episodes = []

    meta = {
        "session_id":    data.get("session_id", path.stem),
        "created_at":    data.get("created_at", ""),
        "source_file":   str(path),
        "episode_count": len(raw_episodes),
    }
    episodes = [summarise_episode(ep) for ep in raw_episodes if isinstance(ep, dict)]
    return meta, episodes


# ── World-state extractor ──────────────────────────────────────────────────────

def derive_world_state(episodes: list[dict]) -> dict:
    state: dict = {
        "last_updated":   "",
        "arm_pose":       "unknown",
        "gripper":        "unknown",
        "handled_objects": [],
        "scene_objects":   [],
        "known_poses":    {},
        "scene":          "",
        "notes":          [],
    }

    seen_objects:  set[str] = set()
    scene_objects: set[str] = set()
    known_poses: dict[str, str] = {}

    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        kargs = ep.get("key_args", {}) if isinstance(ep.get("key_args"), dict) else {}

        if ep.get("scene_description"):
            state["scene"] = ep["scene_description"]

        for mut in ep.get("_pose_mutations", []):
            if not isinstance(mut, dict):
                continue
            op = mut.get("op")
            if op == "list_snapshot":
                if isinstance(mut.get("poses"), dict):
                    known_poses = dict(mut["poses"])
            elif op == "save":
                name = mut.get("name", "")
                if name:
                    known_poses[name] = mut.get("description", "")
            elif op == "delete":
                known_poses.pop(mut.get("name", ""), None)
            elif op == "rename":
                old, new = mut.get("old", ""), mut.get("new", "")
                if old in known_poses:
                    known_poses[new] = known_poses.pop(old)

        for label in ep.get("seen_objects", []):
            seen_objects.add(_normalise_object_label(label))

        for p in kargs.get("placements", []):
            if isinstance(p, dict):
                if p.get("picked"):
                    seen_objects.add(_normalise_object_label(p["picked"]))
                if p.get("placed_on"):
                    seen_objects.add(_normalise_object_label(p["placed_on"]))

        scene_text = ep.get("scene_description", "")
        if scene_text:
            for label in _extract_scene_object_labels(scene_text):
                scene_objects.add(label)

    for ep in reversed(episodes):
        if not isinstance(ep, dict):
            continue
        if not state["last_updated"]:
            state["last_updated"] = ep.get("time", "")

        t     = ep.get("task_type")
        tools = ep.get("tools", [])

        if state["arm_pose"] == "unknown":
            if t == "home" and ep.get("outcome") == "success":
                state["arm_pose"] = "home"
            elif t == "named_pose_move" and ep.get("outcome") == "success":
                m = re.search(r"Moved to '([^']+)'", ep.get("outcome_fact", ""))
                if m:
                    state["arm_pose"] = m.group(1)
            elif t == "cartesian_move" and ep.get("outcome") == "success":
                state["arm_pose"] = "custom"

        if state["gripper"] == "unknown":
            if ep.get("outcome") == "success":
                if "open_the_gripper" in tools:
                    state["gripper"] = "open"
                elif "close_the_gripper" in tools:
                    state["gripper"] = "closed"
                elif t == "pick_and_place":
                    state["gripper"] = "open"
                elif t == "pick_only":
                    state["gripper"] = "closed"

        if state["arm_pose"] != "unknown" and state["gripper"] != "unknown":
            break

    state["handled_objects"] = sorted(seen_objects)
    state["scene_objects"]   = sorted(scene_objects)
    state["known_poses"]     = known_poses

    return state


# ── Procedure miner ────────────────────────────────────────────────────────────

def mine_procedures(episodes: list[dict]) -> dict:
    from collections import Counter

    buckets: dict[str, list[tuple[str, ...]]] = {}
    for ep in episodes:
        if not isinstance(ep, dict) or ep.get("outcome") != "success":
            continue
        task = ep.get("task_type", "other")
        if task == "other":
            continue
        seq = tuple(ep.get("tools", []))
        if not seq:
            continue
        buckets.setdefault(task, []).append(seq)

    procedures: dict = {}
    for task, seqs in buckets.items():
        if not seqs:
            continue
        modal_seq, count = Counter(seqs).most_common(1)[0]
        durations = [
            ep["duration_s"] for ep in episodes
            if isinstance(ep, dict)
            and ep.get("task_type") == task
            and ep.get("outcome") == "success"
            and ep.get("duration_s") is not None
        ]
        avg_dur = round(sum(durations) / len(durations), 1) if durations else None

        procedures[task] = {
            "tool_sequence":  list(modal_seq),
            "observed_count": len(seqs),
            "modal_count":    count,
            "avg_duration_s": avg_dur,
        }

    return procedures


# ── Main aggregator ────────────────────────────────────────────────────────────

def build_summary(session_files: list[Path], top_k: int | None = None) -> dict:
    all_episodes: list[dict] = []

    for path in sorted(session_files):
        _, episodes = summarise_session(path)
        all_episodes.extend(episodes)

    all_episodes = [ep for ep in all_episodes if isinstance(ep, dict)]
    all_episodes.sort(key=lambda e: e.get("time", ""))

    if not all_episodes:
        return {}

    world_state = derive_world_state(all_episodes)
    procedures  = mine_procedures(all_episodes)

    meaningful = [ep for ep in all_episodes if not _is_trivial(ep.get("query", ""))]

    for ep in meaningful:
        ep.pop("_pose_mutations", None)
        ep.pop("_key_args_retries", None)

    if top_k and len(meaningful) > top_k:
        meaningful = meaningful[-top_k:]

    success = sum(1 for e in all_episodes if e.get("outcome") == "success")
    errors  = sum(1 for e in all_episodes if e.get("outcome") == "error")
    valid_durations = [e["duration_s"] for e in all_episodes if e.get("duration_s") is not None]
    avg_dur = sum(valid_durations) / len(valid_durations) if valid_durations else 0

    tool_freq: dict[str, int] = {}
    for ep in all_episodes:
        for t in ep.get("tools", []):
            tool_freq[t] = tool_freq.get(t, 0) + 1
    tool_freq = dict(sorted(tool_freq.items(), key=lambda x: -x[1]))

    task_freq: dict[str, int] = {}
    for ep in all_episodes:
        tt = ep.get("task_type", "other")
        task_freq[tt] = task_freq.get(tt, 0) + 1

    return {
        "generated_at": datetime.now().isoformat()[:19],
        "world_state": world_state,
        "procedures":  procedures,
        "recent_episodes": meaningful,
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
        f"# handled objects: {', '.join(ws.get('handled_objects', [])) or '—'}",
        f"# scene objects:   {', '.join(ws.get('scene_objects', [])) or '—'}",
        f"# known poses: {', '.join(ws.get('known_poses', {}).keys()) or '—'}",
        f"# scene: {ws.get('scene', '—')[:120]}",
        "",
        "# PROCEDURES",
    ]
    for task, proc in summary.get("procedures", {}).items():
        lines.append(f"  {task}: {' → '.join(proc['tool_sequence'])}  "
                     f"(n={proc['observed_count']}, avg {proc['avg_duration_s']}s)")
    lines.append("")
    lines.append("# RECENT MEANINGFUL EPISODES")
    for ep in summary.get("recent_episodes", []):
        status = "✓" if ep.get("outcome") == "success" else ("✗" if ep.get("outcome") == "error" else "?")
        dur    = f"{ep['duration_s']:.1f}s" if ep.get("duration_s") is not None else "?"
        lines.append(
            f"[{ep.get('time','')}] {status} {dur} [{ep.get('task_type','')}] "
            f"Q: {ep.get('query','')} | FACT: {ep.get('outcome_fact','')}"
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

    s  = summary["stats"]
    ws = summary["world_state"]
    print(f"\nStats: {s['total_episodes']} total episodes | "
          f"{s['meaningful_episodes']} meaningful | "
          f"{s['trivial_episodes_pruned']} trivial pruned")
    print(f"       {s['successful']} success | {s['errors']} errors | "
          f"avg {s['average_duration_s']}s/episode")
    print(f"World state: arm={ws['arm_pose']} | gripper={ws['gripper']}")
    if ws.get("known_poses"):
        print(f"Known poses: {', '.join(ws['known_poses'].keys())}")
    if ws.get("scene"):
        print(f"Last scene: {ws['scene'][:100]}…")
    print("Procedures mined:", ", ".join(summary["procedures"].keys()) or "none")


if __name__ == "__main__":
    main()