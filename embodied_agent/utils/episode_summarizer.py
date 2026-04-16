"""
memory_summarizer.py  (v3)
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

# describe_environment calls whose query is about confirming a grasp outcome
# should NOT be stored as the world-state scene — they are outcome checks.
_GRASP_VERIFY_PATTERN = re.compile(
    r"\b(grasped|grasping|assess\s+whether|likely\s+grasped|being\s+held|confirm\s+if"
    r"|is\s+(it\s+)?grasped|visual\s+assessment)\b",
    re.IGNORECASE,
)

# Task-type classifiers  (query → procedure key)
# Order matters: more specific patterns first.
_TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pick.*place|place.*on|stack|put.*on",              re.IGNORECASE), "pick_and_place"),
    (re.compile(r"pick\s+up|grasp|grab",                             re.IGNORECASE), "pick_only"),
    (re.compile(r"go\s+home|home\s+pose|return\s+home",              re.IGNORECASE), "home"),
    (re.compile(r"open\s+gripper",                                   re.IGNORECASE), "open_gripper"),
    (re.compile(r"close\s+gripper",                                  re.IGNORECASE), "close_gripper"),
    (re.compile(r"describe|look|see|observe|what.+see",              re.IGNORECASE), "observe"),
    # Named-pose navigation: "go to place pose", "move to retract", etc.
    (re.compile(r"\b(go\s+to|move\s+to|navigate\s+to)\s+\w",        re.IGNORECASE), "named_pose_move"),
    # Relative Cartesian moves: "move forward 20 cm", "go up 10 cm"
    (re.compile(r"\bmove\s+(forward|backward|left|right|up|down)\b", re.IGNORECASE), "cartesian_move"),
    # Pose management
    (re.compile(r"\bsave\s+(this|current|the)?\s*(pose|position)\b", re.IGNORECASE), "save_pose"),
    (re.compile(r"\bdelete\s+\w+\s+pose\b|\brename\s+\w+",           re.IGNORECASE), "pose_management"),
]


# ── Object label normalisation ─────────────────────────────────────────────────
# Maps any synonym shape word onto a single canonical shape word so that
# "red block" and "red cube" are stored as the same object.
#
# Rules applied in order by _normalise_object_label():
#   1. Lowercase + collapse whitespace.
#   2. Shape-word synonyms: block → cube  (both refer to the same small uniform object).
#   3. Brand/description aliases: any label that contains "nvidia" or describes
#      the large black box collapses to "nvidia box".
#   4. Strip leading articles ("a ", "an ", "the ").

_SHAPE_SYNONYMS: dict[str, str] = {
    "block": "cube",
}

# Regex that matches any label containing "nvidia" or a verbose description of
# the NVIDIA box (e.g. "black rectangular box with the NVIDIA logo …").
_NVIDIA_ALIAS_RE = re.compile(
    r"nvidia|black\s+rectangular\s+box",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def _normalise_object_label(label: str) -> str:
    """
    Return a canonical, lowercase form of an object label.

    Examples
    --------
    "Blue Block"          → "blue cube"
    "red block"           → "red cube"
    "NVIDIA cube"         → "nvidia box"
    "nvidia cube"         → "nvidia box"
    "black rectangular box with the NVIDIA logo …"
                          → "nvidia box"
    "A yellow cube"       → "yellow cube"
    """
    if not label:
        return label

    # 1. Lowercase + collapse internal whitespace
    norm = " ".join(label.lower().split())

    # 2. Strip leading article
    norm = _ARTICLE_RE.sub("", norm).strip()

    # 3. Brand/verbose-description aliases (before shape substitution so we
    #    don't accidentally turn "nvidia block" into "nvidia cube" first)
    if _NVIDIA_ALIAS_RE.search(norm):
        return "nvidia box"

    # 4. Shape-word substitution — only replaces whole words
    for synonym, canonical in _SHAPE_SYNONYMS.items():
        norm = re.sub(rf"\b{re.escape(synonym)}\b", canonical, norm)

    return norm



def _classify_task(query: str) -> str:
    for pattern, label in _TASK_PATTERNS:
        if pattern.search(query):
            return label
    return "other"


def _is_trivial(query: str) -> bool:
    # Empty or whitespace-only queries are always trivial
    if not query or not query.strip():
        return True
    return any(p.fullmatch(query.strip()) for p in _TRIVIAL_QUERY_PATTERNS)


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

    # Retry annotation — shown on all task types when retries occurred
    recorder_retries = ep.get("retries", {})
    retry_count = recorder_retries.get("count", 0)
    # Also check tool-call-based retries for pick/place
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
            # Prepend a terse label so downstream readers know what this is
            first_line = grasp_assessment.splitlines()[0]
            base += f" Visual check: {first_line}"
        return base

    if task == "home":
        return f"Arm returned to home pose{retry_s}. {result_str}."

    if task in ("open_gripper", "close_gripper"):
        action = "opened" if task == "open_gripper" else "closed"
        return f"Gripper {action}{retry_s}. {result_str}."

    if task == "named_pose_move":
        # Extract target pose name from query
        m = re.search(r"\b(?:go\s+to|move\s+to|navigate\s+to)\s+(\w[\w\s]*)", query, re.IGNORECASE)
        pose_name = m.group(1).strip() if m else "named pose"
        return f"Moved to '{pose_name}'{retry_s}. {result_str}."

    if task == "cartesian_move":
        return f"{query.strip().capitalize()}{retry_s}. {result_str}."

    if task == "save_pose":
        # Try to extract the saved pose name
        m = re.search(r"as\s+([\w\s]+?)(?:\s*$|\s*\.)", query, re.IGNORECASE)
        pose_name = m.group(1).strip() if m else "pose"
        return f"Saved pose '{pose_name}'{retry_s}. {result_str}."

    if task == "pose_management":
        return f"{query.strip().capitalize()}{retry_s}. {result_str}."

    # Generic fallback
    return f"{query.strip().capitalize()}{retry_s}. {result_str}."


def _extract_seen_objects(ep: dict) -> list[str]:
    """
    Extract object labels seen during an episode from segment_objects outputs.
    These are the ground-truth labels used by the agent, so they're more
    reliable for handled_objects than free-text from describe_environment.
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
            label = _normalise_object_label(item.get("label", ""))
            if label and label not in labels:
                labels.append(label)
    return labels


def _extract_scene_object_labels(scene_text: str) -> list[str]:
    """
    Parse object labels from a free-text describe_environment scene description.

    The model consistently uses numbered lists like:
        "1. A red cube located on the left side."
        "2. A blue cube located below the red cube."

    We extract the noun phrase between "A/An" and the first locative verb
    ("located", "positioned", "placed", "sitting", "resting", "on the", "at").
    Falls back to splitting on commas for prose-style descriptions.

    Returns a list of lowercase label strings, e.g. ["red cube", "blue cube"].
    """
    labels: list[str] = []

    # Pattern 1: numbered list items — "1. A <label> located …"
    list_item_re = re.compile(
        r"^\s*\d+\.\s+[Aa]n?\s+(.+?)(?:\s+(?:located|positioned|placed|sitting|resting)\b|[,.]|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in list_item_re.finditer(scene_text):
        label = _normalise_object_label(m.group(1).strip())
        if label and label not in labels:
            labels.append(label)

    # Pattern 2: prose fallback — "there are a red cube, a blue cube and a yellow cube"
    if not labels:
        prose_re = re.compile(r"\ban?\s+([a-z][a-z\s]{2,30}?)(?=\s*(?:,|and\b|\.|$))", re.IGNORECASE)
        for m in prose_re.finditer(scene_text):
            label = _normalise_object_label(m.group(1).strip())
            # Basic filter: reject single common words and over-long phrases
            if 2 <= len(label.split()) <= 4 and label not in labels:
                labels.append(label)

    return labels


def _extract_scene_description(ep: dict) -> tuple[str, str]:
    """
    Extract scene description and grasp-verification result separately.

    Calls whose args.query matches _GRASP_VERIFY_PATTERN are outcome checks,
    not scene snapshots. They are returned as the second element so they can
    be folded into outcome_fact rather than polluting world_state.scene.

    Returns:
        (scene_desc, grasp_assessment)
        scene_desc       – last general scene description (empty if none)
        grasp_assessment – last grasp-verification result (empty if none)
    """
    last_scene = ""
    last_grasp = ""
    after_pick = False

    for tc in ep.get("tool_calls", []):
        tool = tc.get("tool", "")

        # Mark boundary after a grasp/place action
        if tool in ("grasp_object", "place_object", "pick_up_object"):
            after_pick = True
            continue

        if tool != "describe_environment":
            continue

        query = tc.get("args", {}).get("query", "")
        text = tc.get("output", "").strip()
        if not text:
            continue

        if after_pick or _GRASP_VERIFY_PATTERN.search(query):
            last_grasp = text[:200] + ("..." if len(text) > 200 else "")
        else:
            last_scene = text[:300] + ("..." if len(text) > 300 else "")

    return last_scene, last_grasp


def _extract_pose_mutations(ep: dict) -> list[dict]:
    """
    Extract pose registry mutations (save / delete / rename / list) from tool calls.
    Returns a list of mutation dicts for use in derive_world_state.
    """
    mutations = []
    for tc in ep.get("tool_calls", []):
        tool = tc.get("tool", "")
        args = tc.get("args", {})
        try:
            out = json.loads(tc.get("output", "{}"))
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
            parts = out.get("renamed", " -> ").split(" -> ")
            if len(parts) == 2:
                mutations.append({
                    "op":      "rename",
                    "old":     parts[0].strip(),
                    "new":     parts[1].strip(),
                })
        elif tool == "list_saved_poses" and out.get("success"):
            # list_saved_poses is a ground-truth snapshot — use it to seed known poses
            mutations.append({
                "op":    "list_snapshot",
                "poses": {
                    name: data.get("description", "")
                    for name, data in out.get("poses", {}).items()
                },
            })
    return mutations


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
                candidate_pick = _normalise_object_label(q)

        elif tool == "pick_up_object":
            pick_attempts += 1
            if success:
                current_held = candidate_pick
                current_target = None
                in_pick_phase = False   # now scanning for place target

        elif tool == "get_place_pose":
            raw_target = args.get("target_object_label")
            current_target = _normalise_object_label(raw_target) if raw_target else None

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
    scene_desc, grasp_assessment = _extract_scene_description(ep)
    pose_muts    = _extract_pose_mutations(ep)

    # ── Retry data from episode_recorder.record_retry() ───────────────────────
    # _extract_key_args() counts pick-attempt retries; record_retry() covers all
    # task types. Use whichever count is larger.
    recorder_retries = ep.get("retries", {})
    retry_count = recorder_retries.get("count", 0)
    if retry_count > key_args.get("retries", 0):
        key_args["retries"] = retry_count

    # Stash tool-call retry count so _extract_outcome_fact can access it
    ep["_key_args_retries"] = key_args.get("retries", 0)
    # Stash grasp assessment so _extract_outcome_fact can append it for pick tasks
    ep["_grasp_assessment"] = grasp_assessment

    # Last failure reason from recorder — used for outcome_fact on failed episodes
    retry_attempts = recorder_retries.get("attempts", [])
    last_failure_reason = (
        retry_attempts[-1].get("failure_reason", "") if retry_attempts else ""
    )

    entry: dict = {
        "time":         ep.get("timestamp_start", "")[:19],
        "duration_s":   ep.get("duration_s"),
        "query":        ep.get("query", ""),
        "task_type":    task,
        "outcome":      ep.get("outcome", "unknown"),
        "outcome_fact": _extract_outcome_fact(ep),
        "tools":        seen,
        # Internal field — used by derive_world_state, not injected into agent prompt
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
        entry["error"] = ep["error"]

    # For failed or retried episodes, replace generic outcome_fact with actual
    # failure reason so _compute_failure_stats() gets useful cause strings
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

    handled_objects is populated from:
      1. segment_objects outputs (structured, ground-truth labels) — objects
         the robot has actually attempted to segment / grasp.
      2. pick/place key_args (placement targets also reveal object names)

    scene_objects is populated from:
      1. describe_environment outputs from observe episodes — every object
         label mentioned in a general scene description.  These are parsed
         with a simple heuristic and represent what the robot has *seen*,
         regardless of whether it has ever touched them.

    known_poses is populated from:
      1. list_saved_poses snapshots (ground truth when present)
      2. save_current_pose / delete_saved_pose / rename_saved_pose mutations
    """
    state: dict = {
        "last_updated":   "",
        "arm_pose":       "unknown",
        "gripper":        "unknown",
        "handled_objects": [],   # objects the robot has segmented / grasped
        "scene_objects":   [],   # objects seen in describe_environment scene snapshots
        "known_poses":    {},    # name → description
        "scene":          "",    # most recent *general* scene description (grasp-verification excluded)
        "notes":          [],
    }

    seen_objects:  set[str] = set()
    scene_objects: set[str] = set()
    known_poses: dict[str, str] = {}

    for ep in episodes:
        t     = ep.get("task_type")
        kargs = ep.get("key_args", {})

        # ── Scene description: always take the most recent ─────────────────────
        if ep.get("scene_description"):
            state["scene"] = ep["scene_description"]

        # ── Pose registry mutations ────────────────────────────────────────────
        for mut in ep.get("_pose_mutations", []):
            op = mut.get("op")
            if op == "list_snapshot":
                # Ground-truth snapshot from list_saved_poses — overwrite entirely
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

        # ── Handled objects (segmented / grasped) ─────────────────────────────
        for label in ep.get("seen_objects", []):
            seen_objects.add(_normalise_object_label(label))

        for p in kargs.get("placements", []):
            if p.get("picked"):
                seen_objects.add(_normalise_object_label(p["picked"]))
            if p.get("placed_on"):
                seen_objects.add(_normalise_object_label(p["placed_on"]))

        # ── Scene objects (visible in describe_environment snapshots) ──────────
        scene_text = ep.get("scene_description", "")
        if scene_text:
            for label in _extract_scene_object_labels(scene_text):
                scene_objects.add(label)

        if ep["outcome"] != "success":
            continue

    # ── Reverse pass for arm / gripper state (most recent wins) ───────────────
    for ep in reversed(episodes):
        if not state["last_updated"]:
            state["last_updated"] = ep.get("time", "")

        t     = ep.get("task_type")
        tools = ep.get("tools", [])

        if state["arm_pose"] == "unknown":
            if t == "home" and ep["outcome"] == "success":
                state["arm_pose"] = "home"
            elif t == "named_pose_move" and ep["outcome"] == "success":
                # Extract destination from outcome_fact ("Moved to 'place_pose'. SUCCESS.")
                m = re.search(r"Moved to '([^']+)'", ep.get("outcome_fact", ""))
                if m:
                    state["arm_pose"] = m.group(1)
            elif t == "cartesian_move" and ep["outcome"] == "success":
                state["arm_pose"] = "custom"

        if state["gripper"] == "unknown":
            # FIX: check actual tool calls first — this correctly handles compound
            # queries like "go home and open gripper" where task_type is "home"
            # (first pattern match wins) but open_the_gripper was still executed.
            if ep["outcome"] == "success":
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
    """
    Derive canonical tool-call sequences from successful episodes
    grouped by task type.  For each task type, keep the most common
    (modal) sequence — that's the procedure the agent should follow.
    Skips "other" since it's a catch-all with no coherent procedure.
    """
    from collections import Counter

    buckets: dict[str, list[tuple[str, ...]]] = {}
    for ep in episodes:
        if ep["outcome"] != "success":
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

    # Strip internal fields before storing (not needed downstream)
    for ep in meaningful:
        ep.pop("_pose_mutations", None)
        ep.pop("_key_args_retries", None)

    if top_k and len(meaningful) > top_k:
        meaningful = meaningful[-top_k:]

    # ── Stats (over all episodes, including trivial) ──────────────────────────
    success = sum(1 for e in all_episodes if e["outcome"] == "success")
    errors  = sum(1 for e in all_episodes if e["outcome"] == "error")
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