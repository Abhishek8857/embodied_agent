"""
recovery_advisor.py
-------------------
Bridges episodic memory and task recovery.
Given a failure reason + query, it searches past episodes for similar
failures and extracts what recovery sequences actually worked — then
returns a concise hint string to inject into the retry prompt.
"""

import re
import json
from pathlib import Path
from .episode_recorder import EpisodeRecorder
from .memory_context import _compute_failure_stats

# ── Task type classifier ───────────────────────────────────────────────────────
_TASK_PATTERNS = {
    "pick":        re.compile(r"\bpick\b|\bgrab\b|\blift\b", re.I),
    "place":       re.compile(r"\bplace\b|\bput\b|\bset down\b", re.I),
    "move":        re.compile(r"\bmove\b|\bgo to\b|\bnavigate\b", re.I),
    "pick_place":  re.compile(r"(pick|grab).+(place|put|on top)", re.I),
    "gripper":     re.compile(r"\bgripper\b|\bopen\b|\bclose\b", re.I),
    "vision":      re.compile(r"\bsee\b|\blook\b|\bdescribe\b|\bscan\b", re.I),
}

def classify_task(query: str) -> str:
    for task, pat in _TASK_PATTERNS.items():
        if pat.search(query):
            return task
    return "other"


# ── RecoveryAdvisor ────────────────────────────────────────────────────────────

class RecoveryAdvisor:
    """
    Reads episodic memory to produce context-aware retry hints.

    Usage
    -----
        advisor = RecoveryAdvisor(recorder, memory_path="memory/memory.json")
        hint = advisor.get_hint(failure_reason, user_query)
        # inject `hint` into the retry message
    """

    def __init__(self, recorder: EpisodeRecorder, memory_path: str = "memory/memory.json"):
        self.recorder = recorder
        self.memory_path = Path(memory_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_hint(self, failure_reason: str, query: str) -> str:
        """
        Return a plain-text hint block for the retry prompt, or "" if
        no relevant history is found.
        """
        task_type = classify_task(query)
        lines = []

        # 1. Check cross-session memory for known failure patterns on this task type
        cross_session_hint = self._cross_session_hint(task_type)
        if cross_session_hint:
            lines.append(cross_session_hint)

        # 2. Check current session for a previously successful recovery on same task
        session_hint = self._session_recovery_hint(task_type, failure_reason)
        if session_hint:
            lines.append(session_hint)

        # 3. If failure mentions a specific tool, check if that tool has a known fix
        tool_hint = self._tool_specific_hint(failure_reason)
        if tool_hint:
            lines.append(tool_hint)

        if not lines:
            return ""

        return (
            "[Memory-informed recovery hints — apply these before retrying]\n"
            + "\n".join(lines)
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _cross_session_hint(self, task_type: str) -> str:
        """Read memory.json and surface failure patterns for this task type."""
        if not self.memory_path.exists():
            return ""

        try:
            with open(self.memory_path, encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            return ""

        episodes = memory.get("recent_episodes", [])
        stats = _compute_failure_stats(episodes)

        if task_type not in stats:
            return ""

        s = stats[task_type]
        if s["risk_level"] == "LOW":
            return ""

        parts = [
            f"• [{s['risk_level']} RISK] '{task_type}' has failed "
            f"{s['failures']}/{s['attempts']} times across sessions."
        ]
        if s["failure_causes"]:
            parts.append(f"  Known causes: {'; '.join(s['failure_causes'][-2:])}")

        # Surface a successful recovery tool sequence if one exists
        procs = memory.get("procedures", {}).get(task_type, {})
        seq = procs.get("tool_sequence")
        if seq:
            parts.append(f"  Proven sequence: {' → '.join(seq)}")

        return "\n".join(parts)

    def _session_recovery_hint(self, task_type: str, failure_reason: str) -> str:
        """
        Scan current-session episodes: if we've seen this task_type fail
        before and then succeed on a retry, extract what changed.
        """
        all_eps = self.recorder.get_all_episodes()
        relevant = [
            e for e in all_eps
            if e.get("task_type") == task_type
            and e.get("retries", {}).get("count", 0) > 0
            and e.get("outcome") == "success"
        ]
        if not relevant:
            return ""

        # Take the most recent successful recovery
        last = relevant[-1]
        hint_used = last.get("retries", {}).get("hint_used", "")
        sequence  = [tc["tool"] for tc in last.get("tool_calls", [])]

        lines = [f"• This session: '{task_type}' previously recovered successfully."]
        if hint_used:
            lines.append(f"  Hint that worked: {hint_used}")
        if sequence:
            lines.append(f"  Tool sequence that succeeded: {' → '.join(sequence[-6:])}")
        return "\n".join(lines)

    def _tool_specific_hint(self, failure_reason: str) -> str:
        """
        If the failure mentions a known fragile tool, return a targeted fix hint.
        """
        _TOOL_HINTS = {
            "segment_objects":       "Ensure capture_rgbd() was called immediately before segment_objects().",
            "get_latest_grasp_pose": "If no grasp pose is found, re-run capture_rgbd() → segment_objects() → save_segmentation_for_graspnet() before retrying.",
            "pick_up_object":        "Reset gripper to open state before attempting pick. Verify grasp pose is fresh (max_age_s ≤ 3.0).",
            "place_object":          "Confirm object is still in gripper via describe_environment() before executing place.",
            "move_to_pose":          "Verify target coordinates are within workspace bounds. Check for collision with get_current_pose() first.",
        }
        for tool, hint in _TOOL_HINTS.items():
            if tool in failure_reason:
                return f"• Tool-specific fix for '{tool}': {hint}"
        return ""