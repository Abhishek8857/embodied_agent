"""
episode_recorder.py
-------------------
Records every agent interaction as a structured episode and persists them
to a human-readable JSON file.

Episode schema
--------------
{
  "episode_id": "ep_0042",
  "session_id": "session_20260220_143201",
  "timestamp_start": "2026-02-20T14:32:01.123456",
  "timestamp_end":   "2026-02-20T14:32:08.987654",
  "duration_s": 7.86,
  "query": "pick up the red cube and place it on the shelf",
  "tool_calls": [
    {
      "step": 1,
      "tool": "capture_rgbd",
      "args": {},
      "output": "...",
      "tool_call_id": "call_abc123"
    },
    ...
  ],
  "final_response": "I have successfully picked up the red cube and placed it on the shelf.",
  "outcome": "success",         // "success" | "error" | "unknown"
  "error": null                 // error message string if outcome == "error"
}
"""

import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

class ToolCallRecord:
    """One tool invocation inside an episode."""

    def __init__(self, step: int, tool: str, args: dict, output: str, tool_call_id: str):
        self.step = step
        self.tool = tool
        self.args = args
        self.output = output
        self.tool_call_id = tool_call_id

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "output": self.output,
            "tool_call_id": self.tool_call_id,
        }


class Episode:
    """
    A single agent interaction: one query → (N tool calls) → final response.
    """

    def __init__(self, query: str, session_id: str, episode_number: int):
        self.episode_id = f"ep_{episode_number:04d}"
        self.session_id = session_id
        self.query = query
        self.timestamp_start = datetime.now().isoformat()
        self.timestamp_end: Optional[str] = None
        self.duration_s: Optional[float] = None
        self._start_time = datetime.now()
        self.tool_calls: list[ToolCallRecord] = []
        self.final_response: Optional[str] = None
        self.outcome: str = "unknown"   # "success" | "error" | "unknown"
        self.error: Optional[str] = None
        self.retries: list[dict] = []
        
    def record_retry(self, attempt: int, failure_reason: str, hint_used: str):
        """Called before each retry attempt to track what was tried."""
        self.retries["count"] = attempt
        self.retries["attempts"].append({
            "attempt": attempt,
            "failure_reason": failure_reason,
            "hint_used": hint_used,
        })
        
    def add_tool_call(self, tool: str, args: dict, output: str, tool_call_id: str):
        step = len(self.tool_calls) + 1
        self.tool_calls.append(ToolCallRecord(step, tool, args, output, tool_call_id))

    def close(self, final_response: str, outcome: str = "success", error: str = None):
        self.final_response = final_response
        self.outcome = outcome
        self.error = error
        now = datetime.now()
        self.timestamp_end = now.isoformat()
        self.duration_s = round((now - self._start_time).total_seconds(), 3)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_s": self.duration_s,
            "query": self.query,
            "retries": self.retries,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_response": self.final_response,
            "outcome": self.outcome,
            "error": self.error,
        }


# ── Episode recorder ──────────────────────────────────────────────────────────

class EpisodeRecorder:
    """
    Thread-safe recorder that persists agent episodes to a JSON file.

    One JSON file is created per session under `save_dir`, named after the
    session ID. Episodes are appended incrementally so the file is always
    up-to-date even if the process is killed mid-run.

    Usage
    -----
        recorder = EpisodeRecorder(save_dir="episodes")

        ep = recorder.start_episode(query="pick up the red cube")
        ep.add_tool_call(tool="capture_rgbd", args={}, output="...", tool_call_id="c1")
        recorder.close_episode(ep, final_response="Done!", outcome="success")
    """

    def __init__(self, save_dir: str = "episodes", session_id: str = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._lock = threading.Lock()
        self._episode_counter = 0
        self._episodes: list[Episode] = []
        self._seen_tool_call_ids: set[str] = set()  # cross-episode guard against LangGraph state carryover

        # One JSON file per session for easy human reading
        self._filepath = self.save_dir / f"{self.session_id}.json"
        self._init_file()


    def start_episode(self, query: str) -> Episode:
        """Open a new episode for the given user query."""
        with self._lock:
            self._episode_counter += 1
            ep = Episode(query=query, session_id=self.session_id, episode_number=self._episode_counter)
        return ep

    def close_episode(self, episode: Episode, final_response: str,
                      outcome: str = "success", error: str = None):
        """Finalise and persist an episode to disk."""
        episode.close(final_response=final_response, outcome=outcome, error=error)
        with self._lock:
            self._episodes.append(episode)
            self._flush()

    def close_episode_from_formatted_response(self, episode: Episode,
                                              formatted: dict,
                                              outcome: str = "success",
                                              error: str = None):
        """
        Convenience wrapper: populate tool calls from the dict produced by
        utils.format_response() and then close the episode.

        formatted = {
            "human_messages": [...],
            "ai_messages":    [...],   # tool call intentions
            "tool_calls":     [...],   # tool outputs
            "final_response": [...],
        }
        """
        # Build a quick lookup: tool_call_id → output dict
        tool_output_map = {t["tool_call_id"]: t for t in formatted.get("tool_calls", [])}

        for ai_msg in formatted.get("ai_messages", []):
            tool_call_id = ai_msg.get("id", "")
            tool_name    = ai_msg.get("name", "unknown")
            args         = ai_msg.get("args", {})
            output_entry = tool_output_map.get(tool_call_id, {})
            output_str   = output_entry.get("output", "")
            if tool_call_id in self._seen_tool_call_ids:
                continue  # skip: carried over from a previous episode's LangGraph state
            self._seen_tool_call_ids.add(tool_call_id)
            episode.add_tool_call(tool=tool_name, args=args, output=output_str,
                                  tool_call_id=tool_call_id)

        final_responses = formatted.get("final_response", [])
        final_text = final_responses[-1]["content"] if final_responses else ""

        self.close_episode(episode, final_response=final_text, outcome=outcome, error=error)

    def get_all_episodes(self) -> list[dict]:
        """Return all recorded episodes as plain dicts."""
        with self._lock:
            return [ep.to_dict() for ep in self._episodes]

    def get_episode(self, episode_id: str) -> Optional[dict]:
        """Look up a single episode by its ID."""
        with self._lock:
            for ep in self._episodes:
                if ep.episode_id == episode_id:
                    return ep.to_dict()
        return None

    def summary(self) -> dict:
        """High-level stats for the current session."""
        with self._lock:
            total   = len(self._episodes)
            success = sum(1 for ep in self._episodes if ep.outcome == "success")
            errors  = sum(1 for ep in self._episodes if ep.outcome == "error")
            avg_dur = (sum(ep.duration_s or 0 for ep in self._episodes) / total) if total else 0
            return {
                "session_id": self.session_id,
                "total_episodes": total,
                "successful": success,
                "errors": errors,
                "average_duration_s": round(avg_dur, 3),
                "file": str(self._filepath),
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_file(self):
        """Write an empty session skeleton so the file exists immediately."""
        skeleton = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "episodes": [],
        }
        self._write(skeleton)

    def _flush(self):
        """Rewrite the full JSON file with all episodes. Must be called under lock."""
        data = {
            "session_id": self.session_id,
            "created_at": self._filepath.stat().st_ctime
                          if self._filepath.exists() else datetime.now().isoformat(),
            "episodes": [ep.to_dict() for ep in self._episodes],
        }
        self._write(data)

    def _write(self, data: dict):
        # Write to a temp file then rename for atomic updates
        tmp = self._filepath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(self._filepath)