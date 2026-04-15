"""
recovery_advisor.py
-------------------
Bridges episodic memory and task recovery.
Given a failure reason + query, it searches past episodes for similar
failures and uses an LLM to generate a context-aware recovery hint.
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
    Uses an LLM to generate context-aware retry hints drawn from:
      - cross-session memory (memory.json)
      - current-session episode history
      - the specific failure reason

    Usage
    -----
        advisor = RecoveryAdvisor(recorder, llm=get_qwen_llm(), memory_path="memory/memory.json")
        hint = advisor.get_hint(failure_reason, user_query)
    """

    def __init__(self, recorder: EpisodeRecorder, llm, memory_path: str = "memory/memory.json"):
        self.recorder = recorder
        self.llm = llm
        self.memory_path = Path(memory_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_hint(self, failure_reason: str, query: str) -> str:
        """
        Return a plain-text hint block for the retry prompt, or "" if
        no relevant history is found.
        """
        task_type = classify_task(query)

        # Gather raw context from memory and current session
        cross_session_context = self._load_cross_session_context(task_type)
        session_context        = self._load_session_context(task_type)

        # If there's nothing to reason over, return empty
        if not cross_session_context and not session_context:
            return ""

        # Ask the LLM to synthesize a recovery hint
        return self._generate_hint(
            query=query,
            task_type=task_type,
            failure_reason=failure_reason,
            cross_session_context=cross_session_context,
            session_context=session_context,
        )

    # ── Context loaders ────────────────────────────────────────────────────────

    def _load_cross_session_context(self, task_type: str) -> dict:
        """
        Load relevant data from memory.json for this task type.
        Returns a dict with failure stats, known causes, and proven sequences.
        """
        if not self.memory_path.exists():
            return {}

        try:
            with open(self.memory_path, encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            return {}

        episodes = memory.get("recent_episodes", [])
        stats = _compute_failure_stats(episodes)
        task_stats = stats.get(task_type)

        # Pull relevant episodes for this task type (successes after retries,
        # and failures) so the LLM can reason over what actually happened
        relevant_episodes = [
            {
                "query":          ep.get("query"),
                "outcome":        ep.get("outcome"),
                "retries":        ep.get("retries", {}),
                "tool_sequence":  [tc["tool"] for tc in ep.get("tool_calls", [])],
                "final_response": ep.get("final_response", "")[:300],  # truncate for context window
            }
            for ep in episodes
            if classify_task(ep.get("query", "")) == task_type
        ]

        return {
            "task_type":          task_type,
            "stats":              task_stats,
            "procedures":         memory.get("procedures", {}).get(task_type, {}),
            "relevant_episodes":  relevant_episodes[-5:],  # last 5 relevant episodes
        }

    def _load_session_context(self, task_type: str) -> list[dict]:
        """
        Scan current-session episodes for same task type, focusing on
        retried episodes and what recovery sequences succeeded.
        """
        all_eps = self.recorder.get_all_episodes()
        return [
            {
                "query":         ep.get("query"),
                "outcome":       ep.get("outcome"),
                "retries":       ep.get("retries", {}),
                "tool_sequence": [tc["tool"] for tc in ep.get("tool_calls", [])],
                "final_response": ep.get("final_response", "")[:300],
            }
            for ep in all_eps
            if classify_task(ep.get("query", "")) == task_type
        ][-5:]  # last 5 relevant from this session

    # ── LLM hint generation ────────────────────────────────────────────────────

    def _generate_hint(
        self,
        query: str,
        task_type: str,
        failure_reason: str,
        cross_session_context: dict,
        session_context: list,
    ) -> str:
        """
        Prompt the LLM with all gathered context and ask it to generate
        a concise, actionable recovery hint.
        """
        prompt = f"""You are a recovery advisor for a 7-DOF robot arm agent.
        The agent just failed a task and is about to retry. Your job is to generate a concise, 
        actionable recovery hint based on the failure and past episode history.

        === CURRENT FAILURE ===
        Task type    : {task_type}
        User query   : {query}
        Failure reason: {failure_reason}

        === CROSS-SESSION MEMORY (past episodes across all sessions) ===
        {json.dumps(cross_session_context, indent=2, default=str)}

        === CURRENT SESSION HISTORY (episodes so far this session) ===
        {json.dumps(session_context, indent=2, default=str)}

        === INSTRUCTIONS ===
        - Analyse what has gone wrong historically for this task type.
        - Identify what tool sequences or strategies have led to success after failure.
        - Generate a SHORT, SPECIFIC, ACTIONABLE hint (2 bullet points max).
        - Focus on what the agent should do on the retry. It is fine if  
        - Do NOT repeat generic advice. Only mention things grounded in the history above.
        - Do NOT ask clarifying questions. Do NOT request more information. Always commit to a hint.
        - NEVER suggest reusing a previous grasp pose, saved coordinates, or cached tool outputs —
        the object may have moved since the last attempt. Always recommend fresh perception.
        - If there is genuinely no useful pattern in the history, respond with exactly: NO_HINT
        Recovery hint:"""

        try:
            response = self.llm.invoke(prompt)
            # Handle both string responses and LangChain message objects
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()

            if not text or text == "NO_HINT":
                return ""

            return "[Memory-informed recovery hints — apply these before retrying]\n" + text

        except Exception as e:
            return ""