"""
memory_context.py  (navigation stack)
--------------------------------------
Reads memory.json and produces a compact, agent-ready context string covering:

  - Current robot state  (pose, last known location)
  - Known locations      (all waypoints the robot has visited or saved)
  - Previous tasks       (what was done, outcomes, any failures)
  - Learned behaviours   (which tool sequences work, average durations)
  - Failure risk signals (tasks/conditions with elevated failure probability)

The output is a short prose block (~200-400 tokens) injected directly into
the agent's system prompt.  Because the LLM decides what's relevant, the block
stays useful as memory grows.

Usage
-----
    from .memory_context import build_memory_context

    context_block = build_memory_context("memory.json")
    system_prompt = get_prompts() + context_block
"""

import json
from collections import defaultdict
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage


# ── Prompt templates ───────────────────────────────────────────────────────────

MAX_CONTEXT_WORDS = 350

_SYSTEM_PROMPT = f"""You are a memory distillation assistant for a wheeled indoor navigation robot agent.
Your job is to read a structured memory file and produce a compact, useful context block
that will be prepended to the robot agent's system prompt.

Rules:
- HARD LIMIT: {MAX_CONTEXT_WORDS} words maximum. Be ruthlessly concise.
- Use clear sections with short headers.
- Only include facts that are actually present in the memory — never invent or assume.
- Prioritise: current pose > last location > known locations > failure risks > recent task outcomes > learned patterns.
- Skip sections that have no data (e.g. omit "Current Pose" if pose is unknown).
- If an archived_summary exists, treat it as condensed history — do not expand it.
- Write in a factual, direct tone suitable for an agent system prompt.
- Do NOT include meta-commentary like "the memory shows..." — just state the facts directly.

For the Failure Risk section:
- A task type is HIGH RISK if its failure_rate exceeds 0.4 OR it has required 2+ retries in any episode.
- A task type is MEDIUM RISK if its failure_rate is between 0.2–0.4 OR it has a known fragile precondition.
- Only list LOW RISK tasks if they share a tool with a HIGH/MEDIUM risk task.
- For each risk entry state: the task type, the observed failure rate or retry count, and the most
  likely root cause if it can be inferred from the episode outcomes."""

_USER_PROMPT_TEMPLATE = """Here is the navigation robot agent's memory file. Extract and summarise the most
relevant context for the agent's next session.

MEMORY FILE:
{memory_json}

Produce a context block with these sections (omit any section with no data):

## Current Robot State
(current pose x/y/yaw, last known location name)

## Known Locations
List all location names the robot has successfully navigated to or saved.
These are valid navigation targets.

## Failure Risk Signals
CRITICAL — read this before planning any task.
List task types ordered from highest to lowest failure risk. For each entry include:
- Risk level: HIGH / MEDIUM / LOW
- Task type name
- Failure rate or retry pattern (e.g. "failed 1/3 attempts")
- Most likely cause (e.g. "goal tolerance too tight near Table D", "yaw drift on long runs")
If no failures or retries exist in memory, write: "No failure patterns recorded yet."

## User Preferences & Facts
(anything the user has stated about their preferences, critical tasks, or repeat patterns)

## Previous Tasks
(last 3-5 most relevant navigation tasks and their outcomes)

Write only the context block, nothing else."""


def build_memory_context(
    summary_path: str | Path = "memory.json",
    llm=None,
    fallback_to_manual: bool = True,
) -> str:
    """
    Read memory.json and return a compact context string for the agent.

    Parameters
    ----------
    summary_path       : path to memory.json (from memory_summarizer.py)
    llm                : a LangChain chat model.
                         If None, falls back to manual formatting.
    fallback_to_manual : if True and the LLM call fails, fall back to manual
                         formatting instead of raising an exception.

    Returns
    -------
    A formatted string ready to append to the agent's system prompt.
    """
    path = Path(summary_path)
    if not path.exists():
        print(f"[memory_context] WARNING: {path} not found — no memory context injected.")
        return ""

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    if llm is None:
        print("[memory_context] No LLM provided — using manual formatting.")
        return _manual_format(summary)

    return _llm_format(summary, llm, fallback_to_manual)


EPISODE_WINDOW = 10


def _llm_format(summary: dict, llm, fallback: bool) -> str:
    """Send memory JSON to the LLM and return the distilled context block."""
    payload = {
        **{k: v for k, v in summary.items() if k != "recent_episodes"},
        "recent_episodes": summary.get("recent_episodes", [])[-EPISODE_WINDOW:],
        "_failure_stats":  _compute_failure_stats(summary.get("recent_episodes", [])),
    }
    memory_json_str = json.dumps(payload, indent=2, ensure_ascii=False)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_PROMPT_TEMPLATE.format(memory_json=memory_json_str)),
    ]

    try:
        response  = llm.invoke(messages, max_tokens=int(MAX_CONTEXT_WORDS * 1.4))
        context   = response.content.strip()
        word_count = len(context.split())
        if word_count >= MAX_CONTEXT_WORDS * 0.95:
            print(f"[memory_context] WARNING: output near limit ({word_count}/{MAX_CONTEXT_WORDS} words)")
        else:
            print(f"[memory_context] Memory distilled: {word_count} words")
            print(context)
        return f"\n\n{context}\n"

    except Exception as e:
        print(f"[memory_context] LLM call failed: {e}")
        if fallback:
            print("[memory_context] Falling back to manual formatting.")
            return _manual_format(summary)
        raise


RISK_HIGH_FAILURE_RATE      = 0.40
RISK_MEDIUM_FAILURE_RATE    = 0.20
RISK_HIGH_RETRY_THRESHOLD   = 2
RISK_MEDIUM_RETRY_THRESHOLD = 1


def _compute_failure_stats(episodes: list[dict]) -> dict:
    """
    Aggregate per-task-type failure statistics from the episode list.
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "attempts":      0,
        "failures":      0,
        "max_retries":   0,
        "total_retries": 0,
        "failure_causes": [],
    })

    for ep in episodes:
        task = ep.get("task_type", "other")
        if task == "other":
            continue

        s = stats[task]
        s["attempts"] += 1

        retries = ep.get("key_args", {}).get("retries", 0)
        s["total_retries"] += retries
        s["max_retries"]    = max(s["max_retries"], retries)

        if ep.get("outcome") != "success":
            s["failures"] += 1
            cause = ep.get("outcome_fact", "").strip()
            if cause and cause not in s["failure_causes"]:
                s["failure_causes"].append(cause)
        elif retries > 0:
            cause = ep.get("outcome_fact", "").strip()
            if cause and cause not in s["failure_causes"]:
                s["failure_causes"].append(f"[recovered] {cause}")

    result = {}
    for task, s in stats.items():
        rate = s["failures"] / s["attempts"] if s["attempts"] else 0.0
        s["failure_rate"] = round(rate, 3)

        if rate >= RISK_HIGH_FAILURE_RATE or s["max_retries"] >= RISK_HIGH_RETRY_THRESHOLD:
            risk = "HIGH"
        elif rate >= RISK_MEDIUM_FAILURE_RATE or s["max_retries"] >= RISK_MEDIUM_RETRY_THRESHOLD:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        s["risk_level"] = risk
        result[task] = s

    return result



def _manual_format(summary: dict) -> str:
    """
    Fallback: manually format the memory summary into a prompt block.
    No LLM required — used when the model is unavailable.
    """
    ws    = summary.get("world_state", {})
    procs = summary.get("procedures", {})
    uf    = summary.get("user_facts", {})
    eps   = summary.get("recent_episodes", [])[-5:]

    lines = ["\n\n## Robot Memory\n"]

    pose      = ws.get("current_pose", {})
    last_loc  = ws.get("last_location")
    known_locs = ws.get("known_locations", [])

    has_pose = pose.get("x") is not None
    if has_pose or last_loc:
        lines.append("### Current Robot State")
        if has_pose:
            lines.append(
                f"- Pose: x={pose['x']:.3f} m, y={pose['y']:.3f} m, "
                f"yaw={pose['yaw_degrees']:.1f}°"
            )
        if last_loc:
            lines.append(f"- Last known location: {last_loc}")

    if known_locs:
        lines.append("\n### Known Locations")
        lines.append(f"- {', '.join(known_locs)}")

    all_eps    = summary.get("recent_episodes", [])
    fail_stats = _compute_failure_stats(all_eps)

    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risky = sorted(
        [(t, s) for t, s in fail_stats.items() if s["risk_level"] != "LOW"],
        key=lambda x: risk_order[x[1]["risk_level"]],
    )

    lines.append("\n### Failure Risk Signals")
    if not risky:
        lines.append("- No failure patterns recorded yet.")
    else:
        for task, s in risky:
            rate_pct  = f"{s['failure_rate']*100:.0f}%"
            retry_str = (f", max {s['max_retries']} retr{'ies' if s['max_retries'] != 1 else 'y'}/episode"
                         if s["max_retries"] > 0 else "")
            cause_str = f" — {s['failure_causes'][-1]}" if s["failure_causes"] else ""
            lines.append(
                f"- [{s['risk_level']}] {task}: "
                f"{s['failures']}/{s['attempts']} failed ({rate_pct}){retry_str}{cause_str}"
            )

    if uf:
        lines.append("\n### User Preferences & Facts")
        for key, value in uf.items():
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {value}")

    meaningful_eps = [e for e in eps if e.get("task_type") not in ("other", "query_pose", "query_locations")]
    if meaningful_eps:
        lines.append("\n### Previous Tasks")
        for ep in meaningful_eps:
            status = "✓" if ep["outcome"] == "success" else "✗"
            lines.append(f"- {status} [{ep['task_type']}] {ep['outcome_fact']}")

    real_procs = {k: v for k, v in procs.items() if k != "other" and v.get("tool_sequence")}
    if real_procs:
        lines.append("\n### Learned Behaviours")
        for task, proc in real_procs.items():
            seq = " → ".join(proc["tool_sequence"])
            dur = f"~{proc['avg_duration_s']}s" if proc.get("avg_duration_s") else ""
            lines.append(f"- {task} ({dur}): {seq}")

    return "\n".join(lines)