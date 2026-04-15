"""
memory_context.py
-----------------
Uses Qwen (via OpenRouter) to read the full memory_summary.json and produce
a compact, agent-ready context string covering:

  - Current robot state  (arm pose, gripper, known objects)
  - User preferences     (favorite color, name, anything stated)
  - Previous tasks       (what was done, outcomes, retry patterns)
  - Learned behaviours   (which tool sequences work, average durations)
  - Failure risk signals (tasks/conditions with elevated failure probability)

The output is a short prose block (~200-400 tokens) injected directly into
the agent's system prompt.  Because Qwen decides what's relevant, the block
stays useful as memory grows — it won't bloat with irrelevant history.

Usage
-----
    from .memory_context import build_memory_context

    context_block = build_memory_context("memory_summary.json")
    system_prompt = get_prompts() + context_block
"""

import json
from collections import defaultdict
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage


# ── Prompt templates ───────────────────────────────────────────────────────────

# Maximum words Qwen is allowed to produce — controls injected prompt size.
# Raise this if you find the context too sparse; lower it to save tokens.
MAX_CONTEXT_WORDS = 350  # slightly raised to accommodate the new risk section

_SYSTEM_PROMPT = f"""You are a memory distillation assistant for an embodied robot agent.
Your job is to read a structured memory file and produce a compact, useful context block
that will be prepended to the robot agent's system prompt.

Rules:
- HARD LIMIT: {MAX_CONTEXT_WORDS} words maximum. Be ruthlessly concise. Cut anything not directly useful for the next task.
- Use clear sections with short headers.
- Only include facts that are actually present in the memory — never invent or assume.
- Prioritise: current physical state > failure risks > user preferences > recent task outcomes > learned patterns.
- For robot state, always list scene_objects (what is visible) separately from handled_objects (what has been grasped before). Make clear that scene_objects are valid pick targets even if not in handled_objects.
- Skip sections that have no data (e.g. omit "Robot State" if arm_pose is unknown).
- If an archived_summary exists, treat it as a condensed history — do not expand it.
- Write in a factual, direct tone suitable for an agent system prompt.
- Do NOT include meta-commentary like "the memory shows..." — just state the facts directly.

For the Failure Risk section specifically:
- A task type is HIGH RISK if its failure_rate exceeds 0.4 OR it has required 2+ retries in any single episode.
- A task type is MEDIUM RISK if its failure_rate is between 0.2–0.4 OR it has a known fragile precondition.
- Only list LOW RISK tasks if they share a tool with a HIGH/MEDIUM risk task (to flag contamination).
- For each risk entry state: the task type, the observed failure rate or retry count, and the most
  likely root cause or missing precondition if it can be inferred from the episode outcomes."""

_USER_PROMPT_TEMPLATE = """Here is the robot agent's memory file. Extract and summarise the most
relevant context for the agent's next session.

MEMORY FILE:
{memory_json}

Produce a context block with these sections (omit any section with no data):

## Current Robot State
(arm pose, gripper state, scene objects, previously handled objects)

**Important distinction:**
- `scene_objects` = objects currently visible in the workspace (from describe_environment). DO NOT assume these objects are still present. Always capture and describe a new image to confirm the scene.
- `handled_objects` = objects the robot has previously segmented or grasped. A subset of scene_objects.
- An object being absent from `handled_objects` does NOT mean it cannot be picked — if it appears in `scene_objects` or the scene description, it is a valid target.

## Failure Risk Signals
CRITICAL — read this before planning any task.
List task types ordered from highest to lowest failure risk. For each entry include:
- Risk level: HIGH / MEDIUM / LOW
- Task type name
- Failure rate or retry pattern (e.g. "failed 1/2 attempts", "always needs 1 retry")
- Most likely cause or fragile precondition (e.g. "gripper not reset between picks",
  "object_detection times out when >3 objects present")
If no failures or retries exist in memory, write: "No failure patterns recorded yet."

## User Preferences & Facts
(anything the user has told the agent about themselves or anything task that has been mentioned as critical or would be repeated in the future)

## Previous Tasks
(what tasks were performed, outcomes, any retries needed — last 3-5 most relevant)

Write only the context block, nothing else."""

# ## Learned Behaviours
# (which tool sequences work well, typical durations, anything worth knowing for future tasks)


# ── Main function ──────────────────────────────────────────────────────────────

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
    llm                : a LangChain chat model — pass get_qwen_llm() here.
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


# How many recent episodes to send to Qwen.
# world_state + user_facts + procedures are always sent in full.
# Archived summary (if present) replaces older episodes so this window
# only needs to cover genuinely recent activity.
EPISODE_WINDOW = 10


def _llm_format(summary: dict, llm, fallback: bool) -> str:
    """Send memory JSON to Qwen with an episode window and return the distilled context block."""

    # Always send world_state, user_facts, procedures, archived_summary in full.
    # Cap recent_episodes to the last EPISODE_WINDOW entries — older history is
    # already represented by archived_summary once compression has run.
    payload = {
        **{k: v for k, v in summary.items() if k != "recent_episodes"},
        "recent_episodes": summary.get("recent_episodes", [])[-EPISODE_WINDOW:],
        # Inject pre-computed failure stats so Qwen doesn't have to count
        "_failure_stats": _compute_failure_stats(summary.get("recent_episodes", [])),
    }
    memory_json_str = json.dumps(payload, indent=2, ensure_ascii=False)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_PROMPT_TEMPLATE.format(memory_json=memory_json_str)),
    ]

    try:
        # Pass max_tokens so the API hard-stops output at our word budget.
        # ~1.3 tokens/word is a safe estimate for English prose.
        response = llm.invoke(messages, max_tokens=int(MAX_CONTEXT_WORDS * 1.4))
        context  = response.content.strip()
        word_count = len(context.split())
        if word_count >= MAX_CONTEXT_WORDS * 0.95:
            print(f"[memory_context] WARNING: output near limit ({word_count}/{MAX_CONTEXT_WORDS} words) — consider raising MAX_CONTEXT_WORDS or running --compress")
        else:
            print(f"[memory_context] Qwen distilled memory: {word_count} words")
            print(context)
        return f"\n\n{context}\n"

    except Exception as e:
        print(f"[memory_context] LLM call failed: {e}")
        if fallback:
            print("[memory_context] Falling back to manual formatting.")
            return _manual_format(summary)
        raise


# ── Failure statistics helper ─────────────────────────────────────────────────

# Thresholds for risk classification used by both the LLM payload and manual formatter.
RISK_HIGH_FAILURE_RATE   = 0.40   # ≥40% failure rate → HIGH
RISK_MEDIUM_FAILURE_RATE = 0.20   # 20–39% → MEDIUM; <20% → LOW (only shown if tool overlap)
RISK_HIGH_RETRY_THRESHOLD   = 2   # any episode with ≥2 retries → HIGH regardless of rate
RISK_MEDIUM_RETRY_THRESHOLD = 1   # any episode with 1 retry → at least MEDIUM


def _compute_failure_stats(episodes: list[dict]) -> dict:
    """
    Aggregate per-task-type failure statistics from the episode list.

    Returns a dict keyed by task_type with the shape:
    {
        "pick_and_place": {
            "attempts":      5,
            "failures":      2,
            "failure_rate":  0.4,
            "max_retries":   2,          # worst single episode
            "total_retries": 3,
            "risk_level":    "HIGH",
            "failure_causes": ["gripper open at pick", "object not found"],
        },
        ...
    }
    Cause strings are taken from episode["outcome_fact"] on failed/retried episodes.
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "attempts": 0,
        "failures": 0,
        "max_retries": 0,
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
        s["max_retries"] = max(s["max_retries"], retries)

        if ep.get("outcome") != "success":
            s["failures"] += 1
            cause = ep.get("outcome_fact", "").strip()
            if cause and cause not in s["failure_causes"]:
                s["failure_causes"].append(cause)
        elif retries > 0:
            # Succeeded but only after retries — still a fragility signal.
            cause = ep.get("outcome_fact", "").strip()
            if cause and cause not in s["failure_causes"]:
                s["failure_causes"].append(f"[recovered] {cause}")

    # Compute derived fields and risk level.
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
    No LLM required — used when Qwen is unavailable.
    """
    ws    = summary.get("world_state", {})
    procs = summary.get("procedures", {})
    uf    = summary.get("user_facts", {})
    eps   = summary.get("recent_episodes", [])[-5:]

    lines = ["\n\n## Robot Memory\n"]

    # ── Robot state ───────────────────────────────────────────────────────────
    has_state = (ws.get("arm_pose") not in (None, "unknown")
                 or ws.get("gripper") not in (None, "unknown")
                 or ws.get("handled_objects")
                 or ws.get("scene_objects")
                 or ws.get("stack"))

    if has_state:
        lines.append("### Current Robot State")
        if ws.get("arm_pose") not in (None, "unknown"):
            lines.append(f"- Arm: {ws['arm_pose']}")
        if ws.get("gripper") not in (None, "unknown"):
            lines.append(f"- Gripper: {ws['gripper']}")
        if ws.get("scene_objects"):
            lines.append(f"- Objects visible in scene: {', '.join(ws['scene_objects'])}")
        if ws.get("handled_objects"):
            lines.append(f"- Previously handled (segmented/grasped): {', '.join(ws['handled_objects'])}")
        for i, tower in enumerate(ws.get("stacks", []), 1):
            label = f"Stack {i}" if len(ws.get("stacks", [])) > 1 else "Stack"
            lines.append(f"- {label} (bottom→top): {' > '.join(tower)}")

    # ── Failure risk signals ──────────────────────────────────────────────────
    all_eps   = summary.get("recent_episodes", [])
    fail_stats = _compute_failure_stats(all_eps)

    # Sort: HIGH first, then MEDIUM, then LOW — skip LOW unless there's a
    # tool-sequence overlap with a riskier task (keep the fallback simple:
    # just omit pure-LOW entries).
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
            cause_str = ""
            if s["failure_causes"]:
                # Show the most recent / most informative cause only.
                cause_str = f" — {s['failure_causes'][-1]}"
            lines.append(
                f"- [{s['risk_level']}] {task}: "
                f"{s['failures']}/{s['attempts']} failed ({rate_pct}){retry_str}{cause_str}"
            )

    # ── User preferences ──────────────────────────────────────────────────────
    if uf:
        lines.append("\n### User Preferences & Facts")
        for key, value in uf.items():
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {value}")

    # ── Previous tasks ────────────────────────────────────────────────────────
    meaningful_eps = [e for e in eps if e.get("task_type") != "other"]
    if meaningful_eps:
        lines.append("\n### Previous Tasks")
        for ep in meaningful_eps:
            retries = ep.get("key_args", {}).get("retries", 0)
            retry_s = f" ({retries} retr{'ies' if retries != 1 else 'y'})" if retries else ""
            status  = "✓" if ep["outcome"] == "success" else "✗"
            lines.append(f"- {status}{retry_s} [{ep['task_type']}] {ep['outcome_fact']}")

    # ── Learned behaviours ────────────────────────────────────────────────────
    real_procs = {k: v for k, v in procs.items() if k != "other" and v.get("tool_sequence")}
    if real_procs:
        lines.append("\n### Learned Behaviours")
        for task, proc in real_procs.items():
            seq = " → ".join(proc["tool_sequence"])
            dur = f"~{proc['avg_duration_s']}s" if proc.get("avg_duration_s") else ""
            lines.append(f"- {task} ({dur}): {seq}")

    return "\n".join(lines) + "\n"