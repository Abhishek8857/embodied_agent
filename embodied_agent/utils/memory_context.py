"""
memory_context.py
-----------------
Uses Qwen (via OpenRouter) to read the full memory_summary.json and produce
a compact, agent-ready context string covering:

  - Current robot state  (arm pose, gripper, known objects, stack)
  - User preferences     (favorite color, name, anything stated)
  - Previous tasks       (what was done, outcomes, retry patterns)
  - Learned behaviours   (which tool sequences work, average durations)
  - Any other useful facts the LLM deems relevant

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
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage


# ── Prompt templates ───────────────────────────────────────────────────────────

# Maximum words Qwen is allowed to produce — controls injected prompt size.
# Raise this if you find the context too sparse; lower it to save tokens.
MAX_CONTEXT_WORDS = 300

_SYSTEM_PROMPT = f"""You are a memory distillation assistant for an embodied robot agent.
Your job is to read a structured memory file and produce a compact, useful context block
that will be prepended to the robot agent's system prompt.

Rules:
- HARD LIMIT: {MAX_CONTEXT_WORDS} words maximum. Be ruthlessly concise. Cut anything not directly useful for the next task.
- Use clear sections with short headers.
- Only include facts that are actually present in the memory — never invent or assume.
- Prioritise: current physical state > user preferences > recent task outcomes > learned patterns.
- Skip sections that have no data (e.g. omit "Robot State" if arm_pose is unknown).
- If an archived_summary exists, treat it as a condensed history — do not expand it.
- Write in a factual, direct tone suitable for an agent system prompt.
- Do NOT include meta-commentary like "the memory shows..." — just state the facts directly."""

_USER_PROMPT_TEMPLATE = """Here is the robot agent's memory file. Extract and summarise the most
relevant context for the agent's next session.

MEMORY FILE:
{memory_json}

Produce a context block with these sections (omit any section with no data):

## Current Robot State
(arm pose, gripper state, known objects, stack configuration)

## User Preferences & Facts
(anything the user has told the agent about themselves or anything task that has been mentioned as critical or would be repeated in the future )

## Previous Tasks
(what tasks were performed, outcomes, any retries needed — last 3-5 most relevant)

## Learned Behaviours
(which tool sequences work well, typical durations, anything worth knowing for future tasks)

Write only the context block, nothing else."""


# ── Main function ──────────────────────────────────────────────────────────────

def build_memory_context(
    summary_path: str | Path = "memory_summary.json",
    llm=None,
    fallback_to_manual: bool = True,
) -> str:
    """
    Read memory_summary.json and return a compact context string for the agent.

    Parameters
    ----------
    summary_path       : path to memory_summary.json (from memory_summarizer.py)
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
EPISODE_WINDOW = 15


def _llm_format(summary: dict, llm, fallback: bool) -> str:
    """Send memory JSON to Qwen with an episode window and return the distilled context block."""

    # Always send world_state, user_facts, procedures, archived_summary in full.
    # Cap recent_episodes to the last EPISODE_WINDOW entries — older history is
    # already represented by archived_summary once compression has run.
    payload = {
        **{k: v for k, v in summary.items() if k != "recent_episodes"},
        "recent_episodes": summary.get("recent_episodes", [])[-EPISODE_WINDOW:],
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
                 or ws.get("known_objects")
                 or ws.get("stack"))

    if has_state:
        lines.append("### Current Robot State")
        if ws.get("arm_pose") not in (None, "unknown"):
            lines.append(f"- Arm: {ws['arm_pose']}")
        if ws.get("gripper") not in (None, "unknown"):
            lines.append(f"- Gripper: {ws['gripper']}")
        if ws.get("known_objects"):
            lines.append(f"- Known objects: {', '.join(ws['known_objects'])}")
        for i, tower in enumerate(ws.get("stacks", []), 1):
            label = f"Stack {i}" if len(ws.get("stacks", [])) > 1 else "Stack"
            lines.append(f"- {label} (bottom→top): {' > '.join(tower)}")

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