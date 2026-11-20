"""LangGraph state schema, custom reducers, and a helper to build per-turn input."""
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages

RESET_KEY = "__reset__"


def reset_fragment() -> dict:
    """Marker that tells `merge_contexts` to REPLACE the context instead of appending."""
    return {"source": RESET_KEY, "content": "", "metadata": {}}


# ── Reducers ─────────────────────────────────────────────────────────────────
def merge_contexts(left, right):
    """Append-only within a turn; a reset marker in `right` replaces the history."""
    if right and any(isinstance(f, dict) and f.get("source") == RESET_KEY for f in right):
        return [f for f in right if f.get("source") != RESET_KEY]
    return (left or []) + (right or [])


def merge_flags(left: Optional[dict], right: Optional[dict]) -> dict:
    """Merge per-agent status flags; a `__reset__` key in `right` starts a fresh dict."""
    right = right or {}
    if right.get(RESET_KEY):
        return {k: v for k, v in right.items() if k != RESET_KEY}
    return {**(left or {}), **right}


def keep_last_non_none(left, right):
    """Scalar fields: keep the latest non-None value."""
    return right if right is not None else left


# ── State ────────────────────────────────────────────────────────────────────
class HospitalAgentState(TypedDict):
    # Chat history — append-only via add_messages
    messages: Annotated[list, add_messages]

    # Caller identity (set once at entry)
    user_id: str
    user_clearance: str
    user_department: Optional[str]

    # Orchestration
    intent: Annotated[Optional[str], keep_last_non_none]
    is_safe: Annotated[bool, keep_last_non_none]
    next_agent: Annotated[Optional[str], keep_last_non_none]

    # Every retriever APPENDS: {"source": ..., "content": ..., "metadata": {...}}
    retrieved_context: Annotated[List[dict], merge_contexts]

    # Per-agent diagnostics, e.g. {"policy_retriever": "ok"}
    agent_status: Annotated[dict, merge_flags]

    final_output: Annotated[Optional[str], keep_last_non_none]
    revision_count: Annotated[int, keep_last_non_none]


def build_initial_state(message: str, user_id: str, clearance: str, department: Optional[str]) -> dict:
    """Input for one conversation turn. Resets per-turn fields so nothing leaks between turns."""
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "user_clearance": clearance,
        "user_department": department,
        "retrieved_context": [reset_fragment()],
        "agent_status": {RESET_KEY: True},
        "final_output": "",
        "revision_count": 0,
    }