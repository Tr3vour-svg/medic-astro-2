"""Graph assembly: nodes, conditional routing, checkpointer, compile."""
import os
import sqlite3
import tempfile
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from core.config import SENSITIVE_INTENTS
from core.guardrails import content_sanitizer_node, input_guardrail_node, output_guardrail_node
from core.nodes import (
    hr_salary_agent_node,
    human_review_node,
    medic_astro_node,
    medic_nova_node,
    medic_reviewer_node,
    patient_agent_node,
    policy_retriever_node,
    staff_agent_node,
    web_search_node,
)
from core.state import HospitalAgentState


# ═════════════════════════════════════════════════════════════════════════════
# Conditional edges
# ═════════════════════════════════════════════════════════════════════════════
def route_after_guardrail(state: HospitalAgentState) -> str:
    """If the input guardrail blocked the request, skip straight to END."""
    return "medic_astro" if state.get("is_safe", True) else "end_blocked"


def route_from_medic_astro(state: HospitalAgentState) -> str:
    if not state.get("is_safe", True):
        return "end_blocked"

    mapping = {
        "policy":     "policy_retriever",
        "hr_salary":  "hr_salary_agent",
        "staff":      "staff_agent",
        "patient":    "patient_agent",
        "web_search": "web_search_agent",
        "general":    "medic_nova",
        "blocked":    "end_blocked",
    }
    return mapping.get(state.get("intent", "general"), "medic_nova")


def route_after_reviewer(state: HospitalAgentState) -> str:
    """
    rejected                    -> back to MEDIC NOVA (the reviewer node itself stops
                                   rejecting once MAX_REVISIONS is reached)
    approved + sensitive intent -> human review
    approved + other intents    -> output guardrail
    """
    status = (state.get("agent_status") or {}).get("medic_reviewer", "")
    intent = state.get("intent", "general")

    if status == "rejected":
        print("↩️  Reviewer rejected — routing back to MEDIC NOVA for revision.")
        return "medic_nova"

    if intent in SENSITIVE_INTENTS:
        print(f"🧑‍⚕️ Sensitive intent '{intent}' — escalating to human review.")
        return "human_review"

    return "output_guardrail"


def route_after_human(state: HospitalAgentState) -> str:
    """If the human rejected the draft, loop back to MEDIC NOVA."""
    status = (state.get("agent_status") or {}).get("human_review", "")
    if status == "rejected":
        print("🚫 Human rejected the draft — looping back to MEDIC NOVA.")
        return "medic_nova"
    return "output_guardrail"


# ═════════════════════════════════════════════════════════════════════════════
# Checkpointer
# ═════════════════════════════════════════════════════════════════════════════
# DEV_MODE=true  -> throwaway checkpoint DB in the OS temp dir (fresh each run)
# DEV_MODE=false -> persistent DB at CHECKPOINT_DB_PATH (default: project root)
DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"

if DEV_MODE:
    CHECKPOINT_DB_PATH = os.path.join(
        tempfile.gettempdir(), f"hospital_agent_dev_{uuid.uuid4().hex[:6]}.sqlite"
    )
else:
    CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "hospital_agent_checkpoints.sqlite")


# ═════════════════════════════════════════════════════════════════════════════
# Build & compile
# ═════════════════════════════════════════════════════════════════════════════
def build_app():
    graph = StateGraph(HospitalAgentState)

    # Nodes
    graph.add_node("input_guardrail",   input_guardrail_node)
    graph.add_node("medic_astro",       medic_astro_node)
    graph.add_node("policy_retriever",  policy_retriever_node)
    graph.add_node("web_search_agent",  web_search_node)
    graph.add_node("content_sanitizer", content_sanitizer_node)
    graph.add_node("staff_agent",       staff_agent_node)
    graph.add_node("hr_salary_agent",   hr_salary_agent_node)
    graph.add_node("patient_agent",     patient_agent_node)
    graph.add_node("medic_nova",        medic_nova_node)
    graph.add_node("medic_reviewer",    medic_reviewer_node)
    graph.add_node("human_review",      human_review_node)
    graph.add_node("output_guardrail",  output_guardrail_node)

    graph.set_entry_point("input_guardrail")

    # Guardrail -> supervisor
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"medic_astro": "medic_astro", "end_blocked": END},
    )

    # Supervisor -> specialists
    graph.add_conditional_edges(
        "medic_astro",
        route_from_medic_astro,
        {
            "policy_retriever": "policy_retriever",
            "hr_salary_agent":  "hr_salary_agent",
            "staff_agent":      "staff_agent",
            "patient_agent":    "patient_agent",
            "web_search_agent": "web_search_agent",
            "medic_nova":       "medic_nova",
            "end_blocked":      END,
        },
    )

    # Specialists -> synthesizer
    graph.add_edge("policy_retriever",  "medic_nova")
    graph.add_edge("hr_salary_agent",   "medic_nova")
    graph.add_edge("patient_agent",     "medic_nova")
    graph.add_edge("staff_agent",       "medic_nova")
    graph.add_edge("web_search_agent",  "content_sanitizer")
    graph.add_edge("content_sanitizer", "medic_nova")

    # Synthesizer -> reviewer -> (revise | human | output)
    graph.add_edge("medic_nova", "medic_reviewer")
    graph.add_conditional_edges(
        "medic_reviewer",
        route_after_reviewer,
        {
            "medic_nova":       "medic_nova",
            "human_review":     "human_review",
            "output_guardrail": "output_guardrail",
        },
    )

    # Human review -> (revise | output)
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"medic_nova": "medic_nova", "output_guardrail": "output_guardrail"},
    )

    graph.add_edge("output_guardrail", END)

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


app = build_app()
print(f"✅ Graph compiled ({'dev, ephemeral' if DEV_MODE else 'prod, persistent'} checkpointer: {CHECKPOINT_DB_PATH})")