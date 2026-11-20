"""Security layers: input guardrail, content sanitizer (indirect injection), output guardrail."""
import re
import unicodedata
from typing import Optional

from core.config import VALID_CLEARANCES, message_text
from core.state import HospitalAgentState, reset_fragment

# ═════════════════════════════════════════════════════════════════════════════
# Prompt-injection detection (shared by input guardrail, retriever, sanitizer)
# ═════════════════════════════════════════════════════════════════════════════
# Direct markers in the user's message (first line of defense, not the only one)
JAILBREAK_KEYWORDS = [
    "ignore previous instructions",
    "disregard prior directions",
    "system update:",
    "bypass security",
    "override rbac",
    "you are now",
    "act as if you are",
]

# Indirect markers hidden inside retrieved documents / web pages
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions?",
    r"disregard\s+(all\s+)?(prior|previous)\s+(rules|directions?)",
    r"system\s*(prompt|update|override)\s*:",
    r"override\s+(permissions?|rbac|security)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
]


def normalize_for_scan(text: str) -> str:
    """Strip zero-width chars, Unicode confusables, and HTML comments before scanning."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.lower()


def scan_for_injection(text: str) -> Optional[str]:
    """Return the matched pattern if an injection marker is found, else None."""
    normalized = normalize_for_scan(text)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return pattern
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Input guardrail
# ═════════════════════════════════════════════════════════════════════════════
def input_guardrail_node(state: HospitalAgentState) -> dict:
    """Zero-trust entry check: identity, clearance validity, prompt-injection scan."""
    messages = state.get("messages") or []
    latest_message = message_text(messages[-1]) if messages else ""
    user_id = state.get("user_id", "")
    user_clearance = state.get("user_clearance", "")

    if not user_id:
        return {
            "is_safe": False,
            "final_output": "SECURITY WARNING: Missing user identity. Request blocked.",
            "intent": "blocked",
            "agent_status": {"input_guardrail": "no_identity"},
        }

    if user_clearance not in VALID_CLEARANCES:
        return {
            "is_safe": False,
            "final_output": (
                f"SECURITY WARNING: Unknown clearance level '{user_clearance}'. "
                f"Request blocked."
            ),
            "intent": "blocked",
            "agent_status": {"input_guardrail": "invalid_clearance"},
        }

    lowered = latest_message.lower()
    detected = next((kw for kw in JAILBREAK_KEYWORDS if kw in lowered), None)
    if detected:
        print(f"🚨 SECURITY ALERT: Prompt injection pattern detected ('{detected}')")
        return {
            "is_safe": False,
            "final_output": (
                "SECURITY WARNING: Your input triggered our automated prompt "
                "injection guardrail. Request blocked."
            ),
            "intent": "blocked",
            "agent_status": {"input_guardrail": f"injection:{detected}"},
        }

    print("✅ Input passed security guardrail verification.")
    # `intent` is written only by the supervisor.
    return {"is_safe": True, "agent_status": {"input_guardrail": "ok"}}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Content sanitizer (web / external retrievers)
# ═════════════════════════════════════════════════════════════════════════════
# Policy retriever sanitizes inline, so only external sources are scanned here.
SANITIZABLE_SOURCES = {"web_search", "external_web_search"}


def content_sanitizer_node(state: HospitalAgentState) -> dict:
    """Drop fragments from external sources that contain injection markers."""
    fragments = state.get("retrieved_context") or []
    if not fragments:
        return {"agent_status": {"content_sanitizer": "empty"}}

    kept: list = []
    blocked = 0
    for frag in fragments:
        source = frag.get("source", "")
        if source not in SANITIZABLE_SOURCES:
            kept.append(frag)
            continue

        matched = scan_for_injection(frag.get("content", ""))
        if matched:
            blocked += 1
            print(f"🚨 Injection in {source} fragment (pattern: {matched}) — dropping it.")
            continue
        kept.append(frag)

    status = f"blocked:{blocked}" if blocked else "ok"
    print(f"🧼 Content sanitizer done: {len(fragments)} → {len(kept)} fragment(s) [{status}]")

    # `merge_contexts` APPENDS, so returning `kept` alone would duplicate every fragment
    # and leave the blocked ones in state. The reset marker makes the reducer REPLACE.
    return {
        "retrieved_context": [reset_fragment(), *kept],
        "agent_status": {"content_sanitizer": status},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. Output guardrail & PII redaction
# ═════════════════════════════════════════════════════════════════════════════
# True: redact in place (soft block). False: replace the whole response (hard block).
GUARDRAIL_REDACT_MODE = True
# Prefix a short notice when something was redacted.
GUARDRAIL_NOTICE_USER = True

LEAK_PATTERNS = [
    (r"MEDIC[_ ](ASTRO|NOVA)\s+(PROMPT|SYSTEM|INSTRUCTIONS?)", "[REDACTED_LEAK]"),
    (r"SUPERVISOR_SYSTEM_PROMPT|MEDIC_ASTRO_PROMPT|MEDIC_NOVA_PROMPT", "[REDACTED_LEAK]"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "[REDACTED_API_KEY]"),            # OpenAI-style
    (r"\bsk-or-v1-[A-Za-z0-9]{20,}\b", "[REDACTED_API_KEY]"),      # OpenRouter
    (r"\bpcsk_[A-Za-z0-9_]{20,}\b", "[REDACTED_API_KEY]"),         # Pinecone
    (r"PINECONE_API_KEY|OPENROUTER_API_KEY|DEEPSEEK_API_KEY", "[REDACTED_SECRET]"),
    (r"\b(?:tool|function)_(?:call|name)\s*[:=]\s*\w+", "[REDACTED_TOOL]"),
]

PHI_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    (r"\bMRN[:\s#]*\d{6,10}\b", "[REDACTED_MRN]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[REDACTED_PHONE]"),
    (r"(?:DOB|born|date of birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "[REDACTED_DOB]"),
    (r"\[REDACTED_PHI\][^.\n]{0,60}(diagnos|treat|prescr)", "[REDACTED_PHI_CONTEXT]"),
]

# Model ignored an access denial and kept describing the withheld data
DENIAL_PATTERNS = [
    (r"ACCESS\s+DENIED[^.\n]*\.?\s*(?:However|But|Although)[^.]*\.",
     "[REDACTED_DENIAL_RECONSTRUCTION]"),
]

_CATEGORIES = {
    "leak": [(re.compile(p, re.IGNORECASE), r) for p, r in LEAK_PATTERNS],
    "phi": [(re.compile(p, re.IGNORECASE), r) for p, r in PHI_PATTERNS],
    "denial": [(re.compile(p, re.IGNORECASE), r) for p, r in DENIAL_PATTERNS],
}


def scan_and_redact(text: str):
    """Apply all pattern sets. Returns (sanitized_text, {category: hit_count})."""
    hits = {category: 0 for category in _CATEGORIES}
    sanitized = text
    for category, patterns in _CATEGORIES.items():
        for regex, replacement in patterns:
            sanitized, n = regex.subn(replacement, sanitized)
            hits[category] += n
    return sanitized, hits


def output_guardrail_node(state: HospitalAgentState) -> dict:
    """Final safety filter on final_output before it reaches the user."""
    raw_output = state.get("final_output") or ""

    if not raw_output.strip():
        print("⚠️ Output guardrail received empty final_output.")
        return {
            "final_output": "I could not produce a valid response. Please rephrase your question.",
            "agent_status": {"output_guardrail": "empty_input"},
        }

    sanitized, hits = scan_and_redact(raw_output)
    total_hits = sum(hits.values())

    if total_hits == 0:
        print("✅ Output passed guardrail verification.")
        return {"final_output": sanitized, "agent_status": {"output_guardrail": "ok"}}

    print(f"🚨 Output guardrail blocked {total_hits} pattern(s): {hits}")
    hit_flags = {f"hits_{k}": v for k, v in hits.items() if v}

    if not GUARDRAIL_REDACT_MODE:
        return {
            "final_output": (
                "⚠️ The response was withheld because it contained information that could not "
                "be safely disclosed. Please rephrase your question or contact the appropriate "
                "department."
            ),
            "agent_status": {"output_guardrail": f"hard_block:{total_hits}", **hit_flags},
        }

    if GUARDRAIL_NOTICE_USER:
        categories_hit = [k for k, v in hits.items() if v]
        notice = f"⚠️ Note: portions of this response were sanitized ({', '.join(categories_hit)})."
        sanitized = f"{notice}\n\n{sanitized}"

    return {
        "final_output": sanitized,
        "agent_status": {"output_guardrail": f"redacted:{total_hits}", **hit_flags},
    }