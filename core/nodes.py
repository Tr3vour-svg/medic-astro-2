"""LangGraph nodes: supervisor, specialist agents, synthesizer, reviewer, human-in-the-loop."""
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from core.config import (
    MAX_REVISIONS,
    SENSITIVE_INTENTS,
    extractor_llm,
    message_text,
    reranker,
    reviewer_llm,
    supervisor_llm,
    synthesizer_llm,
    vector_store,
)
from core.guardrails import scan_for_injection
from core.state import HospitalAgentState
from core.tools import (
    WEB_CACHE_TTL_SECONDS,
    WEB_MAX_ATTEMPTS,
    WEB_MAX_RESULT_CHARS,
    WEB_NEG_CACHE_TTL_SECONDS,
    cache_get,
    cache_key,
    cache_put,
    format_patient,
    format_salary,
    format_staff_table,
    normalize_emp_id,
    query_patient_record,
    query_salary_record,
    query_staff_records,
    query_staff_records_by_emp_id,
    search_with_retries,
)

REVISION_MARKER = "[REVISION NEEDED]"


# ═════════════════════════════════════════════════════════════════════════════
# 1. MEDIC ASTRO — supervisor / router
# ═════════════════════════════════════════════════════════════════════════════
VALID_INTENTS = {"policy", "hr_salary", "staff", "patient", "web_search", "general"}

MEDIC_ASTRO_PROMPT = (
    "You are MEDIC ASTRO, the Chief Medical Supervisor Agent for a secure hospital. "
    "Your job is to route the user's LATEST message to exactly one specialist agent.\n\n"
    "Categories:\n"
    "1. 'policy'     -> Hospital guidelines, rules, SOPs, compliance, code protocols, handbooks.\n"
    "                   e.g. 'What is the leave policy?', 'What is the escalation SOP?'\n"
    "2. 'hr_salary'  -> Salary, compensation, bonuses, allowances for a specific employee.\n"
    "                   e.g. 'What is the salary for EMP-000001?'\n"
    "3. 'staff'      -> Staff directory, who works where, roles, reporting lines.\n"
    "                   e.g. 'Who is the head of Cardiology?', 'Find EMP-000009'.\n"
    "4. 'patient'    -> Patient records, admissions, medical notes, billing status, "
    "AND the physician assigned to the patient.\n"
    "                   e.g. 'Where is PAT-0000001 admitted?'\n"
    "5. 'web_search' -> External medical literature, definitions, or referral hospitals "
    "NOT in our internal system.\n"
    "                   e.g. 'What are the latest WHO sepsis guidelines?'\n"
    "6. 'general'    -> Greetings, chit-chat, questions about you or your abilities, or queries "
    "that fit NONE of the above.\n"
    "                   e.g. 'Hi', 'What is your name?', 'What can you do?'\n\n"
    "RULES:\n"
    "- When in doubt between 'policy' and 'general', ALWAYS prefer 'policy'. "
    "Hospital questions are usually operational.\n"
    "- The caller's clearance is provided. Use it ONLY to decide if a sensitive intent "
    "(like 'hr_salary') is appropriate.\n\n"
    "Respond with ONLY one exact keyword: policy, hr_salary, staff, patient, web_search, general."
)


def _parse_intent(raw: str) -> str:
    """Extract the first valid intent token from a possibly noisy LLM response."""
    for token in (raw or "").lower().split():
        token = token.strip(".,!?\"'`")
        if token in VALID_INTENTS:
            return token
    return ""


def medic_astro_node(state: HospitalAgentState) -> dict:
    """Classify the latest message. Single writer of `intent`."""
    messages = state.get("messages") or []
    if not messages:
        return {"intent": "general", "agent_status": {"medic_astro": "empty_input"}}

    user_clearance = state.get("user_clearance", "Unknown")
    supervisor_messages = [
        SystemMessage(content=MEDIC_ASTRO_PROMPT),
        SystemMessage(content=f"Caller clearance level: {user_clearance}"),
        messages[-1],  # only the latest message, not the whole history
    ]

    raw_content = "NO_RESPONSE"
    try:
        response = supervisor_llm.invoke(supervisor_messages)
        raw_content = message_text(response)
        detected_intent = _parse_intent(raw_content)
    except Exception as e:
        print(f"⚠️ MEDIC ASTRO routing error: {e}")
        detected_intent = ""

    if detected_intent not in VALID_INTENTS:
        print(f"⚠️ MEDIC ASTRO could not parse intent (raw={raw_content!r}). Defaulting to 'general'.")
        detected_intent = "general"

    print(f"🧭 MEDIC ASTRO routed intent -> [{detected_intent.upper()}]")
    return {
        "intent": detected_intent,
        "agent_status": {"medic_astro": f"routed:{detected_intent}"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. Policy retriever (vector search -> rerank -> inline sanitize)
# ═════════════════════════════════════════════════════════════════════════════
INITIAL_CANDIDATE_K = 15      # recall before rerank
MIN_RERANK_SCORE = 0.001      # bge-reranker-v2-m3 returns small scores
MIN_RELEVANCE_SCORE = 0.10    # only used on the vector-score fallback path

KNOWN_DEPARTMENTS = {
    "Administration", "Cardiology", "Neurology", "Pediatrics",
    "Oncology", "Radiology", "Emergency", "ICU", "HR",
    "Nursing", "Surgery", "Pharmacy",
    "All",
}


def _build_retrieval_query(state: HospitalAgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    raw = message_text(messages[-1])
    return re.sub(r"\s+", " ", raw).strip().rstrip("?.! ")


def _resolve_department(user_department: Optional[str]) -> Optional[str]:
    """Return a department that exists in the index, or None (skip filter)."""
    if not user_department:
        return None
    if user_department in KNOWN_DEPARTMENTS:
        return user_department
    for d in KNOWN_DEPARTMENTS:
        if d.lower() == user_department.lower():
            return d
    print(f"⚠️ '{user_department}' is not a known department — skipping filter.")
    return None


def _retrieve_with_scores(query: str, k: int, department: Optional[str]) -> list:
    """Fallback retriever used only if the reranker fails."""
    filter_expr = {"department": {"$in": ["All", department]}} if department else None
    try:
        return vector_store.similarity_search_with_relevance_scores(query, k=k, filter=filter_expr)
    except Exception as e:
        print(f"⚠️ Scored retrieval failed, falling back to plain search: {e}")
        try:
            docs = vector_store.similarity_search(query, k=k, filter=filter_expr)
            return [(d, 0.5) for d in docs]
        except Exception:
            return []


def policy_retriever_node(state: HospitalAgentState) -> dict:
    query = _build_retrieval_query(state)
    if not query:
        return {"retrieved_context": [], "agent_status": {"policy_retriever": "empty_query"}}

    user_department = _resolve_department(state.get("user_department"))
    print(f"📥 Policy retrieval: '{query}' (dept={user_department or 'unfiltered'})")

    # ── Stage 1: vector search ──────────────────────────────────────────
    candidate_docs = []
    filter_used = "unfiltered"
    if user_department:
        try:
            filter_expr = {"department": {"$in": ["All", user_department]}}
            candidate_docs = vector_store.similarity_search(
                query, k=INITIAL_CANDIDATE_K, filter=filter_expr
            )
            if candidate_docs:
                filter_used = f"dept={user_department}"
        except Exception as e:
            print(f"⚠️ Filtered search failed: {e}")

    if not candidate_docs:
        if user_department:
            print("⚠️ Filtered search returned 0 — falling back to unfiltered.")
            filter_used = "unfiltered_fallback"
        try:
            candidate_docs = vector_store.similarity_search(query, k=INITIAL_CANDIDATE_K)
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return {
                "retrieved_context": [],
                "agent_status": {"policy_retriever": "error:vector_search"},
            }

    if not candidate_docs:
        return {"retrieved_context": [], "agent_status": {"policy_retriever": "no_match"}}

    # ── Stage 2: rerank (with fallback) ─────────────────────────────────
    used_reranker = False
    try:
        reranked_docs = reranker.compress_documents(documents=candidate_docs, query=query)
        used_reranker = True
        normalized = [
            {
                "text": d.page_content,
                "metadata": d.metadata,
                "score": float(d.metadata.get("relevance_score", 0.0)),
            }
            for d in reranked_docs
        ]
    except Exception as e:
        print(f"⚠️ Rerank failed: {e}. Falling back to vector scores.")
        scored = _retrieve_with_scores(
            query, k=5,
            department=user_department if filter_used.startswith("dept=") else None,
        )
        if scored:
            normalized = [
                {"text": d.page_content, "metadata": d.metadata, "score": s} for d, s in scored
            ]
        else:
            normalized = [
                {"text": d.page_content, "metadata": d.metadata, "score": 0.5}
                for d in candidate_docs[:5]
            ]

    # ── Stage 3: sanitize + build fragments ─────────────────────────────
    fragments = []
    seen_hashes = set()
    blocked = 0
    score_floor = MIN_RERANK_SCORE if used_reranker else MIN_RELEVANCE_SCORE

    for item in normalized:
        if item["score"] < score_floor:
            continue

        content = item["text"]
        h = hash(content[:200])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        if scan_for_injection(content):
            blocked += 1
            print(f"🚨 Indirect injection in chunk {item['metadata'].get('chunk_id')} — dropping it.")
            continue

        fragments.append({
            "source": "policy_retriever",
            "content": content,
            "metadata": {
                **item["metadata"],
                ("rerank_score" if used_reranker else "vector_score"): round(item["score"], 4),
            },
        })

    status = "ok" if fragments else "no_match"
    if blocked:
        status += f"|sanitized:{blocked}"
    if not used_reranker:
        status += "|rerank_fallback"
    if filter_used.endswith("fallback"):
        status += "|unfiltered"

    print(f"✅ Returning {len(fragments)} fragment(s) [{status}]")
    return {"retrieved_context": fragments, "agent_status": {"policy_retriever": status}}


# ═════════════════════════════════════════════════════════════════════════════
# 3. HR salary agent
# ═════════════════════════════════════════════════════════════════════════════
SALARY_ALLOWED_CLEARANCES = {"Executive", "Department Head"}
EMP_ID_RE = re.compile(r"\bEMP-\d{6}\b", re.IGNORECASE)


def _extract_emp_id(raw: str) -> Optional[str]:
    """Regex first; fall back to a small LLM extraction."""
    match = EMP_ID_RE.search(raw)
    if match:
        return match.group(0).upper()
    try:
        resp = extractor_llm.invoke([
            SystemMessage(content=(
                "Extract the employee ID (format EMP-000000) from the message. "
                "If none is present, reply with exactly 'NONE'."
            )),
            HumanMessage(content=raw),
        ])
        candidate = message_text(resp).strip().upper()
        if re.fullmatch(r"EMP-\d{6}", candidate):
            return candidate
    except Exception as e:
        print(f"⚠️ emp_id extraction failed: {e}")
    return None


def hr_salary_agent_node(state: HospitalAgentState) -> dict:
    messages = state.get("messages") or []
    user_clearance = state.get("user_clearance", "Unknown")

    # 1) Authorization gate
    if user_clearance not in SALARY_ALLOWED_CLEARANCES:
        print(f"🚫 Salary access denied for clearance '{user_clearance}'.")
        return {
            "retrieved_context": [{
                "source": "hr_salary_agent",
                "content": (
                    f"ACCESS DENIED: Clearance level '{user_clearance}' is not authorized "
                    f"to view salary records. Only Executive and Department Head may do so."
                ),
                "metadata": {"denied": True, "clearance": user_clearance},
            }],
            "agent_status": {"hr_salary_agent": "denied"},
        }

    # 2) Identify the employee
    raw = message_text(messages[-1]) if messages else ""
    emp_id = _extract_emp_id(raw)
    if not emp_id:
        return {
            "retrieved_context": [{
                "source": "hr_salary_agent",
                "content": "I need an employee ID (e.g. EMP-000001) to look up salary information.",
                "metadata": {"missing": "emp_id"},
            }],
            "agent_status": {"hr_salary_agent": "missing_emp_id"},
        }

    # 3) Query Turso
    print(f"💼 Querying salary for {emp_id} (Turso)...")
    try:
        record = query_salary_record(emp_id)
        if record is None:
            result, status = f"No salary record found for Employee ID {emp_id}.", "not_found"
        else:
            result, status = format_salary(emp_id, record), "ok"
    except Exception as e:
        print(f"⚠️ Salary lookup failed: {e}")  # keep infra details out of user-facing text
        result, status = "Error accessing salary data.", "error"

    return {
        "retrieved_context": [{
            "source": "hr_salary_agent",
            "content": result,
            "metadata": {"emp_id": emp_id},
        }],
        "agent_status": {"hr_salary_agent": status},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. Patient records agent
# ═════════════════════════════════════════════════════════════════════════════
PATIENT_ALLOWED_CLEARANCES = {"Executive", "Department Head", "Physician", "Intern"}
PATIENT_ID_RE = re.compile(r"\bPAT-\d{7}\b", re.IGNORECASE)


def patient_agent_node(state: HospitalAgentState) -> dict:
    messages = state.get("messages") or []
    user_clearance = state.get("user_clearance", "Unknown")

    # 1) Authorization gate
    if user_clearance not in PATIENT_ALLOWED_CLEARANCES:
        return {
            "retrieved_context": [{
                "source": "patient_agent",
                "content": f"ACCESS DENIED: clearance '{user_clearance}' cannot view patient records.",
                "metadata": {"denied": True},
            }],
            "agent_status": {"patient_agent": "denied"},
        }

    # 2) Extract PAT-XXXXXXX
    raw = message_text(messages[-1]) if messages else ""
    match = PATIENT_ID_RE.search(raw)
    patient_id = match.group(0).upper() if match else None
    if not patient_id:
        return {
            "retrieved_context": [{
                "source": "patient_agent",
                "content": "I need a patient ID (e.g. PAT-0000001) to look up records.",
                "metadata": {"missing": "patient_id"},
            }],
            "agent_status": {"patient_agent": "missing_patient_id"},
        }

    # 3) Query patients DB + staff DB (PHI masking happens in format_patient)
    print(f"🏥 Querying patient {patient_id} + physician details (cross-DB)...")
    try:
        record = query_patient_record(patient_id)
        if record is None:
            result, status = f"No patient found with ID {patient_id}.", "not_found"
        else:
            result, status = format_patient(patient_id, record), "ok"
    except Exception as e:
        print(f"⚠️ Patient lookup failed: {e}")
        result, status = "Error accessing patient data.", "error"

    return {
        "retrieved_context": [{
            "source": "patient_agent",
            "content": result,
            "metadata": {"patient_id": patient_id},
        }],
        "agent_status": {"patient_agent": status},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Staff directory agent
# ═════════════════════════════════════════════════════════════════════════════
STAFF_CROSS_DEPT_CLEARANCES = {"Executive", "Department Head"}


def _extract_staff_filters(raw: str):
    """LLM extraction of (department, role, name) filters; any may be None."""
    try:
        resp = extractor_llm.invoke([
            SystemMessage(content=(
                "Extract search filters from the staff-directory query. "
                "Reply with ONLY a JSON object with these keys (use null when absent):\n"
                '{"department": "...", "role": "...", "name": "..."}\n'
                "Do not add commentary or markdown."
            )),
            HumanMessage(content=raw),
        ])
        text = message_text(resp).strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(text[start:end + 1])
            return parsed.get("department") or None, parsed.get("role") or None, parsed.get("name") or None
    except Exception as e:
        print(f"⚠️ Staff filter extraction failed: {e}")
    return None, None, None


def staff_agent_node(state: HospitalAgentState) -> dict:
    messages = state.get("messages") or []
    user_clearance = state.get("user_clearance", "Unknown")
    user_department = state.get("user_department") or None
    user_id = normalize_emp_id(state.get("user_id") or "")
    raw = message_text(messages[-1]) if messages else ""

    # 1) Filters
    department, role, name = _extract_staff_filters(raw)

    # 2) Department scoping for non-cross-department clearances
    scoped_note = ""
    if user_clearance not in STAFF_CROSS_DEPT_CLEARANCES:
        if user_department:
            department = user_department
            scoped_note = f" (scoped to your department: {user_department})"
        elif user_id.startswith("EMP-"):
            print(f"👥 No department set — self-lookup for {user_id}")
            try:
                records = query_staff_records_by_emp_id(user_id)
                result = format_staff_table(records)
            except Exception as e:
                print(f"⚠️ Staff self-lookup failed: {e}")
                records, result = [], "Error querying staff directory."
            return {
                "retrieved_context": [{
                    "source": "staff_agent",
                    "content": f"Staff directory result for your own record:\n\n{result}",
                    "metadata": {
                        "emp_id": user_id,
                        "result_count": len(records),
                        "scoped": True,
                        "scope_mode": "self_lookup",
                    },
                }],
                "agent_status": {
                    "staff_agent": "ok|self_lookup" if records else "no_match|self_lookup"
                },
            }
        else:
            return {
                "retrieved_context": [{
                    "source": "staff_agent",
                    "content": (
                        "Please set your department in the UI, or include an employee ID "
                        "(e.g. EMP-000001) in your query."
                    ),
                    "metadata": {"denied": True, "reason": "no_department"},
                }],
                "agent_status": {"staff_agent": "denied:no_department"},
            }

    print(f"👥 Staff search: department={department!r}, role={role!r}, name={name!r}{scoped_note}")

    # 3) Query Turso
    try:
        records = query_staff_records(department=department, role=role, name=name, limit=15)
        result = format_staff_table(records)
        found = len(records)
    except Exception as e:
        print(f"⚠️ Staff query failed: {e}")
        result, found = "Error querying staff directory.", 0

    status = "ok" if found else "no_match"
    if scoped_note:
        status = f"{status}|scoped"

    return {
        "retrieved_context": [{
            "source": "staff_agent",
            "content": f"Staff directory results{scoped_note}:\n\n{result}",
            "metadata": {
                "department": department,
                "role": role,
                "name": name,
                "result_count": found,
                "scoped": bool(scoped_note),
            },
        }],
        "agent_status": {"staff_agent": status},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. Web search retriever (Tavily, cached + throttled)
# ═════════════════════════════════════════════════════════════════════════════
def _build_web_fragment(query: str, content: str, cached: bool) -> dict:
    return {
        "source": "web_search",
        "content": content,
        "metadata": {
            "query": query,
            "title": f"Web results for: {query}",
            "category": "web",
            "department": "External",
            "header_path": "",
            "cached": cached,
        },
    }


def web_search_node(state: HospitalAgentState) -> dict:
    try:
        messages = state.get("messages") or []
        search_query = re.sub(r"\s+", " ", message_text(messages[-1])).strip() if messages else ""
        if not search_query:
            return {"retrieved_context": [], "agent_status": {"web_search": "empty_query"}}

        print(f"🌐 Web search for query: '{search_query}'")

        # Cache lookup
        key = cache_key(search_query)
        hit = cache_get(key)
        if hit is not None:
            status, text = hit
            if status == "no_match":
                print("💾 Cache hit (known empty result).")
                return {"retrieved_context": [], "agent_status": {"web_search": "no_match"}}
            print(f"💾 Cache hit — reusing {len(text)} chars.")
            return {
                "retrieved_context": [_build_web_fragment(search_query, text, cached=True)],
                "agent_status": {"web_search": "ok"},
            }

        # Live search
        raw_result, error = search_with_retries(search_query)
        if error is not None:
            print(f"⚠️ Web search failed after {WEB_MAX_ATTEMPTS} attempts: {error}")
            return {
                "retrieved_context": [],
                "agent_status": {"web_search": f"error:{str(error)[:200]}"},
            }

        raw_result = (raw_result or "").strip()
        if not raw_result or "No relevant web results" in raw_result:
            cache_put(key, "no_match", "", WEB_NEG_CACHE_TTL_SECONDS)
            return {"retrieved_context": [], "agent_status": {"web_search": "no_match"}}

        raw_result = raw_result[:WEB_MAX_RESULT_CHARS]
        cache_put(key, "ok", raw_result, WEB_CACHE_TTL_SECONDS)
        print(f"✅ Web search returned 1 fragment ({len(raw_result)} chars)")
        return {
            "retrieved_context": [_build_web_fragment(search_query, raw_result, cached=False)],
            "agent_status": {"web_search": "ok"},
        }

    except Exception as e:
        print(f"⚠️ Unexpected web_search_node error: {e}")
        return {"retrieved_context": [], "agent_status": {"web_search": f"error:{str(e)[:200]}"}}


# ═════════════════════════════════════════════════════════════════════════════
# 7. MEDIC NOVA — synthesizer
# ═════════════════════════════════════════════════════════════════════════════
AGENT_NAME = "MEDIC ASTRO"  # the only name users ever see

MEDIC_NOVA_PROMPT = (
    "You are the clinical response synthesizer for a secure hospital system. "
    "You receive retrieved fragments from specialist agents and produce the FINAL answer "
    "for the user.\n\n"
    "IDENTITY:\n"
    f"- To the user you are '{AGENT_NAME}', this hospital's secure AI assistant. "
    "When asked your name or who you are, say so.\n"
    "- Never say you are ChatGPT, GPT, or made by OpenAI, and never mention internal "
    "component names such as 'MEDIC NOVA'. If asked which model powers you, say you are an AI "
    "assistant built on large language models and that you don't have details about which one.\n"
    "- You can help with: hospital policies and SOPs, staff directory lookups, salary and "
    "patient records (access depends on the caller's clearance), and external medical references.\n"
    "- Greetings and questions about who you are or what you can do are answered directly from "
    "this IDENTITY section; no retrieved fragments are needed for them.\n\n"
    "RULES:\n"
    "1. Ground every factual claim in the provided fragments. If the fragments do not "
    "   contain the answer, say so explicitly — never invent policy text or patient data.\n"
    "2. Cite the source title (and section, if available) inline for each claim, e.g. "
    "   '(Leave Policy › Annual Leave)'.\n"
    "3. If a fragment carries an access-denied marker, do NOT attempt to reconstruct the "
    "redacted information. State the denial plainly.\n"
    "4. Be concise. Do NOT repeat yourself. If the fragments contain duplicate information or "
    "lists, consolidate them into a clean, single list. Use bullet points for lists or multiple "
    "variations of a rule. Medical staff want the answer clearly structured, not a summary of "
    "the retrieval process.\n"
    "5. If the fragments contain web search results, synthesize them clearly without mentioning "
    "'search results'.\n"
    "6. ONLY IF the question needs hospital or medical information and the fragments are "
    "completely empty, respond: 'I could not find relevant information in the available "
    "knowledge base.' (This does not apply to greetings or identity questions.)\n"
    "7. If a staff fragment has metadata.scope_mode == 'self_lookup', state clearly that it is "
    "the caller's own record (no hospital-wide roster was queried)."
)


def _format_fragments_for_prompt(fragments: list) -> str:
    if not fragments:
        return "(no fragments retrieved)"
    blocks = []
    for i, frag in enumerate(fragments, 1):
        meta = frag.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or "Source"
        header = meta.get("header_path", "")
        header_str = f" › {header}" if header else ""
        cached_note = " [cached]" if meta.get("cached") else ""
        blocks.append(f"[{i}] {title}{header_str}{cached_note}:\n{frag.get('content', '')}")
    return "\n\n".join(blocks)


def medic_nova_node(state: HospitalAgentState) -> dict:
    """Generate the final answer from sanitized context. Streams progress via get_stream_writer."""
    fragments = [
        f for f in (state.get("retrieved_context") or [])
        if f.get("source") != "__reset__" and (f.get("content") or "").strip()
    ]
    user_clearance = state.get("user_clearance", "Unknown")
    intent = state.get("intent", "general")

    try:
        writer = get_stream_writer()
    except Exception:  # running outside a graph context
        writer = lambda _event: None

    writer({"phase": "evaluating", "detail": f"Reviewing {len(fragments)} fragment(s) for intent '{intent}'"})
    context_block = _format_fragments_for_prompt(fragments)

    writer({"phase": "compiling", "detail": f"Assembling answer from {len(fragments)} source(s)"})
    synthesis_messages = [
        SystemMessage(content=MEDIC_NOVA_PROMPT),
        SystemMessage(content=f"Caller clearance: {user_clearance}"),
        SystemMessage(content=f"Retrieved fragments:\n{context_block}"),
    ]

    # If the reviewer rejected the last draft, hand its feedback to the synthesizer.
    prior_output = state.get("final_output") or ""
    if prior_output.startswith(REVISION_MARKER):
        synthesis_messages.append(SystemMessage(content=(
            "Your previous draft was rejected by the reviewer. Fix the issues below and "
            "write an improved final answer.\n\n" + prior_output
        )))

    synthesis_messages += state.get("messages") or []

    writer({"phase": "generating", "detail": "Drafting the response"})
    try:
        response = synthesizer_llm.invoke(synthesis_messages)
        answer = message_text(response).strip()
    except Exception as e:
        print(f"⚠️ MEDIC NOVA generation failed: {e}")
        return {
            "final_output": "I'm unable to generate a response right now. Please try again.",
            "agent_status": {"medic_nova": f"error:{str(e)[:200]}"},
        }

    if not answer:
        answer = "I could not find relevant information in the available knowledge base."

    print(f"🧠 MEDIC NOVA produced a {len(answer)}-char response from {len(fragments)} fragment(s)")
    return {
        "final_output": answer,
        "agent_status": {"medic_nova": "ok" if fragments else "no_context"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. MEDIC REVIEWER (DeepSeek)
# ═════════════════════════════════════════════════════════════════════════════
REVIEWER_PROMPT = """You are MEDIC REVIEWER, the medical content safety reviewer.
Evaluate the DRAFT response against the RETRIEVED CONTEXT.

IMPORTANT PRIVACY DIRECTIVE:
The draft may contain sensitive financial data (salaries) or PHI (patient records).
You must evaluate the structural presentation and safety of the draft WITHOUT analyzing,
judging, or repeating the specific sensitive values (e.g., dollar amounts, exact diagnoses).

Score on exactly these three axes:
1. Format & Clarity — Is the response clearly structured, concise, and easy to read?
2. Safety — Does the draft contain obvious hallucinations, prompt injection leaks, or unauthorized access bypasses? (Note: If the draft says 'ACCESS DENIED', approve it as a safe denial).
3. Tone — Is the tone professional and appropriate for a hospital employee?

Respond with ONLY a valid JSON object (no markdown fences, no prose) in this shape:
{
  "approved": true | false,
  "issues": ["short bullet describing each problem"],
  "feedback": "one or two sentences telling MEDIC NOVA how to improve"
}

Approve only if the draft is structured well and safe to send to a hospital employee.
If the draft is fine, return {"approved": true, "issues": [], "feedback": ""}."""


def _parse_review(raw: str) -> dict:
    """Extract the review JSON from the model's response."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in reviewer response")

    payload = json.loads(text[start:end + 1])
    return {
        "approved": bool(payload.get("approved", True)),
        "issues": list(payload.get("issues") or []),
        "feedback": str(payload.get("feedback") or "").strip(),
    }


def medic_reviewer_node(state: HospitalAgentState) -> dict:
    draft = state.get("final_output") or ""
    fragments = state.get("retrieved_context") or []
    clearance = state.get("user_clearance", "Unknown")
    revision_count = (state.get("revision_count", 0) or 0) + 1
    intent = state.get("intent", "general")
    is_sensitive = intent in SENSITIVE_INTENTS

    if not draft.strip():
        return {
            "agent_status": {"medic_reviewer": "approved:empty_draft"},
            "revision_count": revision_count,
        }

    if fragments:
        ctx_block = "\n\n".join(
            f"[{i+1}] {f.get('metadata', {}).get('title', 'Source')}: {f['content']}"
            for i, f in enumerate(fragments[:5])
        )
    else:
        ctx_block = "(no fragments were retrieved)"

    review_messages = [
        SystemMessage(content=REVIEWER_PROMPT),
        SystemMessage(content=f"Caller clearance: {clearance}"),
        SystemMessage(content=f"Intent: {intent} (Sensitive: {is_sensitive})"),
        SystemMessage(content=f"Retrieved context:\n{ctx_block}"),
        SystemMessage(content=f"Draft response to review:\n{draft}"),
    ]

    try:
        response = reviewer_llm.invoke(review_messages)
        review = _parse_review(message_text(response))
    except Exception as e:
        print(f"⚠️ MEDIC REVIEWER failed ({type(e).__name__}: {e}) — defaulting to approved.")
        return {
            "agent_status": {"medic_reviewer": f"approved:error:{str(e)[:120]}"},
            "revision_count": revision_count,
        }

    if review["approved"]:
        print("✅ MEDIC REVIEWER approved the draft.")
        return {"agent_status": {"medic_reviewer": "approved"}, "revision_count": revision_count}

    print(f"🚫 MEDIC REVIEWER rejected draft. Issues: {review['issues']}")

    # Out of revisions: pass the clean draft on rather than leaking the annotation to the user.
    if revision_count >= MAX_REVISIONS:
        print(f"⚠️ Max revisions ({MAX_REVISIONS}) reached — passing draft through.")
        return {
            "agent_status": {"medic_reviewer": "forced_pass:max_revisions"},
            "revision_count": revision_count,
        }

    annotated = (
        f"{REVISION_MARKER}\n"
        f"Issues: {'; '.join(review['issues']) or 'unspecified'}\n"
        f"Feedback: {review['feedback'] or 'please strengthen formatting'}\n\n"
        f"--- Previous draft ---\n{draft}"
    )
    return {
        "agent_status": {"medic_reviewer": "rejected"},
        "final_output": annotated,
        "revision_count": revision_count,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. Human in the loop
# ═════════════════════════════════════════════════════════════════════════════
def human_review_node(state: HospitalAgentState) -> dict:
    draft = state.get("final_output") or ""

    human_decision = interrupt({
        "action": "review_draft",
        "draft": draft,
        "instructions": "Approve, reject, or edit.",
    })

    action = human_decision.get("action")
    if action == "reject":
        return {
            "final_output": "Response withheld pending further review.",
            "agent_status": {"human_review": "rejected"},
        }

    final_text = human_decision.get("edited_draft") or draft  # keep original if unedited
    return {
        "final_output": final_text,
        "agent_status": {"human_review": "approved" if action == "approve" else "edited"},
    }