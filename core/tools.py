"""External integrations: Turso (SQL data), Pinecone policy search, Tavily web search.

NOTE: the graph nodes call the plain functions here (query_*_record, run_web_search)
and enforce clearance checks themselves. The @tool wrappers at the bottom do NOT check
clearance, so don't bind them to an LLM without adding RBAC.
"""
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict
from typing import Optional

import turso_serverless
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from core.config import vector_store


# ═════════════════════════════════════════════════════════════════════════════
# Turso connections & query helper
# ═════════════════════════════════════════════════════════════════════════════
def turso_main():
    """Staff + salaries database (short-lived connection)."""
    return turso_serverless.connect(
        os.environ["TURSO_DATABASE_URL"],
        auth_token=os.environ["TURSO_AUTH_TOKEN"],
    )


def turso_patients():
    """Patients database (short-lived connection)."""
    return turso_serverless.connect(
        os.environ["TURSO_PATIENTS_URL"],
        auth_token=os.environ["TURSO_PATIENTS_TOKEN"],
    )


def _run_query(connect, sql: str, params: tuple = ()) -> list:
    """Run a parameterized query and return rows as a list of dicts."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if rows else []
    finally:
        conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ═════════════════════════════════════════════════════════════════════════════
# Salary
# ═════════════════════════════════════════════════════════════════════════════
def query_salary_record(emp_id: str) -> Optional[dict]:
    rows = _run_query(turso_main, "SELECT * FROM salaries WHERE emp_id = ?", (emp_id,))
    if not rows:
        return None
    record = rows[0]
    allowances = record.get("allowances")
    if isinstance(allowances, str):  # stored as a JSON string
        try:
            record["allowances"] = json.loads(allowances)
        except Exception:
            record["allowances"] = {"raw": allowances}
    return record


def format_salary(emp_id: str, record: dict) -> str:
    allowances = record.get("allowances") or {}
    allowance_str = ", ".join(f"{k}: {v}" for k, v in allowances.items())
    base = record.get("base_salary_usd", 0) or 0
    return (
        f"Salary details for {emp_id}:\n"
        f"- Tier: {record.get('tier')}\n"
        f"- Base Salary: ${base:,}\n"
        f"- Bonus Eligible: {record.get('bonus_eligible')}\n"
        f"- Allowances: {allowance_str}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Patients (patients DB + staff DB joined in Python)
# ═════════════════════════════════════════════════════════════════════════════
PHI_FIELDS = ("diagnosis", "treatment_notes")


def query_patient_record(patient_id: str) -> Optional[dict]:
    rows = _run_query(turso_patients, "SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    if not rows:
        return None
    record = rows[0]

    physician_id = record.get("primary_physician_id")
    if physician_id:
        phys = _run_query(
            turso_main,
            "SELECT name, department, role FROM staff WHERE emp_id = ?",
            (physician_id,),
        )
        if phys:
            record["physician_name"] = phys[0].get("name")
            record["physician_dept"] = phys[0].get("department")
            record["physician_role"] = phys[0].get("role")
    return record


def format_patient(patient_id: str, record: dict) -> str:
    """Format a patient record. Free-text clinical fields are always masked."""
    record = dict(record)
    for phi in PHI_FIELDS:
        if record.get(phi) is not None:
            record[phi] = "[REDACTED_PHI]"

    charges = record.get("total_charges_usd", 0) or 0
    return (
        f"Patient details for {patient_id}:\n"
        f"- Name: {record.get('full_name')}\n"
        f"- DOB: {record.get('dob')}\n"
        f"- Admission Date: {record.get('admission_date')}\n"
        f"- Room: {record.get('room_number')}\n"
        f"- Billing Status: {record.get('billing_status')}\n"
        f"- Total Charges: ${charges:,}\n"
        f"- Diagnosis: {record.get('diagnosis')}\n"
        f"- Treatment Notes: {record.get('treatment_notes')}\n"
        f"- Assigned Physician: {record.get('physician_name') or 'Unknown'} "
        f"(ID: {record.get('primary_physician_id') or 'None'}, "
        f"Dept: {record.get('physician_dept') or 'Unknown'}, "
        f"Role: {record.get('physician_role') or 'Unknown'})\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Staff directory
# ═════════════════════════════════════════════════════════════════════════════
STAFF_COLUMNS = "emp_id, name, role, department"


def query_staff_records(department=None, role=None, name=None, limit: int = 15) -> list:
    clauses, params = [], []
    if department:
        clauses.append("department LIKE ?")
        params.append(f"%{department}%")
    if role:
        clauses.append("role LIKE ?")
        params.append(f"%{role}%")
    if name:
        clauses.append("name LIKE ?")
        params.append(f"%{name}%")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT {STAFF_COLUMNS} FROM staff{where} LIMIT ?"
    params.append(int(limit))
    return _run_query(turso_main, sql, tuple(params))


def query_staff_records_by_emp_id(emp_id: str) -> list:
    return _run_query(
        turso_main, f"SELECT {STAFF_COLUMNS} FROM staff WHERE emp_id = ?", (emp_id,)
    )


def normalize_emp_id(raw_id: str) -> str:
    """'emp-50' -> 'EMP-000050'. Returns the upper-cased input if no number is found."""
    if not raw_id:
        return ""
    match = re.search(r"EMP-(\d+)", raw_id.upper())
    return f"EMP-{int(match.group(1)):06d}" if match else raw_id.upper()


def format_staff_table(records: list) -> str:
    if not records:
        return "No matching staff records found."
    lines = ["| emp_id | name | role | department |", "|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r.get('emp_id', '')} | {r.get('name', '')} | {r.get('role', '')} | "
            f"{r.get('department', '')} |"
        )
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Web search (Tavily) + cache / throttle / retry helpers
# ═════════════════════════════════════════════════════════════════════════════
_tavily = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    api_key=os.environ["TAVILY_API_KEY"],
)

WEB_CACHE_TTL_SECONDS = 60 * 60
WEB_NEG_CACHE_TTL_SECONDS = 5 * 60
WEB_CACHE_MAX_ENTRIES = 256
WEB_MIN_INTERVAL_SECONDS = 1.0
WEB_MAX_ATTEMPTS = 3
WEB_BACKOFF_BASE_SECONDS = 2.0
WEB_MAX_RESULT_CHARS = 8000

_web_cache: "OrderedDict[str, tuple]" = OrderedDict()
_web_cache_lock = threading.Lock()
_web_rate_lock = threading.Lock()
_web_last_call = {"t": 0.0}


def run_web_search(query: str) -> str:
    """Run a Tavily search and format the results. RAISES on failure (so retries work)."""
    response = _tavily.invoke({"query": query})
    results = response.get("results") if isinstance(response, dict) else None
    if not results:
        return "No relevant web results found."
    formatted = []
    for i, res in enumerate(results, 1):
        formatted.append(
            f"Source {i} [{res.get('title', 'Source')} - {res.get('url', '')}]:\n"
            f"{res.get('content', '')}"
        )
    return "\n\n".join(formatted)


def cache_key(query: str) -> str:
    return query.casefold()


def cache_get(key: str):
    with _web_cache_lock:
        entry = _web_cache.get(key)
        if entry is None:
            return None
        expires_at, status, text = entry
        if time.monotonic() >= expires_at:
            del _web_cache[key]
            return None
        _web_cache.move_to_end(key)
        return status, text


def cache_put(key: str, status: str, text: str, ttl: float) -> None:
    with _web_cache_lock:
        _web_cache[key] = (time.monotonic() + ttl, status, text)
        _web_cache.move_to_end(key)
        while len(_web_cache) > WEB_CACHE_MAX_ENTRIES:
            _web_cache.popitem(last=False)


def _throttle() -> None:
    with _web_rate_lock:
        now = time.monotonic()
        wait = _web_last_call["t"] + WEB_MIN_INTERVAL_SECONDS - now
        _web_last_call["t"] = max(now, _web_last_call["t"] + WEB_MIN_INTERVAL_SECONDS)
    if wait > 0:
        time.sleep(wait)


def search_with_retries(query: str):
    """Returns (result_text, None) on success or (None, last_error) after all attempts fail."""
    last_error = None
    for attempt in range(1, WEB_MAX_ATTEMPTS + 1):
        _throttle()
        try:
            return run_web_search(query), None
        except Exception as e:
            last_error = e
            if attempt == WEB_MAX_ATTEMPTS:
                break
            delay = WEB_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"⚠️ Web search attempt {attempt}/{WEB_MAX_ATTEMPTS} failed "
                  f"({str(e)[:120]}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
    return None, last_error


# ═════════════════════════════════════════════════════════════════════════════
# LangChain @tool wrappers (not used by the current graph — see module docstring)
# ═════════════════════════════════════════════════════════════════════════════
@tool
def search_hospital_policies(query: str, department: str = None) -> str:
    """Search hospital operational policies, HR guidelines, and clinical handbooks
    using semantic vector search. Optionally filter by department (e.g. 'Nursing', 'HR')."""
    filter_expr = {"department": {"$in": ["All", department]}} if department else None
    docs = vector_store.similarity_search(query, k=3, filter=filter_expr)
    if not docs:
        return "No relevant hospital policies found."

    results = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Policy")
        dept = doc.metadata.get("department", "All")
        header = doc.metadata.get("header_path", "")
        header_str = f" | Section: {header}" if header else ""
        results.append(f"Source {i+1} [{title} - Dept: {dept}{header_str}]:\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def query_staff_directory(department: str = None, role: str = None) -> str:
    """Query the staff directory for employee details and department mappings."""
    return format_staff_table(query_staff_records(department=department, role=role, limit=10))


@tool
def query_salary_structure(emp_id: str) -> str:
    """Query employee compensation for a given emp_id. Caller must verify clearance first."""
    record = query_salary_record(emp_id)
    if record is None:
        return f"No salary record found for Employee ID {emp_id}."
    return format_salary(emp_id, record)


@tool
def query_patient_records(patient_id: str) -> str:
    """Query patient admission, room, and billing status. Clinical notes are masked."""
    record = query_patient_record(patient_id)
    if record is None:
        return f"No patient found with ID {patient_id}."
    return format_patient(patient_id, record)


@tool
def external_web_search(query: str) -> str:
    """Search the open web for external medical literature, clinical definitions,
    and international referral hospital directories."""
    try:
        return run_web_search(query)
    except Exception as e:
        return f"Web search failed: {e}"


ALL_TOOLS = [
    search_hospital_policies,
    query_staff_directory,
    query_salary_structure,
    query_patient_records,
    external_web_search,
]