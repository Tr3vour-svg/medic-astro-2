"""Central configuration: environment, models, embeddings, vector store, shared constants."""
import os

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone.rerank import PineconeRerank

load_dotenv()  # must run before any os.environ[...] access

# Optional LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "default")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Models & index ───────────────────────────────────────────────────────────
CONFIG = {
    "llm_model": "openai/gpt-4o",
    "cheap_llm_model": "openai/gpt-4o-mini",
    "reviewer_model": "deepseek-chat",
    "embedding_model": "openai/text-embedding-3-small",
    "transcription_model": "openai/whisper-1",
    "pinecone_index_name": "hospital-policies",
    "pinecone_dimension": 1536,
}

# ── Shared constants ─────────────────────────────────────────────────────────
VALID_CLEARANCES = {"Executive", "Department Head", "Physician", "Intern"}
SENSITIVE_INTENTS = {"hr_salary", "patient"}   # always go through human review
MAX_REVISIONS = 2                              # reviewer <-> synthesizer loop limit

# ── Embeddings, vector store, reranker ───────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model=CONFIG["embedding_model"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
    tiktoken_model_name="text-embedding-3-small",
)

vector_store = PineconeVectorStore(
    index_name=CONFIG["pinecone_index_name"],
    embedding=embeddings,
)

reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=3)

# ── LLMs ─────────────────────────────────────────────────────────────────────
supervisor_llm = ChatOpenAI(
    model=CONFIG["cheap_llm_model"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
    temperature=0,
)

# Small structured-extraction calls (emp_id, staff filters) reuse the cheap router model.
extractor_llm = supervisor_llm

synthesizer_llm = ChatOpenAI(
    model=CONFIG["llm_model"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
    temperature=0.2,
    streaming=True,
)

reviewer_llm = ChatOpenAI(
    model=CONFIG["reviewer_model"],
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
    temperature=0.1,
)

whisper_client = openai.OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
)


# ── Utility ──────────────────────────────────────────────────────────────────
def message_text(message) -> str:
    """Extract text from a message object/dict; tolerates list-of-blocks content."""
    content = (
        message.get("content") if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content if isinstance(content, str) else str(content or "")