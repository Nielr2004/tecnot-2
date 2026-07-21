"""
nl_to_sql.py
------------
Converts a natural-language question into a SQL SELECT statement by calling
the Hugging Face Inference API with an instruct-tuned text model.

Responsibilities:
  - Build the system + user prompt (schema-aware)
  - Call the HF InferenceClient
  - Strip markdown fences / preamble from the model response
  - Retry once with a stricter "return SQL only" prompt if the first response
    cannot be parsed as a bare SQL statement
  - Raise ValueError on hard failure so callers can show a friendly message
"""

from __future__ import annotations

import os
import re
import textwrap

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from db import SCHEMA_TEXT

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL assistant. Your ONLY job is to convert a user's natural-language
question into a single, valid SQLite SELECT statement.

STRICT RULES — you MUST follow all of them:
1. Return ONLY the raw SQL statement — no explanation, no markdown fences (```), no prose.
2. The query MUST start with the word SELECT.
3. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, or MERGE.
4. If the question is ambiguous, write the most reasonable SELECT query you can.
5. Do NOT include any text before or after the SQL statement.
6. Use only the tables and columns listed in the schema below.
7. Always use table aliases when joining multiple tables.

SCHEMA:
{schema}
""").strip()

_USER_TEMPLATE = "Question: {question}"

_STRICT_SYSTEM_PROMPT = textwrap.dedent("""
You are a SQL generator. Output ONLY a single SQLite SELECT statement, nothing else.
No markdown, no explanation, no backticks. Start your response with SELECT.

SCHEMA:
{schema}
""").strip()


# ── SQL extraction ────────────────────────────────────────────────────────────

# Matches a fenced code block (```sql ... ``` or ``` ... ```)
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Matches a bare SELECT ... statement (greedy, up to semicolon or end)
_SELECT_RE = re.compile(r"(SELECT\b.*?)(?:;|\Z)", re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str | None:
    """
    Try to pull a clean SQL SELECT statement out of the model's raw text output.
    Returns None if nothing recognisable is found.
    """
    # 1. Look inside a code fence first
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.upper().startswith("SELECT"):
            return candidate.rstrip(";").strip()

    # 2. Find the first SELECT keyword in the raw text
    select_match = _SELECT_RE.search(text)
    if select_match:
        return select_match.group(1).strip()

    return None


# ── Client factory ────────────────────────────────────────────────────────────

def _get_client() -> InferenceClient:
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set.  Add it to your .env file or Streamlit secrets."
        )
    return InferenceClient(token=HF_TOKEN)


# ── Main public function ──────────────────────────────────────────────────────

def generate_sql(
    natural_language: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Convert *natural_language* to a SQL SELECT statement.

    Returns:
        A cleaned SQL string (no fences, no preamble).

    Raises:
        ValueError  — if the API call fails or no valid SQL can be extracted
                      after two attempts.
    """
    client = _get_client()

    system_msg = _SYSTEM_PROMPT.format(schema=SCHEMA_TEXT)
    user_msg = _USER_TEMPLATE.format(question=natural_language)

    # ── First attempt ──────────────────────────────────────────────────────
    raw1 = _call_model(client, model, system_msg, user_msg)
    sql = _extract_sql(raw1)

    if sql:
        return sql

    # ── Retry with stricter prompt ─────────────────────────────────────────
    strict_system = _STRICT_SYSTEM_PROMPT.format(schema=SCHEMA_TEXT)
    strict_user = (
        f"Convert this question to a SELECT statement: {natural_language}\n"
        "Remember: output the SQL only, nothing else."
    )
    raw2 = _call_model(client, model, strict_system, strict_user)
    sql = _extract_sql(raw2)

    if sql:
        return sql

    raise ValueError(
        "The model did not return a recognisable SQL statement after two attempts.  "
        "Please rephrase your question and try again.\n\n"
        f"Model output (first attempt):\n{raw1[:500]}"
    )


def _call_model(
    client: InferenceClient,
    model: str,
    system_content: str,
    user_content: str,
) -> str:
    """
    Send a chat-completion request and return the assistant's raw text.
    Raises ValueError on API errors.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=512,
            temperature=0.1,   # low temperature → more deterministic SQL
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned an empty response.")
        return content.strip()
    except Exception as exc:
        # Re-raise with a cleaner message; the caller shows it in st.error
        raise ValueError(f"Hugging Face API error: {exc}") from exc
