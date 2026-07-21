"""
db.py
-----
Handles all SQLite interaction: read-only connections, query execution,
result formatting, and friendly error conversion.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).parent / "ecommerce.db"

# ── Safety guard ──────────────────────────────────────────────────────────────

_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def is_safe_query(sql: str) -> tuple[bool, str]:
    """
    Returns (True, "") if the query is a safe SELECT statement.
    Returns (False, reason_message) otherwise.
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False, "Query must start with SELECT.  Only read queries are permitted."

    match = _BLOCKED_KEYWORDS.search(stripped)
    if match:
        return False, (
            f"Blocked keyword detected: '{match.group().upper()}'.  "
            "Only SELECT queries are permitted."
        )
    return True, ""


# ── Connection helper ─────────────────────────────────────────────────────────

def _get_ro_connection() -> sqlite3.Connection:
    """Open a read-only URI connection as defence-in-depth."""
    uri = f"file:{DB_PATH}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ── Query execution ───────────────────────────────────────────────────────────

def run_query(sql: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute *sql* (must already be safety-checked) against the SQLite DB.

    Returns:
        (DataFrame, None)  on success
        (None, error_msg)  on failure
    """
    try:
        con = _get_ro_connection()
        try:
            df = pd.read_sql_query(sql, con)
            return df, None
        except pd.errors.DatabaseError as exc:
            return None, _friendly_error(str(exc))
        finally:
            con.close()
    except sqlite3.OperationalError as exc:
        # e.g. DB file not found
        return None, _friendly_error(str(exc))
    except sqlite3.Error as exc:
        return None, _friendly_error(str(exc))


def _friendly_error(raw: str) -> str:
    """Convert raw SQLite error strings into plain-English messages."""
    msg = raw.lower()

    if "no such table" in msg:
        table = _extract_quoted(raw, "no such table: ")
        return f"Table not found: '{table}'.  Please check the table name."

    if "no such column" in msg:
        col = _extract_quoted(raw, "no such column: ")
        return f"Column not found: '{col}'.  Please check the column name."

    if "syntax error" in msg:
        return "The generated SQL has a syntax error.  Try rephrasing your question."

    if "unable to open database" in msg:
        return (
            "Database file not found.  "
            "Please run `python seed.py` first to create and populate the database."
        )

    if "attempt to write a readonly database" in msg:
        return "Write operations are not permitted.  Only SELECT queries are allowed."

    return f"Database error: {raw}"


def _extract_quoted(text: str, prefix: str) -> str:
    idx = text.lower().find(prefix.lower())
    if idx == -1:
        return text
    return text[idx + len(prefix):].split("\n")[0].strip()


# ── Schema description (fed to the LLM prompt) ────────────────────────────────

SCHEMA_TEXT = """
DATABASE SCHEMA (SQLite — ecommerce.db)
========================================

TABLE: customers
  - id          INTEGER  PRIMARY KEY AUTOINCREMENT
  - name        TEXT     customer full name
  - email       TEXT     unique email address
  - city        TEXT     city of residence
  - signup_date TEXT     ISO-8601 date (YYYY-MM-DD) when the customer registered

TABLE: products
  - id             INTEGER  PRIMARY KEY AUTOINCREMENT
  - name           TEXT     product display name
  - category       TEXT     one of: Electronics, Clothing, Home & Garden, Sports,
                            Books, Toys, Beauty, Automotive, Food & Grocery, Office Supplies
  - price          REAL     listed price in USD
  - stock_quantity INTEGER  units currently in stock

TABLE: orders
  - id          INTEGER  PRIMARY KEY AUTOINCREMENT
  - customer_id INTEGER  FOREIGN KEY → customers(id)
  - order_date  TEXT     ISO-8601 datetime (YYYY-MM-DD HH:MM:SS)
  - status      TEXT     one of: pending, shipped, delivered, cancelled

TABLE: order_items
  - id         INTEGER  PRIMARY KEY AUTOINCREMENT
  - order_id   INTEGER  FOREIGN KEY → orders(id)
  - product_id INTEGER  FOREIGN KEY → products(id)
  - quantity   INTEGER  number of units purchased
  - unit_price REAL     price per unit at time of purchase (may differ from products.price)

RELATIONSHIPS:
  orders.customer_id   → customers.id   (many orders per customer)
  order_items.order_id → orders.id      (many items per order)
  order_items.product_id → products.id  (many items can reference the same product)
""".strip()
