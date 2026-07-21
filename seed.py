"""
seed.py
-------
Run ONCE before starting the app:

    python seed.py

Creates `ecommerce.db` (SQLite) and populates it with realistic fake data using
the Faker library.  Safe to re-run — it drops and recreates all tables.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

DB_PATH = Path(__file__).parent / "ecommerce.db"

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Schema ────────────────────────────────────────────────────────────────────

DDL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE,
    city        TEXT    NOT NULL,
    signup_date TEXT    NOT NULL   -- ISO-8601 date string
);

CREATE TABLE products (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    category       TEXT    NOT NULL,
    price          REAL    NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT    NOT NULL,   -- ISO-8601 datetime string
    status      TEXT    NOT NULL    -- 'pending' | 'shipped' | 'delivered' | 'cancelled'
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL DEFAULT 1,
    unit_price REAL    NOT NULL
);
"""

# ── Seed data ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
    "Toys", "Beauty", "Automotive", "Food & Grocery", "Office Supplies",
]

STATUSES = ["pending", "shipped", "delivered", "cancelled"]


def random_date(start_days_ago: int = 730, end_days_ago: int = 0) -> str:
    """Return a random ISO-8601 date string in the past."""
    delta = random.randint(end_days_ago, start_days_ago)
    dt = datetime.now() - timedelta(days=delta)
    return dt.strftime("%Y-%m-%d")


def random_datetime(start_days_ago: int = 365, end_days_ago: int = 0) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    dt = datetime.now() - timedelta(days=delta, hours=random.randint(0, 23),
                                    minutes=random.randint(0, 59))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def seed_customers(cur: sqlite3.Cursor, n: int = 60) -> list[int]:
    rows = []
    emails_seen: set[str] = set()
    while len(rows) < n:
        email = fake.unique.email()
        if email in emails_seen:
            continue
        emails_seen.add(email)
        rows.append((
            fake.name(),
            email,
            fake.city(),
            random_date(730, 30),
        ))
    cur.executemany(
        "INSERT INTO customers (name, email, city, signup_date) VALUES (?,?,?,?)",
        rows,
    )
    cur.execute("SELECT id FROM customers")
    return [r[0] for r in cur.fetchall()]


def seed_products(cur: sqlite3.Cursor, n: int = 60) -> list[int]:
    rows = []
    for _ in range(n):
        rows.append((
            fake.catch_phrase()[:60],          # product name
            random.choice(CATEGORIES),
            round(random.uniform(4.99, 499.99), 2),
            random.randint(0, 500),
        ))
    cur.executemany(
        "INSERT INTO products (name, category, price, stock_quantity) VALUES (?,?,?,?)",
        rows,
    )
    cur.execute("SELECT id FROM products")
    return [r[0] for r in cur.fetchall()]


def seed_orders(
    cur: sqlite3.Cursor,
    customer_ids: list[int],
    n: int = 80,
) -> list[int]:
    rows = [
        (
            random.choice(customer_ids),
            random_datetime(365, 0),
            random.choice(STATUSES),
        )
        for _ in range(n)
    ]
    cur.executemany(
        "INSERT INTO orders (customer_id, order_date, status) VALUES (?,?,?)",
        rows,
    )
    cur.execute("SELECT id FROM orders")
    return [r[0] for r in cur.fetchall()]


def seed_order_items(
    cur: sqlite3.Cursor,
    order_ids: list[int],
    product_ids: list[int],
    target: int = 150,
) -> None:
    rows = []
    for _ in range(target):
        order_id = random.choice(order_ids)
        product_id = random.choice(product_ids)
        # Fetch the product price to use as unit_price (with minor variation)
        cur.execute("SELECT price FROM products WHERE id = ?", (product_id,))
        base_price = cur.fetchone()[0]
        unit_price = round(base_price * random.uniform(0.85, 1.05), 2)
        rows.append((order_id, product_id, random.randint(1, 5), unit_price))
    cur.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?,?,?,?)",
        rows,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[*] Seeding database at: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Execute DDL statements one by one (executescript commits, so use execute loop)
    for statement in DDL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            cur.execute(stmt)
    con.commit()

    print("  [OK] Tables created")

    customer_ids = seed_customers(cur)
    con.commit()
    print(f"  [OK] {len(customer_ids)} customers inserted")

    product_ids = seed_products(cur)
    con.commit()
    print(f"  [OK] {len(product_ids)} products inserted")

    order_ids = seed_orders(cur, customer_ids)
    con.commit()
    print(f"  [OK] {len(order_ids)} orders inserted")

    seed_order_items(cur, order_ids, product_ids)
    con.commit()
    print("  [OK] 150+ order_items inserted")

    con.close()
    print("[DONE] Seeding complete!  Run: streamlit run app.py")


if __name__ == "__main__":
    main()
