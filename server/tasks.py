"""
server/tasks.py — Task definitions and graders for the SQL Debug Environment.

Three tasks of increasing difficulty, each with:
  - Schema DDL  (CREATE TABLE statements)
  - Seed data   (INSERT statements)
  - Buggy query (what the agent receives)
  - Expected result (what a correct query should return)
  - Hint text
  - Grader function: score(sql, conn) -> float in [0.0, 1.0]
"""

from __future__ import annotations
import sqlite3
from typing import List, Tuple, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_query(sql: str, conn: sqlite3.Connection) -> Tuple[bool, str, List[Tuple]]:
    """
    Execute *sql* against *conn*.
    Returns (success, message, rows).
    """
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        if rows:
            result_str = "\n".join(str(r) for r in rows)
        else:
            result_str = "No rows returned."
        return True, result_str, rows
    except Exception as exc:
        return False, str(exc), []


def _rows_match(actual: List[Tuple], expected: List[Tuple]) -> bool:
    """Order-insensitive comparison of result row sets."""
    return sorted(str(r) for r in actual) == sorted(str(r) for r in expected)


def _partial_match_ratio(actual: List[Tuple], expected: List[Tuple]) -> float:
    """
    Fraction of expected rows present in actual (for partial credit).
    """
    if not expected:
        return 0.0
    exp_set = set(str(r) for r in expected)
    act_set = set(str(r) for r in actual)
    matched = len(exp_set & act_set)
    return matched / len(exp_set)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Syntax typo fix
# ---------------------------------------------------------------------------

TASK_1_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL
);
"""

TASK_1_SEED = """
INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 'admin');
INSERT INTO users VALUES (2, 'Bob',   'bob@example.com',   'user');
INSERT INTO users VALUES (3, 'Carol', 'carol@example.com', 'user');
INSERT INTO users VALUES (4, 'Dave',  'dave@example.com',  'admin');
"""

TASK_1_BUGGY = "SELCT id, name, email FORM users WHERE role = 'admin';"

TASK_1_EXPECTED: List[Tuple[Any, ...]] = [
    (1, "Alice", "alice@example.com"),
    (4, "Dave",  "dave@example.com"),
]

TASK_1_HINT = (
    "Check the SQL keywords carefully. 'SELCT' and 'FORM' are misspelled. "
    "The correct keywords are SELECT and FROM."
)


def grade_task_1(sql: str, conn: sqlite3.Connection) -> float:
    """
    Grader for Task 1 (Syntax Fix).
    Score breakdown:
      - 1.0 : query runs AND returns exactly the expected admin rows
      - 0.6 : query runs without error, returns ALL 4 users (no WHERE) or partial
      - 0.3 : query runs without error but wrong columns/result
      - 0.0 : query fails with a syntax/runtime error
    """
    success, _, rows = _run_query(sql, conn)
    if not success:
        return 0.0
    if _rows_match(rows, TASK_1_EXPECTED):
        return 1.0
    # Ran fine but wrong result — give partial credit
    ratio = _partial_match_ratio(rows, TASK_1_EXPECTED)
    if ratio > 0:
        return 0.3 + 0.3 * ratio   # 0.3 … 0.6
    return 0.3   # ran, but completely wrong result


# ---------------------------------------------------------------------------
# Task 2 — Medium: Missing JOIN clause
# ---------------------------------------------------------------------------

TASK_2_SCHEMA = """
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL
);
"""

TASK_2_SEED = """
INSERT INTO customers VALUES (1, 'Alice', 'US');
INSERT INTO customers VALUES (2, 'Bob',   'UK');
INSERT INTO customers VALUES (3, 'Carol', 'US');
INSERT INTO orders VALUES (101, 1, 250.00, 'shipped');
INSERT INTO orders VALUES (102, 2, 100.00, 'pending');
INSERT INTO orders VALUES (103, 1, 400.00, 'shipped');
INSERT INTO orders VALUES (104, 3,  75.00, 'shipped');
"""

TASK_2_BUGGY = (
    "SELECT o.order_id, o.amount, o.status "
    "FROM orders o, customers c "
    "WHERE o.status = 'shipped';"
)

# Expected: shipped orders with correct customer name — requires JOIN
TASK_2_EXPECTED: List[Tuple[Any, ...]] = [
    (101, 250.00, "shipped"),
    (103, 400.00, "shipped"),
    (104,  75.00, "shipped"),
]

TASK_2_HINT = (
    "The buggy query does a Cartesian product between orders and customers. "
    "You need to add a JOIN condition: WHERE o.customer_id = c.customer_id (or use INNER JOIN). "
    "The expected result is 3 shipped orders."
)

CORRECT_JOIN_SNIPPET = "customer_id"   # must appear in a JOIN / WHERE condition


def grade_task_2(sql: str, conn: sqlite3.Connection) -> float:
    """
    Grader for Task 2 (Missing JOIN).
    Score breakdown:
      - 1.0 : correct 3 rows (no duplicates from Cartesian product)
      - 0.6 : query runs, has JOIN condition present, row count plausible (3–5)
      - 0.3 : query runs without error but still has >3 rows (Cartesian)
      - 0.0 : syntax error
    """
    success, _, rows = _run_query(sql, conn)
    if not success:
        return 0.0
    if _rows_match(rows, TASK_2_EXPECTED):
        return 1.0
    sql_lower = sql.lower()
    has_join_condition = (
        "join" in sql_lower or
        ("o.customer_id" in sql_lower and "c.customer_id" in sql_lower)
    )
    if has_join_condition and 0 < len(rows) <= 5:
        return 0.6
    if len(rows) > 0:
        return 0.3
    return 0.0


# ---------------------------------------------------------------------------
# Task 3 — Hard: Wrong GROUP BY + HAVING (logic bug, no syntax error)
# ---------------------------------------------------------------------------

TASK_3_SCHEMA = """
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    salesperson TEXT NOT NULL,
    region TEXT NOT NULL,
    amount REAL NOT NULL,
    month INTEGER NOT NULL
);
"""

TASK_3_SEED = """
INSERT INTO sales VALUES (1,  'Alice', 'North', 3000.0, 1);
INSERT INTO sales VALUES (2,  'Alice', 'North', 4500.0, 2);
INSERT INTO sales VALUES (3,  'Alice', 'South', 2000.0, 3);
INSERT INTO sales VALUES (4,  'Bob',   'North', 1500.0, 1);
INSERT INTO sales VALUES (5,  'Bob',   'South', 5000.0, 2);
INSERT INTO sales VALUES (6,  'Bob',   'South', 2200.0, 3);
INSERT INTO sales VALUES (7,  'Carol', 'East',  8000.0, 1);
INSERT INTO sales VALUES (8,  'Carol', 'East',  1000.0, 2);
INSERT INTO sales VALUES (9,  'Carol', 'West',  3500.0, 3);
INSERT INTO sales VALUES (10, 'Dave',  'West',  900.0,  1);
INSERT INTO sales VALUES (11, 'Dave',  'West',  1100.0, 2);
"""

# Bug: groups by region instead of salesperson, HAVING uses wrong threshold
TASK_3_BUGGY = (
    "SELECT region, COUNT(*) as num_sales, SUM(amount) as total "
    "FROM sales "
    "GROUP BY region "
    "HAVING COUNT(*) > 4;"
)

# Correct: top salespersons by total with at least 3 sales
# Expected result: salesperson, num_sales, total — only those with >= 3 sales
TASK_3_EXPECTED: List[Tuple[Any, ...]] = [
    ("Alice", 3, 9500.0),
    ("Bob",   3, 8700.0),
    ("Carol", 3, 12500.0),
]

TASK_3_HINT = (
    "The query groups by 'region' but the goal is to group by 'salesperson'. "
    "The HAVING clause should filter salespersons with at least 3 sales "
    "(COUNT(*) >= 3). Rewrite as: "
    "SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total "
    "FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;"
)

CORRECT_TASK_3_SQL = (
    "SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total "
    "FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;"
)


def grade_task_3(sql: str, conn: sqlite3.Connection) -> float:
    """
    Grader for Task 3 (Aggregation Logic Bug).
    Score breakdown:
      - 1.0 : exact match of 3 rows (Alice/Bob/Carol with correct totals)
      - 0.7 : groups by salesperson correctly, result is close (right columns)
      - 0.4 : runs without error, has some aggregation, partially correct
      - 0.1 : runs without error but completely wrong result
      - 0.0 : syntax/runtime error
    """
    success, _, rows = _run_query(sql, conn)
    if not success:
        return 0.0
    if _rows_match(rows, TASK_3_EXPECTED):
        return 1.0
    sql_lower = sql.lower()
    groups_by_salesperson = "group by salesperson" in sql_lower
    has_sum = "sum(" in sql_lower
    has_having = "having" in sql_lower
    if groups_by_salesperson and has_sum and has_having:
        ratio = _partial_match_ratio(rows, TASK_3_EXPECTED)
        return 0.5 + 0.2 * ratio   # 0.5 … 0.7
    if groups_by_salesperson and (has_sum or has_having):
        return 0.4
    if len(rows) > 0 and ("sum(" in sql_lower or "count(" in sql_lower):
        return 0.1
    return 0.0


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "task_1_syntax": {
        "id": "task_1_syntax",
        "name": "Syntax Fix",
        "difficulty": "easy",
        "description": "Fix a SQL query with keyword typos (SELCT, FORM) to retrieve admin users.",
        "schema": TASK_1_SCHEMA,
        "seed": TASK_1_SEED,
        "buggy_query": TASK_1_BUGGY,
        "expected": TASK_1_EXPECTED,
        "hint": TASK_1_HINT,
        "grader": grade_task_1,
        "max_steps": 10,
    },
    "task_2_join": {
        "id": "task_2_join",
        "name": "Missing JOIN",
        "difficulty": "medium",
        "description": "Add a proper JOIN condition to avoid a Cartesian product and return only shipped orders.",
        "schema": TASK_2_SCHEMA,
        "seed": TASK_2_SEED,
        "buggy_query": TASK_2_BUGGY,
        "expected": TASK_2_EXPECTED,
        "hint": TASK_2_HINT,
        "grader": grade_task_2,
        "max_steps": 10,
    },
    "task_3_aggregation": {
        "id": "task_3_aggregation",
        "name": "Aggregation Logic Bug",
        "difficulty": "hard",
        "description": "Fix GROUP BY and HAVING to get per-salesperson totals for those with >= 3 sales.",
        "schema": TASK_3_SCHEMA,
        "seed": TASK_3_SEED,
        "buggy_query": TASK_3_BUGGY,
        "expected": TASK_3_EXPECTED,
        "hint": TASK_3_HINT,
        "grader": grade_task_3,
        "max_steps": 10,
    },
}

TASK_IDS = list(TASKS.keys())
