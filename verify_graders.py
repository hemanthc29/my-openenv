"""
verify_graders.py — Confirms graders produce varied scores (not always the same).
Tests wrong, partial, and correct SQL for each task.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from server.tasks import TASKS, _run_query
import sqlite3

def make_conn(task):
    conn = sqlite3.connect(":memory:")
    conn.executescript(task["schema"] + task["seed"])
    conn.commit()
    return conn

results = []
PASS = FAIL = 0
def ok(msg): global PASS; PASS+=1; print(f"  PASS  {msg}")
def fail(msg): global FAIL; FAIL+=1; print(f"  FAIL  {msg}")

print("\n=== Grader Score Diversity Verification ===\n")

# --- Task 1 ---
task = TASKS["task_1_syntax"]
conn = make_conn(task)
g = task["grader"]

s_wrong   = g("SELCT id FORM users;", conn)          # syntax error → 0.0
s_partial = g("SELECT id, name, email FROM users;", conn)   # runs, no WHERE → partial
s_correct = g("SELECT id, name, email FROM users WHERE role = 'admin';", conn)

print("task_1_syntax:")
print(f"  wrong SQL    -> score={s_wrong}   (expect 0.0)")
print(f"  partial SQL  -> score={s_partial} (expect 0.3–0.6)")
print(f"  correct SQL  -> score={s_correct} (expect 1.0)")

if s_wrong == 0.0: ok("wrong scores 0.0")
else: fail(f"wrong should be 0, got {s_wrong}")
if 0.0 < s_partial < 1.0: ok(f"partial scores between 0 and 1: {s_partial}")
else: fail(f"partial should be in (0,1), got {s_partial}")
if s_correct == 1.0: ok("correct scores 1.0")
else: fail(f"correct should be 1.0, got {s_correct}")
assert len({s_wrong, s_partial, s_correct}) == 3, "All 3 scores must be DIFFERENT"
ok("all 3 scores are distinct")

# --- Task 2 ---
task = TASKS["task_2_join"]
conn = make_conn(task)
g = task["grader"]

s_wrong   = g("SELCT * FORM orders;", conn)
s_partial = g("SELECT o.order_id, o.amount, o.status FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'shipped' LIMIT 2;", conn)
s_correct = g("SELECT o.order_id, o.amount, o.status FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'shipped';", conn)

print("\ntask_2_join:")
print(f"  wrong SQL    -> score={s_wrong}   (expect 0.0)")
print(f"  partial SQL  -> score={s_partial} (expect 0.3–0.9)")
print(f"  correct SQL  -> score={s_correct} (expect 1.0)")

if s_wrong == 0.0: ok("wrong scores 0.0")
else: fail(f"wrong should be 0, got {s_wrong}")
if 0.0 < s_partial < 1.0: ok(f"partial scores between 0 and 1: {s_partial}")
else: fail(f"partial should be in (0,1), got {s_partial}")
if s_correct == 1.0: ok("correct scores 1.0")
else: fail(f"correct should be 1.0, got {s_correct}")

# --- Task 3 ---
task = TASKS["task_3_aggregation"]
conn = make_conn(task)
g = task["grader"]

s_wrong   = g("SELCT * FORM sales;", conn)
s_partial = g("SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total FROM sales GROUP BY salesperson;", conn)
s_correct = g("SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;", conn)

print("\ntask_3_aggregation:")
print(f"  wrong SQL    -> score={s_wrong}   (expect 0.0)")
print(f"  partial SQL  -> score={s_partial} (expect 0.1–0.9)")
print(f"  correct SQL  -> score={s_correct} (expect 1.0)")

if s_wrong == 0.0: ok("wrong scores 0.0")
else: fail(f"wrong should be 0, got {s_wrong}")
if 0.0 < s_partial < 1.0: ok(f"partial scores between 0 and 1: {s_partial}")
else: fail(f"partial should be in (0,1), got {s_partial}")
if s_correct == 1.0: ok("correct scores 1.0")
else: fail(f"correct should be 1.0, got {s_correct}")

# --- Episode reward shaping test ---
print("\n--- Reward shaping test ---")
from server.environment import SQLDebugEnvironment
from models import SQLAction

env = SQLDebugEnvironment()
obs = env.reset(task_id="task_1_syntax", seed=42)
assert obs.done == False
assert obs.score == 0.0

# Step 1: error
obs2 = env.step(SQLAction(action_type="rewrite_query", sql_query="SELCT * FORM users;", reasoning="still wrong"))
assert obs2.score == 0.0
assert obs2.error_message is not None
ok(f"step 1 with error: reward={obs2.reward}, score={obs2.score}")

# Step 2: partial fix
obs3 = env.step(SQLAction(action_type="rewrite_query", sql_query="SELECT * FROM users;", reasoning="fixed typos, no WHERE yet"))
assert 0.0 < obs3.score < 1.0
ok(f"step 2 partial fix: reward={obs3.reward:.4f}, score={obs3.score}")

# Step 3: rewrite — hint should unlock (steps >= 3)
obs4 = env.step(SQLAction(action_type="rewrite_query", sql_query="SELECT id, name FROM users WHERE role = 'admin';", reasoning="added WHERE"))
assert obs4.hint is not None, f"hint must unlock at step >= 3, got None"
ok(f"step 3 hint unlocked: '{obs4.hint[:50]}...'")

# Step 4: submit correct answer
obs5 = env.step(SQLAction(action_type="submit", sql_query="SELECT id, name, email FROM users WHERE role = 'admin';", reasoning="correct"))
assert obs5.score == 1.0 and obs5.done == True
ok(f"step 4 submit correct: reward={obs5.reward}, score={obs5.score}, done={obs5.done}")

print(f"\n{'='*46}")
print(f"  TOTAL: {PASS} passed, {FAIL} failed")
print(f"{'='*46}\n")
sys.exit(0 if FAIL == 0 else 1)
