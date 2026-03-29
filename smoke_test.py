"""smoke_test.py - Quick verification of all endpoints."""
import requests, json, sys

BASE = "http://127.0.0.1:7861"
PASS = 0
FAIL = 0

def ok(msg): global PASS; PASS += 1; print(f"  PASS  {msg}")
def fail(msg, err=""): global FAIL; FAIL += 1; print(f"  FAIL  {msg}: {err}")

print("\n=== SQL Debug Env Smoke Test ===\n")

# 1. Health
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    ok("GET /health -> 200 ok")
except Exception as e:
    fail("GET /health", e); sys.exit(1)

# 2. /tasks
try:
    r = requests.get(f"{BASE}/tasks", timeout=5)
    tasks = r.json()
    assert len(tasks) == 3, f"expected 3 tasks, got {len(tasks)}"
    ids = [t["id"] for t in tasks]
    assert "task_1_syntax" in ids
    assert "task_2_join" in ids
    assert "task_3_aggregation" in ids
    ok(f"GET /tasks -> 3 tasks: {ids}")
except Exception as e:
    fail("GET /tasks", e)

# 3. reset task_1
try:
    r = requests.post(f"{BASE}/reset", json={"task_id": "task_1_syntax", "seed": 42}, timeout=5)
    obs = r.json()
    assert obs["task_id"] == "task_1_syntax"
    assert obs["difficulty"] == "easy"
    assert "SELCT" in obs["buggy_query"] or "FORM" in obs["buggy_query"]
    assert obs["done"] == False
    ok(f"POST /reset task_1 -> buggy_query starts with: {obs['buggy_query'][:40]}")
except Exception as e:
    fail("POST /reset task_1", e)

# 4. step with wrong SQL
try:
    r = requests.post(f"{BASE}/step", json={
        "action_type": "rewrite_query",
        "sql_query": "SELCT id FORM users;",
        "reasoning": "still wrong"
    }, timeout=5)
    obs = r.json()
    assert obs["score"] < 0.5, f"wrong query should have low score, got {obs['score']}"
    assert obs["error_message"] is not None
    ok(f"POST /step wrong SQL -> score={obs['score']}, error='{obs['error_message'][:50]}'")
except Exception as e:
    fail("POST /step wrong SQL", e)

# 5. step with correct SQL
try:
    r = requests.post(f"{BASE}/step", json={
        "action_type": "submit",
        "sql_query": "SELECT id, name, email FROM users WHERE role = 'admin';",
        "reasoning": "fixed both typos"
    }, timeout=5)
    obs = r.json()
    assert obs["score"] == 1.0, f"correct query should score 1.0, got {obs['score']}"
    assert obs["done"] == True
    ok(f"POST /step correct SQL -> score={obs['score']}, done={obs['done']}, reward={obs['reward']}")
except Exception as e:
    fail("POST /step correct SQL", e)

# 6. state
try:
    r = requests.get(f"{BASE}/state", timeout=5)
    state = r.json()
    assert state["task_id"] == "task_1_syntax"
    assert state["best_score"] == 1.0
    ok(f"GET /state -> task_id={state['task_id']}, best_score={state['best_score']}")
except Exception as e:
    fail("GET /state", e)

# 7. grader
try:
    # fresh episode
    requests.post(f"{BASE}/reset", json={"task_id": "task_1_syntax", "seed": 42}, timeout=5)
    requests.post(f"{BASE}/step", json={
        "action_type": "rewrite_query",
        "sql_query": "SELECT id, name, email FROM users WHERE role = 'admin';",
        "reasoning": "test"
    }, timeout=5)
    r = requests.post(f"{BASE}/grader", json={}, timeout=5)
    g = r.json()
    assert 0.0 <= g["score"] <= 1.0, f"grader score out of range: {g['score']}"
    ok(f"POST /grader -> score={g['score']}")
except Exception as e:
    fail("POST /grader", e)

# 8. /baseline
try:
    r = requests.post(f"{BASE}/baseline", json={}, timeout=30)
    b = r.json()
    assert "results" in b and "aggregate_score" in b
    for res in b["results"]:
        assert 0.0 <= res["final_score"] <= 1.0
    ok(f"POST /baseline -> aggregate={b['aggregate_score']:.4f}, tasks={[r['task_id'] for r in b['results']]}")
    for res in b["results"]:
        print(f"          {res['task_id']:25s} ({res['difficulty']:6s}) score={res['final_score']:.4f} steps={res['steps_used']}")
except Exception as e:
    fail("POST /baseline", e)

# 9. All 3 tasks graders produce different scores
print("\n--- Grader diversity check ---")
try:
    scores = {}
    sqls = {
        "task_1_syntax": "SELECT id, name, email FROM users WHERE role = 'admin';",
        "task_2_join": "SELECT o.order_id, o.amount, o.status FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'shipped';",
        "task_3_aggregation": "SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;",
    }
    for tid, sql in sqls.items():
        requests.post(f"{BASE}/reset", json={"task_id": tid, "seed": 1}, timeout=5)
        r = requests.post(f"{BASE}/step", json={"action_type": "submit", "sql_query": sql, "reasoning": "test"}, timeout=5)
        score = r.json()["score"]
        scores[tid] = score
        print(f"  {tid:30s} -> score={score:.4f}")
    assert all(s == 1.0 for s in scores.values()), f"Not all correct queries scored 1.0: {scores}"
    ok("All 3 correct queries score 1.0")
except Exception as e:
    fail("3-task grader check", e)

# Summary
print(f"\n{'='*42}")
print(f"  TOTAL: {PASS} passed, {FAIL} failed")
print(f"{'='*42}\n")
sys.exit(0 if FAIL == 0 else 1)
