# SQL Debug Environment 🛠️

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/hemanthc29/my-openenv)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-FFD21E)](https://huggingface.co/spaces/cvhk6300/my-env)

> 🔗 **GitHub Repository:** https://github.com/hemanthc29/my-openenv
> 🤗 **Hugging Face Space:** https://huggingface.co/spaces/cvhk6300/my-env

An **OpenEnv-compatible** reinforcement learning environment where AI agents learn to debug SQL queries of increasing complexity against a live SQLite database.

## Why SQL Debugging?

SQL debugging is a task that developers, data analysts, and DBAs do every day. A typo in a keyword, a missing JOIN, or a wrong GROUP BY can silently produce wrong results. Training agents on this task has immediate real-world value for code-repair models.

---

## Environment Description

The agent receives a **buggy SQL query** and a **database schema**, then interacts with a live in-memory SQLite instance. Each step the agent submits a new SQL query and receives:
- The execution result (or error)
- A grader score (0.0–1.0) for partial progress
- A shaped reward signal
- (After step 3) A hint

### Episode Flow

```
reset(task_id?) → observation
    ↓
step(action) → observation + reward
    ↓  (repeat up to 10 steps)
done=True → final grader score
```

---

## Action & Observation Spaces

### Action (`SQLAction`)
| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"rewrite_query" \| "fix_syntax" \| "explain" \| "submit"` | Use `submit` when confident |
| `sql_query` | `str` | The SQL you want to execute |
| `reasoning` | `str` | Brief explanation (optional) |

### Observation (`SQLObservation`)
| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Which task is active |
| `difficulty` | `str` | easy / medium / hard |
| `schema_context` | `str` | CREATE TABLE statements |
| `buggy_query` | `str` | The original broken query |
| `current_query` | `str` | Agent's latest attempt |
| `execution_result` | `str` | Rows returned, or empty |
| `error_message` | `str?` | SQLite error, if any |
| `hint` | `str?` | Unlocked after 3 steps |
| `score` | `float` | Current grader score (0.0–1.0) |
| `steps_taken` | `int` | Steps used so far |
| `done` | `bool` | Episode over? |
| `reward` | `float?` | Shaped reward for this step |

---

## Tasks

| Task ID | Difficulty | Problem | Max Steps |
|---------|-----------|---------|-----------|
| `task_1_syntax` | 🟢 Easy | Keyword typos: `SELCT`, `FORM` → fix to `SELECT`, `FROM` | 10 |
| `task_2_join` | 🟡 Medium | Missing JOIN condition causes Cartesian product | 10 |
| `task_3_aggregation` | 🔴 Hard | Wrong `GROUP BY` column + wrong `HAVING` threshold (logic bug, no syntax error) | 10 |

### Reward Function

Each step:
```
reward = (new_grader_score - prev_best_score)   # progress signal
       + 0.05 if query executes without error    # syntax bonus  
       - 0.01                                    # step penalty (efficiency)
```

On terminal step (`submit` / max steps):
```
reward = best_score_achieved_in_episode
```

---

## Baseline Scores

Measured with `gpt-4o-mini` (temperature=0.2, seed=42):

| Task | Score |
|------|-------|
| task_1_syntax (easy) | ~0.85 |
| task_2_join (medium) | ~0.65 |
| task_3_aggregation (hard) | ~0.40 |
| **Aggregate** | **~0.63** |

The `/baseline` endpoint uses the built-in heuristic agent (no API key needed).

---

## Setup & Usage

### Local (Python)

```bash
cd sql_debug_env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then interact:
```bash
# Health
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_syntax"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"submit","sql_query":"SELECT id, name, email FROM users WHERE role = '\''admin'\''","reasoning":"fixed typos"}'

# List tasks and action schema
curl http://localhost:7860/tasks

# Get grader score
curl -X POST http://localhost:7860/grader

# Run heuristic baseline
curl -X POST http://localhost:7860/baseline
```

### Docker

```bash
docker build -t sql-debug-env .
docker run -p 7860:7860 sql-debug-env
```

### LLM Baseline Script

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```

### Python Client

```python
from client import SQLDebugEnv
from models import SQLAction

with SQLDebugEnv(base_url="http://localhost:7860") as env:
    obs = env.reset(task_id="task_2_join")
    print(obs.buggy_query)
    
    obs = env.step(SQLAction(
        action_type="submit",
        sql_query="SELECT o.order_id, o.amount, o.status "
                  "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                  "WHERE o.status = 'shipped'",
        reasoning="Added explicit INNER JOIN"
    ))
    print(f"Score: {obs.score}, Done: {obs.done}")
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take an action |
| GET | `/state` | Full internal state |
| GET | `/tasks` | List tasks + action schema |
| POST | `/grader` | Get current grader score |
| POST | `/baseline` | Run heuristic baseline on all 3 tasks |
| GET | `/docs` | Interactive Swagger UI |

---

## Project Structure

```
sql_debug_env/
├── models.py           ← Pydantic types (Action, Observation, State)
├── client.py           ← Typed HTTP client
├── baseline.py         ← OpenAI LLM baseline inference script
├── openenv.yaml        ← Environment manifest
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── server/
    ├── __init__.py
    ├── app.py          ← FastAPI server + extra endpoints
    ├── environment.py  ← Core reset/step/state logic
    └── tasks.py        ← 3 tasks + deterministic graders
```

---

## Tags
`openenv` · `sql` · `debugging` · `code-repair` · `real-world`
