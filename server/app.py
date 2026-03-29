"""
server/app.py — FastAPI server for the SQL Debug Environment.

Exposes the standard OpenEnv endpoints + required hackathon extras:
  Standard:  POST /reset, POST /step, GET /state, GET /health
  Extra:     GET  /tasks, POST /grader, POST /baseline
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

from models import SQLAction, SQLObservation, SQLState
from server.environment import SQLDebugEnvironment
from server.tasks import TASKS, TASK_IDS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Debug Environment",
    description=(
        "An OpenEnv-compatible environment where AI agents learn to debug "
        "SQL queries of increasing complexity against a live SQLite database."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-instance environment (HF Spaces single-container deployment)
_env = SQLDebugEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str = "rewrite_query"
    sql_query: str
    reasoning: str = ""


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    action_schema: Dict[str, Any]


class GraderResponse(BaseModel):
    task_id: str
    score: float
    current_query: str
    steps_taken: int
    done: bool


class BaselineTaskResult(BaseModel):
    task_id: str
    difficulty: str
    final_score: float
    steps_used: int


class BaselineResponse(BaseModel):
    results: List[BaselineTaskResult]
    aggregate_score: float
    note: str


# ---------------------------------------------------------------------------
# Standard OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Health check — returns 200 when the server is ready."""
    return {"status": "ok", "environment": "sql-debug-env", "version": "1.0.0"}


@app.post("/reset", response_model=SQLObservation)
def reset(req: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Optionally specify task_id (task_1_syntax | task_2_join | task_3_aggregation).
    """
    try:
        obs = _env.reset(
            task_id=req.task_id,
            seed=req.seed,
            episode_id=req.episode_id,
        )
        return obs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=SQLObservation)
def step(req: StepRequest):
    """
    Take one step: send an SQL action, receive observation + reward.
    action_type: rewrite_query | fix_syntax | explain | submit
    """
    try:
        action = SQLAction(
            action_type=req.action_type,
            sql_query=req.sql_query,
            reasoning=req.reasoning,
        )
        obs = _env.step(action)
        return obs
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=SQLState)
def state():
    """Return the full internal episode state."""
    return _env.state


# ---------------------------------------------------------------------------
# Required hackathon extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    """
    Return all available tasks with their difficulty and the action schema
    (fields required for a step action).
    """
    action_schema = {
        "action_type": {
            "type": "string",
            "enum": ["rewrite_query", "fix_syntax", "explain", "submit"],
            "description": "Type of action to perform.",
        },
        "sql_query": {
            "type": "string",
            "description": "SQL query to execute / submit.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the change (optional but helpful).",
        },
    }
    return [
        TaskInfo(
            id=t["id"],
            name=t["name"],
            difficulty=t["difficulty"],
            description=t["description"],
            action_schema=action_schema,
        )
        for t in TASKS.values()
    ]


@app.post("/grader", response_model=GraderResponse)
def grader():
    """
    Return the grader score for the current episode state.
    Call this after one or more steps to get the current score.
    """
    score = _env.grade_current()
    s = _env.state
    return GraderResponse(
        task_id=s.task_id,
        score=round(score, 4),
        current_query=s.current_query,
        steps_taken=s.step_count,
        done=score >= 1.0 or s.step_count >= SQLDebugEnvironment.MAX_STEPS,
    )


@app.post("/baseline", response_model=BaselineResponse)
def baseline_endpoint():
    """
    Run the built-in heuristic baseline agent against all 3 tasks.
    Returns reproducible scores without requiring an OpenAI API key.
    This uses the deterministic rule-based baseline bundled in the environment.
    """
    results = _run_heuristic_baseline()
    aggregate = sum(r.final_score for r in results) / len(results)
    return BaselineResponse(
        results=results,
        aggregate_score=round(aggregate, 4),
        note=(
            "Heuristic baseline using known-correct SQL fixes. "
            "For LLM-based baseline run: python baseline.py"
        ),
    )


# ---------------------------------------------------------------------------
# Internal: heuristic baseline (deterministic, no API key needed)
# ---------------------------------------------------------------------------

# These are the ground-truth correct queries for each task
_KNOWN_FIXES = {
    "task_1_syntax": [
        "SELCT id, name FORM users WHERE role = 'admin';",        # buggy — step 1
        "SELECT id, name, email FROM users WHERE role = 'admin';", # correct — step 2
    ],
    "task_2_join": [
        "SELECT o.order_id, o.amount, o.status FROM orders o, customers c "
        "WHERE o.status = 'shipped';",  # still wrong — step 1
        "SELECT o.order_id, o.amount, o.status FROM orders o "
        "INNER JOIN customers c ON o.customer_id = c.customer_id "
        "WHERE o.status = 'shipped';",  # correct — step 2
    ],
    "task_3_aggregation": [
        "SELECT region, COUNT(*) as num_sales, SUM(amount) as total "
        "FROM sales GROUP BY region HAVING COUNT(*) > 4;",  # wrong — step 1
        "SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total "
        "FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;",  # correct — step 2
    ],
}


def _run_heuristic_baseline() -> List[BaselineTaskResult]:
    results = []
    global _env
    for task_id, queries in _KNOWN_FIXES.items():
        _env.reset(task_id=task_id, seed=42)
        final_score = 0.0
        steps_used = 0
        for i, sql in enumerate(queries):
            is_last = (i == len(queries) - 1)
            obs = _env.step(SQLAction(
                action_type="submit" if is_last else "rewrite_query",
                sql_query=sql,
                reasoning="heuristic baseline",
            ))
            final_score = obs.score
            steps_used = obs.steps_taken
            if obs.done:
                break
        results.append(BaselineTaskResult(
            task_id=task_id,
            difficulty=TASKS[task_id]["difficulty"],
            final_score=round(final_score, 4),
            steps_used=steps_used,
        ))
    # Restore to a clean state
    _env.reset(seed=42)
    return results


# ---------------------------------------------------------------------------
# Entry point (required for multi-mode deployment)
# ---------------------------------------------------------------------------

def main():
    """Start the uvicorn server. Called by [project.scripts] entry point."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
