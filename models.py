"""
models.py — Typed Pydantic models for the SQL Debug Environment.
Defines Action, Observation, and State used by step()/reset()/state() API.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base shims — replicate OpenEnv base model contracts without requiring
# openenv-core to be importable at model-definition time.
# ---------------------------------------------------------------------------

class _Action(BaseModel):
    pass


class _Observation(BaseModel):
    done: bool = False
    reward: Optional[float] = None


class _State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0


# ---------------------------------------------------------------------------
# SQLAction
# ---------------------------------------------------------------------------

class SQLAction(_Action):
    """Action the agent sends each step."""

    action_type: Literal["rewrite_query", "fix_syntax", "explain", "submit"] = Field(
        description=(
            "Type of action. Use 'submit' when you believe your query is correct. "
            "'rewrite_query' / 'fix_syntax' for intermediate attempts. "
            "'explain' to get a hint (uses a hint token)."
        )
    )
    sql_query: str = Field(
        description="The SQL query you want to execute / submit."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what you changed and why."
    )


# ---------------------------------------------------------------------------
# SQLObservation
# ---------------------------------------------------------------------------

class SQLObservation(_Observation):
    """What the agent receives after each step / on reset."""

    task_id: str = Field(description="Which of the 3 tasks is currently active.")
    difficulty: str = Field(description="easy | medium | hard")

    schema_context: str = Field(
        description="CREATE TABLE statements for all tables in the scenario."
    )
    buggy_query: str = Field(
        description="The original broken query the agent must fix."
    )
    current_query: str = Field(
        description="The agent's most recent SQL attempt."
    )

    execution_result: str = Field(
        default="",
        description="Rows returned by the query, or 'No rows returned'."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="SQLite error if the query failed to execute."
    )

    hint: Optional[str] = Field(
        default=None,
        description="A hint unlocked after 3 steps or when action_type=='explain'."
    )

    score: float = Field(
        default=0.0,
        description="Current grader score for this episode (0.0–1.0)."
    )
    steps_taken: int = Field(
        default=0,
        description="Number of steps taken so far in this episode."
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps before the episode terminates."
    )


# ---------------------------------------------------------------------------
# SQLState
# ---------------------------------------------------------------------------

class SQLState(_State):
    """Full internal state, returned by state() endpoint."""

    task_id: str = ""
    difficulty: str = ""
    buggy_query: str = ""
    target_result: List[tuple] = Field(default_factory=list)
    best_score: float = 0.0
    hints_used: int = 0
    schema_context: str = ""
    current_query: str = ""
