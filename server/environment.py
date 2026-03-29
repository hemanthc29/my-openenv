"""
server/environment.py — Core SQLDebugEnvironment logic.

Implements reset() / step() / state() following the OpenEnv pattern.
Uses an in-memory SQLite connection per episode for deterministic execution.
"""

from __future__ import annotations
import random
import sqlite3
import uuid
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import SQLAction, SQLObservation, SQLState
from server.tasks import TASKS, TASK_IDS, _run_query


STEP_PENALTY = 0.01       # Per-step efficiency penalty
SYNTAX_BONUS = 0.05       # Bonus when query executes without error


class SQLDebugEnvironment:
    """
    OpenEnv-compatible environment for SQL query debugging.
    
    Each episode:
      1. reset(task_id?) seeds an in-memory SQLite DB and returns initial obs.
      2. step(action) executes agent SQL, grades it, returns shaped reward.
      3. state returns full internal state.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False  # single-instance server; see app.py

    # Maximum steps before forced termination
    MAX_STEPS = 10

    def __init__(self) -> None:
        self._state = SQLState()
        self._conn: Optional[sqlite3.Connection] = None
        self._task: Optional[dict] = None
        self._best_score: float = 0.0
        self._hints_used: int = 0
        self._current_query: str = ""
        self._steps: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SQLObservation:
        """Start a new episode. Optionally specify task_id; otherwise random."""
        if seed is not None:
            random.seed(seed)

        # Pick task
        if task_id and task_id in TASKS:
            self._task = TASKS[task_id]
        else:
            self._task = TASKS[random.choice(TASK_IDS)]

        # Seed SQLite DB
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(self._task["schema"] + self._task["seed"])
        self._conn.commit()

        # Reset episode state
        self._best_score = 0.0
        self._hints_used = 0
        self._current_query = self._task["buggy_query"]
        self._steps = 0
        self._done = False

        eid = episode_id or str(uuid.uuid4())
        self._state = SQLState(
            episode_id=eid,
            step_count=0,
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            buggy_query=self._task["buggy_query"],
            target_result=list(self._task["expected"]),
            best_score=0.0,
            hints_used=0,
            schema_context=self._task["schema"].strip(),
            current_query=self._task["buggy_query"],
        )

        return SQLObservation(
            done=False,
            reward=None,
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            schema_context=self._task["schema"].strip(),
            buggy_query=self._task["buggy_query"],
            current_query=self._current_query,
            execution_result="",
            error_message=None,
            hint=None,
            score=0.0,
            steps_taken=0,
            max_steps=self.MAX_STEPS,
        )

    def step(self, action: SQLAction, **kwargs) -> SQLObservation:
        """Execute agent's SQL action; return shaped reward and new observation."""
        if self._task is None or self._done:
            raise RuntimeError("Call reset() before step().")

        self._steps += 1
        self._state.step_count = self._steps

        sql = action.sql_query.strip()
        self._current_query = sql
        self._state.current_query = sql

        # --- Execute query ---
        success, exec_result, _ = _run_query(sql, self._conn)

        # --- Grade ---
        new_score = self._task["grader"](sql, self._conn)

        # --- Reward shaping ---
        score_delta = max(0.0, new_score - self._best_score)
        syntax_bonus = SYNTAX_BONUS if success else 0.0
        step_penalty = STEP_PENALTY
        reward = score_delta + syntax_bonus - step_penalty

        # --- Hint logic ---
        # Unlock hint at step 3+ or when agent explicitly asks for one
        hint_unlocked = self._steps >= 3 or action.action_type == "explain"
        if hint_unlocked and self._hints_used == 0:
            self._hints_used = 1           # count only the first unlock
            self._state.hints_used = self._hints_used
        hint = self._task["hint"] if hint_unlocked else None

        # --- Update best score ---
        if new_score > self._best_score:
            self._best_score = new_score
            self._state.best_score = self._best_score

        # --- Episode termination ---
        done = (
            new_score >= 1.0
            or action.action_type == "submit"
            or self._steps >= self.MAX_STEPS
        )
        self._done = done

        # On terminal submit, use best score as final reward
        if done:
            reward = self._best_score

        return SQLObservation(
            done=done,
            reward=round(reward, 4),
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            schema_context=self._task["schema"].strip(),
            buggy_query=self._task["buggy_query"],
            current_query=sql,
            execution_result=exec_result if success else "",
            error_message=None if success else exec_result,
            hint=hint,
            score=round(new_score, 4),
            steps_taken=self._steps,
            max_steps=self.MAX_STEPS,
        )

    @property
    def state(self) -> SQLState:
        return self._state

    def grade_current(self) -> float:
        """Grade the current query without advancing the episode."""
        if self._task is None or self._conn is None:
            return 0.0
        return self._task["grader"](self._current_query, self._conn)
