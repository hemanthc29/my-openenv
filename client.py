"""
client.py — Typed OpenEnv client for the SQL Debug Environment.

Usage (after the server is running):
    from client import SQLDebugEnv
    from models import SQLAction

    env = SQLDebugEnv(base_url="http://localhost:7860")
    obs = env.reset()
    obs = env.step(SQLAction(action_type="submit", sql_query="SELECT ...", reasoning="..."))
    state = env.state()
"""

from __future__ import annotations
import requests
from typing import Optional
from models import SQLAction, SQLObservation, SQLState


class SQLDebugEnv:
    """HTTP client for the SQL Debug Environment server."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Standard OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> SQLObservation:
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if episode_id:
            payload["episode_id"] = episode_id
        resp = self._post("/reset", payload)
        return SQLObservation(**resp)

    def step(self, action: SQLAction) -> SQLObservation:
        payload = {
            "action_type": action.action_type,
            "sql_query": action.sql_query,
            "reasoning": action.reasoning,
        }
        resp = self._post("/step", payload)
        return SQLObservation(**resp)

    def state(self) -> SQLState:
        resp = self._get("/state")
        return SQLState(**resp)

    # ------------------------------------------------------------------
    # Extra endpoints
    # ------------------------------------------------------------------

    def tasks(self) -> list:
        return self._get("/tasks")

    def grader(self) -> dict:
        return self._post("/grader", {})

    def baseline(self) -> dict:
        return self._post("/baseline", {})

    def health(self) -> dict:
        return self._get("/health")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        resp = self._session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()
