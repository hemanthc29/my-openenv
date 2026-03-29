"""
baseline.py — OpenAI-powered baseline inference script.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

Runs gpt-4o-mini against all 3 tasks, up to 8 steps each.
Prints per-task scores and an aggregate. Fixed seed for reproducibility.
"""

from __future__ import annotations
import os
import json
import random
import time
from typing import Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 8
RANDOM_SEED = 42

TASK_IDS = ["task_1_syntax", "task_2_join", "task_3_aggregation"]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL developer. Your task is to debug a broken SQL query.

You will receive:
- The database schema (CREATE TABLE statements)
- The original buggy SQL query
- The result of your latest SQL attempt (or an error message)
- Your current score (0.0 = wrong, 1.0 = perfect)

Your goal: produce a corrected SQL query that returns the expected results.

Respond ONLY with a JSON object in this exact format:
{
  "action_type": "rewrite_query",
  "sql_query": "SELECT ...",
  "reasoning": "Brief explanation of what you changed"
}

Use action_type "submit" when you are confident your query is correct.
Do not add markdown, explanation, or any other text — ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def build_user_message(obs: dict) -> str:
    parts = [
        f"## Schema\n```sql\n{obs['schema_context']}\n```",
        f"## Original Buggy Query\n```sql\n{obs['buggy_query']}\n```",
        f"## Your Current Query\n```sql\n{obs['current_query']}\n```",
    ]
    if obs.get("execution_result"):
        parts.append(f"## Execution Result\n```\n{obs['execution_result']}\n```")
    if obs.get("error_message"):
        parts.append(f"## Error\n```\n{obs['error_message']}\n```")
    if obs.get("hint"):
        parts.append(f"## Hint\n{obs['hint']}")
    parts.append(f"## Current Score: {obs['score']:.2f}  |  Steps left: {obs['max_steps'] - obs['steps_taken']}")
    return "\n\n".join(parts)


def call_llm(client: OpenAI, messages: list) -> dict:
    """Call the OpenAI API and parse the action JSON."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    content = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def run_task(client: OpenAI, task_id: str) -> dict:
    """Run one full episode for a given task. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    # Reset
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": RANDOM_SEED})
    resp.raise_for_status()
    obs = resp.json()
    print(f"Difficulty: {obs['difficulty']}")
    print(f"Buggy query: {obs['buggy_query']}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    steps_used = 0

    for step_num in range(1, MAX_STEPS + 1):
        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        try:
            action = call_llm(client, messages)
        except Exception as exc:
            print(f"  [step {step_num}] LLM parse error: {exc}")
            # Fallback: submit current query
            action = {"action_type": "submit", "sql_query": obs["current_query"], "reasoning": "fallback"}

        # Enforce submit on last step
        if step_num == MAX_STEPS:
            action["action_type"] = "submit"

        print(f"  [step {step_num}] {action['action_type']:16s} | score={obs['score']:.2f} | {action.get('reasoning','')[:60]}")

        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        step_resp.raise_for_status()
        obs = step_resp.json()
        steps_used = step_num

        # Append assistant message
        messages.append({"role": "assistant", "content": json.dumps(action)})

        final_score = obs["score"]
        if obs["done"]:
            print(f"  Episode done at step {step_num} | final score = {final_score:.2f}")
            break

        time.sleep(0.3)  # be kind to the API

    return {
        "task_id": task_id,
        "difficulty": obs["difficulty"],
        "final_score": final_score,
        "steps_used": steps_used,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    random.seed(RANDOM_SEED)

    # Health check
    health = requests.get(f"{BASE_URL}/health", timeout=5)
    health.raise_for_status()
    print(f"Environment online: {health.json()}")

    results = []
    for task_id in TASK_IDS:
        result = run_task(client, task_id)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    aggregate = 0.0
    for r in results:
        print(f"  {r['task_id']:25s} ({r['difficulty']:6s})  score={r['final_score']:.4f}  steps={r['steps_used']}")
        aggregate += r["final_score"]
    aggregate /= len(results)
    print(f"\n  Aggregate score: {aggregate:.4f}")
    print(f"  Model: {MODEL}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
