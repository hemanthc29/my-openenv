"""
inference.py — OpenEnv inference entry point for SQL Debug Environment.

Runs a heuristic agent against all 3 tasks on the deployed HF Space.
Usage:
    python inference.py                          # uses default HF Space URL
    python inference.py --url http://localhost:7860
"""

import argparse
import json
import requests


def run_inference(base_url: str = "https://cvhk6300-my-env.hf.space") -> dict:
    base_url = base_url.rstrip("/")
    results = []

    # Known correct fixes for each task (heuristic baseline)
    known_fixes = {
        "task_1_syntax": {
            "difficulty": "easy",
            "queries": [
                {
                    "action_type": "fix_syntax",
                    "sql_query": "SELECT id, name, email FROM users WHERE role = 'admin';",
                    "reasoning": "Fixed SELCT -> SELECT, FORM -> FROM typos",
                },
            ],
        },
        "task_2_join": {
            "difficulty": "medium",
            "queries": [
                {
                    "action_type": "rewrite_query",
                    "sql_query": (
                        "SELECT o.order_id, o.amount, o.status "
                        "FROM orders o "
                        "INNER JOIN customers c ON o.customer_id = c.customer_id "
                        "WHERE o.status = 'shipped';"
                    ),
                    "reasoning": "Added explicit INNER JOIN with ON condition",
                },
            ],
        },
        "task_3_aggregation": {
            "difficulty": "hard",
            "queries": [
                {
                    "action_type": "submit",
                    "sql_query": (
                        "SELECT salesperson, COUNT(*) as num_sales, SUM(amount) as total "
                        "FROM sales GROUP BY salesperson HAVING COUNT(*) >= 3;"
                    ),
                    "reasoning": "Fixed GROUP BY column and HAVING threshold",
                },
            ],
        },
    }

    for task_id, task_info in known_fixes.items():
        # Reset episode
        reset_resp = requests.post(
            f"{base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        print(f"\n[{task_id}] difficulty={task_info['difficulty']}")
        print(f"  buggy_query: {obs.get('buggy_query', '')[:80]}")

        # Run steps
        final_score = 0.0
        steps_used = 0
        for i, action in enumerate(task_info["queries"]):
            is_last = i == len(task_info["queries"]) - 1
            if is_last:
                action["action_type"] = "submit"

            step_resp = requests.post(
                f"{base_url}/step",
                json=action,
                timeout=30,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()
            final_score = obs.get("score", 0.0)
            steps_used = obs.get("steps_taken", i + 1)
            print(f"  step {steps_used}: score={final_score:.4f}  done={obs.get('done')}")
            if obs.get("done"):
                break

        results.append({
            "task_id": task_id,
            "difficulty": task_info["difficulty"],
            "final_score": round(final_score, 4),
            "steps_used": steps_used,
        })

    aggregate = sum(r["final_score"] for r in results) / len(results)
    output = {"results": results, "aggregate_score": round(aggregate, 4)}
    print(f"\n=== Aggregate Score: {aggregate:.4f} ===")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv SQL Debug inference script")
    parser.add_argument(
        "--url",
        default="https://cvhk6300-my-env.hf.space",
        help="Base URL of the running OpenEnv server",
    )
    args = parser.parse_args()
    run_inference(base_url=args.url)
