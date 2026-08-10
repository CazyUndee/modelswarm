#!/usr/bin/env python3
"""
Sync experiment results from the repository to the Cloudflare API.

This script scans for completed experiments and updates the API
with their results. Run by GitHub Actions after training completes.

Usage:
    python scripts/sync_results_to_api.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
import yaml


def find_completed_experiments() -> list[dict]:
    """Find experiments with results that need to be synced."""
    results = []
    competitions_dir = Path("competitions")

    if not competitions_dir.exists():
        return results

    for comp_dir in competitions_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        exp_dir = comp_dir / "experiments"
        if not exp_dir.exists():
            continue

        for exp_file in exp_dir.glob("*.yaml"):
            with open(exp_file) as f:
                config = yaml.safe_load(f)

            # Check if experiment has results that need syncing
            exp_results = config.get("results", {})
            if exp_results.get("status") == "completed" and exp_results.get("oof_metric"):
                results.append({
                    "experiment_id": config.get("experiment_id"),
                    "competition": config.get("competition"),
                    "oof_metric": exp_results.get("oof_metric"),
                    "fold_metrics": exp_results.get("fold_metrics", []),
                    "runtime_seconds": exp_results.get("runtime_seconds"),
                    "decision": exp_results.get("decision"),
                    "reasoning": exp_results.get("reasoning"),
                    "executing_agent_id": config.get("metadata", {}).get("author"),
                })

    return results


def sync_to_api(api_url: str, api_key: str, experiments: list[dict]) -> None:
    """Sync experiment results to the Cloudflare API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for exp in experiments:
        exp_id = exp["experiment_id"]

        # Check current status via API
        try:
            resp = requests.get(f"{api_url}/api/experiments/{exp_id}", headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"⚠️  Could not fetch {exp_id}: {resp.status_code}")
                continue

            current = resp.json()
            if current.get("status") == "completed":
                print(f"⏭️  {exp_id} already completed, skipping")
                continue
        except Exception as e:
            print(f"⚠️  Error checking {exp_id}: {e}")
            continue

        # Update via API
        try:
            payload = {
                "oof_metric": exp["oof_metric"],
                "fold_metrics": exp.get("fold_metrics", []),
                "runtime_seconds": exp.get("runtime_seconds"),
                "decision": exp.get("decision", "inconclusive"),
                "reasoning": exp.get("reasoning", "Synced from GitHub Actions"),
                "executing_agent_id": exp.get("executing_agent_id"),
            }

            resp = requests.post(
                f"{api_url}/api/experiments/{exp_id}/complete",
                json=payload,
                headers=headers,
                timeout=10,
            )

            if resp.status_code == 200:
                print(f"✅ Synced {exp_id} (OOF: {exp['oof_metric']:.5f})")
            else:
                print(f"❌ Failed to sync {exp_id}: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"❌ Error syncing {exp_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync results to API")
    parser.add_argument("--api-url", default=os.environ.get("MODELSWARM_API_URL"))
    parser.add_argument("--api-key", default=os.environ.get("MODELSWARM_API_KEY"))
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("❌ Set MODELSWARM_API_URL and MODELSWARM_API_KEY")
        sys.exit(1)

    print("🔄 Syncing experiment results to API...")

    experiments = find_completed_experiments()
    if not experiments:
        print("No completed experiments to sync.")
        return

    print(f"Found {len(experiments)} completed experiment(s)")
    sync_to_api(args.api_url, args.api_key, experiments)


if __name__ == "__main__":
    main()
