#!/usr/bin/env python3
"""
Health check script for the operations agent.

Run periodically to:
- Detect stalled agents
- Release expired experiment claims
- Update research state
- Check runner availability
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_API_URL = "https://modelswarm.workers.dev"
STALE_THRESHOLD_MINUTES = 15
CLAIM_EXPIRY_MINUTES = 30


def check_agents(api_url: str) -> list[dict]:
    """Check for stalled agents and mark them."""
    resp = requests.get(f"{api_url}/api/agents", timeout=30)
    resp.raise_for_status()
    agents = resp.json().get("agents", [])

    stalled = []
    now = datetime.now(timezone.utc)

    for agent in agents:
        if agent["status"] != "active":
            continue

        last_hb = agent.get("last_heartbeat")
        if not last_hb:
            continue

        last_hb_dt = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
        if (now - last_hb_dt).total_seconds() > STALE_THRESHOLD_MINUTES * 60:
            stalled.append(agent)
            # Mark as stalled
            requests.post(
                f"{api_url}/api/agents/{agent['agent_id']}/status",
                json={"status": "stalled"},
                timeout=10,
            )
            print(f"  Marked {agent['agent_id']} as stalled (last heartbeat: {last_hb})")

    return stalled


def release_expired_claims(api_url: str) -> list[str]:
    """Release experiment claims that have expired."""
    resp = requests.get(f"{api_url}/api/experiments?status=claimed", timeout=30)
    resp.raise_for_status()
    claimed = resp.json().get("experiments", [])

    released = []
    now = datetime.now(timezone.utc)

    for exp in claimed:
        claimed_at = exp.get("claimed_at")
        if not claimed_at:
            continue

        claimed_dt = datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))
        if (now - claimed_dt).total_seconds() > CLAIM_EXPIRY_MINUTES * 60:
            released.append(exp["experiment_id"])
            # Release by resetting to queued
            requests.post(
                f"{api_url}/api/experiments/{exp['experiment_id']}/claim",
                json={"agent_id": exp["claimed_by"], "release": True},
                timeout=10,
            )
            print(f"  Released {exp['experiment_id']} (claimed by {exp['claimed_by']})")

    return released


def check_runners(api_url: str) -> list[dict]:
    """Check runner availability."""
    resp = requests.get(f"{api_url}/api/runners", timeout=30)
    if resp.status_code != 200:
        return []
    runners = resp.json().get("runners", [])

    available = [r for r in runners if r["status"] == "available"]
    print(f"  Runners: {len(available)}/{len(runners)} available")
    return available


def main():
    parser = argparse.ArgumentParser(description="ModelSwarm health check")
    parser.add_argument("--api-url", default=os.environ.get("MODELSWARM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--ops-agent-id", default=os.environ.get("OPS_AGENT_ID"))
    parser.add_argument("--ops-api-key", default=os.environ.get("MODELSWARM_OPS_KEY"))
    args = parser.parse_args()

    print("=" * 50)
    print("ModelSwarm Health Check")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 50)

    # Check agents
    print("\n[1/3] Checking agent heartbeats...")
    stalled = check_agents(args.api_url)
    print(f"  Found {len(stalled)} stalled agents")

    # Release expired claims
    print("\n[2/2] Releasing expired experiment claims...")
    released = release_expired_claims(args.api_url)
    print(f"  Released {len(released)} experiments")

    # Check runners
    print("\n[3/3] Checking runners...")
    runners = check_runners(args.api_url)

    print("\n" + "=" * 50)
    print("Health check complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
