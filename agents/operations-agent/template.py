#!/usr/bin/env python3
"""
Background Operations Agent Template.

The operations agent is a persistent background agent responsible for
keeping the research program functioning. It does NOT replace research agents.

Responsibilities:
- Experiment registry maintenance
- Research-state maintenance
- Compute allocation
- Runner monitoring
- Task assignment
- Experiment prioritization
- Duplicate detection
- Stalled-agent detection
- Dead-branch cleanup
- Artifact organization
- External research discovery
- Research synthesis
- Follow-up task generation

Authority:
- Task allocation
- Experiment lifecycle
- Compute allocation
- Runner management
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from modelswarm import Client
from modelswarm.identity import load_identity


# Configuration
HEARTBEAT_INTERVAL_SECONDS = 300  # 5 minutes
STALE_THRESHOLD_MINUTES = 15
CLAIM_EXPIRY_MINUTES = 30
DUPLICATE_SIMILARITY_THRESHOLD = 0.8


class OperationsAgent:
    """Background operations agent for swarm health and coordination."""

    def __init__(self, api_url: str | None = None):
        self.client = Client(api_url=api_url)
        self.identity = None
        try:
            self.identity = load_identity()
        except Exception:
            pass

    def run(self):
        """Main operations loop."""
        print(f"Operations agent starting at {datetime.now(timezone.utc).isoformat()}")

        while True:
            try:
                self.check_agent_heartbeats()
                self.release_expired_claims()
                self.detect_duplicate_experiments()
                self.prioritize_experiment_queue()
                self.check_runner_health()
                self.update_research_state()
                self.detect_stalled_branches()
            except KeyboardInterrupt:
                print("Operations agent stopping.")
                break
            except Exception as e:
                print(f"Error in operations loop: {e}")

            time.sleep(HEARTBEAT_INTERVAL_SECONDS)

    def check_agent_heartbeats(self):
        """Detect and mark stalled agents."""
        agents = self.client.get_agents()
        now = datetime.now(timezone.utc)

        for agent in agents:
            if agent.get("status") != "active":
                continue

            last_hb = agent.get("last_heartbeat")
            if not last_hb:
                continue

            last_hb_dt = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
            if (now - last_hb_dt).total_seconds() > STALE_THRESHOLD_MINUTES * 60:
                agent_id = agent["agent_id"]
                print(f"  Marking {agent_id} as stalled")
                try:
                    self.client._request("POST", f"/api/agents/{agent_id}/status",
                                        json={"status": "stalled"})
                except Exception:
                    pass

    def release_expired_claims(self):
        """Release experiment claims that have expired."""
        claimed = self.client.get_experiments(status="claimed")
        now = datetime.now(timezone.utc)

        for exp in claimed:
            claimed_at = exp.get("claimed_at")
            if not claimed_at:
                continue

            claimed_dt = datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))
            if (now - claimed_dt).total_seconds() > CLAIM_EXPIRY_MINUTES * 60:
                exp_id = exp["experiment_id"]
                print(f"  Releasing expired claim: {exp_id}")
                try:
                    self.client._request("POST", f"/api/experiments/{exp_id}/claim",
                                        json={"release": True})
                except Exception:
                    pass

    def detect_duplicate_experiments(self):
        """Detect and flag duplicate or highly similar experiments."""
        queued = self.client.get_experiments(status="queued")

        for i, exp_a in enumerate(queued):
            for exp_b in queued[i + 1:]:
                similarity = self._compute_similarity(exp_a, exp_b)
                if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                    print(f"  Duplicate detected: {exp_a['experiment_id']} ~ {exp_b['experiment_id']}")

    def _compute_similarity(self, exp_a: dict, exp_b: dict) -> float:
        """Compute similarity between two experiments (0-1)."""
        score = 0.0
        weights = 0.0

        # Same model
        if exp_a.get("model") == exp_b.get("model"):
            score += 0.3
        weights += 0.3

        # Feature overlap
        features_a = set(exp_a.get("features", []))
        features_b = set(exp_b.get("features", []))
        if features_a or features_b:
            overlap = len(features_a & features_b) / max(len(features_a | features_b), 1)
            score += overlap * 0.5
        weights += 0.5

        # Same phase
        if exp_a.get("phase") == exp_b.get("phase"):
            score += 0.2
        weights += 0.2

        return score / weights if weights > 0 else 0.0

    def prioritize_experiment_queue(self):
        """Prioritize the experiment queue using information-gain principles."""
        queued = self.client.get_experiments(status="queued")
        if not queued:
            return

        # Simple prioritization: prefer experiments with parent experiments
        # (building on known results) over completely new hypotheses
        for exp in queued:
            if exp.get("parent_experiment_id"):
                pass  # Higher priority

    def check_runner_health(self):
        """Check compute runner availability."""
        try:
            runners = self.client._request("GET", "/api/runners")
            available = [r for r in runners.get("runners", []) if r["status"] == "available"]
            total = len(runners.get("runners", []))
            print(f"  Runners: {len(available)}/{total} available")
        except Exception:
            pass

    def update_research_state(self):
        """Update the global research state if needed."""
        state = self.client.get_state()
        # Check if champion needs updating
        # Check if phase transition is warranted
        # Update priorities based on recent results
        pass

    def detect_stalled_branches(self):
        """Detect research branches that have stopped making progress."""
        completed = self.client.get_experiments(status="completed")
        promoted = [e for e in completed if e.get("decision") == "promoted"]

        # Check if any promoted experiments lack follow-up
        for exp in promoted:
            follow_ups = self.client.get_experiments(
                parent_experiment_id=exp["experiment_id"]
            )
            if not follow_ups:
                print(f"  Stalled branch: {exp['experiment_id']} has no follow-up")

    def make_decision(self, experiment: dict) -> str:
        """Make a decision on an experiment result.

        Returns one of: 'promote', 'reject', 'replicate', 'continue'
        """
        oof = experiment.get("oof_metric")
        if oof is None:
            return "reject"

        # Compare with current champion
        state = self.client.get_state()
        champion_score = state.get("best_score", 0.0)

        if oof > champion_score:
            return "promote"
        elif oof > champion_score - 0.001:
            return "replicate"  # Close enough to warrant replication
        else:
            return "reject"


def main():
    """Run the operations agent."""
    api_url = os.environ.get("MODELSWARM_API_URL")
    agent = OperationsAgent(api_url=api_url)
    agent.run()


if __name__ == "__main__":
    main()
