#!/usr/bin/env python3
"""
Experiment runner script.

This is a template for running experiments via GitHub Actions.
Real agents would customize this for their specific experiment.

Usage:
    python scripts/run_experiment.py --experiment-id EXP-014 --agent-id agent-7f3c --competition-id s6e8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_experiment(experiment_id: str) -> dict:
    """Load experiment configuration from the experiments directory."""
    # Search in all experiment subdirectories
    for subdir in ["queue", "active", "completed", "rejected"]:
        path = Path(f"experiments/{subdir}/{experiment_id}.yaml")
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    return None


def run_experiment(experiment: dict, agent_id: str, output_dir: str) -> dict:
    """
    Run the actual experiment.

    This is a placeholder. Real implementation would:
    1. Load the dataset
    2. Apply feature engineering
    3. Train the model
    4. Generate OOF predictions
    5. Compute metrics
    6. Save artifacts

    Returns dict with results.
    """
    print(f"Running experiment: {experiment['experiment_id']}")
    print(f"Hypothesis: {experiment['hypothesis']}")
    print(f"Model: {experiment.get('model', 'unknown')}")
    print(f"Features: {experiment.get('features', [])}")

    # Placeholder: real implementation goes here
    results = {
        "experiment_id": experiment["experiment_id"],
        "oof_metric": None,
        "fold_metrics": [],
        "runtime_seconds": 0,
        "artifacts": [],
    }

    return results


def save_results(results: dict, output_dir: str) -> None:
    """Save experiment results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{results['experiment_id']}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run a ModelSwarm experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--competition-id", default="s6e8")
    parser.add_argument("--output-dir", default="agents/agent-example/workspace/artifacts")
    args = parser.parse_args()

    # Load experiment
    experiment = load_experiment(args.experiment_id)
    if not experiment:
        print(f"Experiment {args.experiment_id} not found", file=sys.stderr)
        sys.exit(1)

    # Run
    results = run_experiment(experiment, args.agent_id, args.output_dir)

    # Save
    save_results(results, args.output_dir)

    # Output for GitHub Actions
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"oof_metric={results.get('oof_metric', '')}\n")


if __name__ == "__main__":
    main()
