#!/usr/bin/env python3
"""
Record experiment results back to the experiment config file.

This script reads the results from the experiment output directory
and writes them back to the experiment config YAML.

Usage:
    python scripts/record_results.py --experiment-id EXP-014 --config competitions/s6e8/experiments/EXP-014.yaml --output-dir experiments/output/EXP-014/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def record_results(config_path: str, output_dir: str) -> dict:
    """Read results from output dir and write back to config."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load results
    results_path = Path(output_dir) / "results.json"
    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    # Keys carried from results.json into the experiment record.
    PASSTHROUGH = (
        "oof_metric", "fold_metrics", "runtime_seconds", "features_used",
        "categorical_features", "model_name", "blend_method",
        "member_correlations", "rank_average_diagnostic",
    )

    # Update config with results
    config["results"] = {
        "status": "completed",
        **{k: results.get(k) for k in PASSTHROUGH},
        "members": [
            {"name": m.get("name"), "oof_auc": m.get("oof_auc"),
             "fold_metrics": m.get("fold_metrics", [])}
            for m in results.get("members", [])
        ],
        "artifacts": [
            str(Path(output_dir) / "oof_predictions.csv"),
            str(Path(output_dir) / "results.json"),
        ],
        "decision": None,  # To be decided by agent/operations
        "reasoning": None,  # To be filled by agent/operations
    }

    # Write updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Results recorded to {config_path}")
    return config["results"]


def main():
    # Windows consoles may default to cp1252 and crash on emoji markers.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Record experiment results")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    record_results(args.config, args.output_dir)


if __name__ == "__main__":
    main()
