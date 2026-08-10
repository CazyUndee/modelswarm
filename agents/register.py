#!/usr/bin/env python3
"""
Agent registration script.

Usage:
    python agents/register.py --name "Happy" --model "claude-opus-5" --role research
    python agents/register.py --name "Worker" --model "claude-sonnet-5" --role research --parent agent-7f3c
"""

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

# Default API URL — override with MODELSWARM_API_URL env var
DEFAULT_API_URL = "https://modelswarm.workers.dev"


def generate_agent_id(parent_id: str | None = None) -> str:
    """Generate a unique agent ID."""
    suffix = secrets.token_hex(2)  # 4 hex chars
    if parent_id:
        # Count existing subagents
        parent_dir = Path(f"agents/{parent_id}/subagents")
        existing = len(list(parent_dir.iterdir())) if parent_dir.exists() else 0
        return f"{parent_id}-{existing + 1:02d}"
    return f"agent-{suffix}"


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"ms_{secrets.token_hex(32)}"


def create_workspace(agent_id: str) -> Path:
    """Create agent workspace directory structure."""
    base = Path(f"agents/{agent_id}/workspace")
    dirs = ["scripts", "experiments", "notes", "artifacts", "scratch"]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
    return base


def create_identity(
    agent_id: str,
    name: str,
    model: str,
    role: str,
    parent_id: str | None = None,
) -> dict:
    """Create agent identity dictionary."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "agent_id": agent_id,
        "name": name,
        "model": model,
        "role": role,
        "parent_agent_id": parent_id,
        "registered_at": now,
        "status": "active",
    }


def register_remotely(api_url: str, name: str, model: str, role: str, parent_id: str | None = None) -> dict:
    """Register agent with the Cloudflare API."""
    payload = {"name": name, "model": model, "role": role}
    if parent_id:
        payload["parent_agent_id"] = parent_id

    resp = requests.post(f"{api_url}/api/agents/register", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_identity(identity: dict, path: Path) -> None:
    """Save identity to YAML file."""
    with open(path, "w") as f:
        yaml.dump(identity, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Register a ModelSwarm agent")
    parser.add_argument("--name", required=True, help="Agent name")
    parser.add_argument("--model", required=True, help="AI model (e.g., claude-opus-5)")
    parser.add_argument("--role", default="research", choices=["research", "operations", "review", "infrastructure"])
    parser.add_argument("--parent", help="Parent agent ID (for subagents)")
    parser.add_argument("--api-url", default=os.environ.get("MODELSWARM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--local-only", action="store_true", help="Register locally without API call")
    args = parser.parse_args()

    # Generate ID
    agent_id = generate_agent_id(args.parent)
    api_key = generate_api_key()

    if args.local_only:
        # Local-only registration (no API call)
        identity = create_identity(agent_id, args.name, args.model, args.role, args.parent)
        workspace = create_workspace(agent_id)
        identity_path = Path(f"agents/{agent_id}/identity.yaml")
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        save_identity(identity, identity_path)
        print(f"Agent registered locally: {agent_id}")
        print(f"Workspace: {workspace}")
        print(f"Identity: {identity_path}")
        return

    # Remote registration
    try:
        result = register_remotely(args.api_url, args.name, args.model, args.role, args.parent)
        agent_id = result["agent_id"]
        api_key = result["api_key"]
        print(f"Registered with API: {agent_id}")
    except requests.RequestException as e:
        print(f"API registration failed: {e}", file=sys.stderr)
        print("Falling back to local-only registration", file=sys.stderr)
        identity = create_identity(agent_id, args.name, args.model, args.role, args.parent)
        workspace = create_workspace(agent_id)
        identity_path = Path(f"agents/{agent_id}/identity.yaml")
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        save_identity(identity, identity_path)
        print(f"Agent registered locally: {agent_id}")
        return

    # Create workspace and identity
    workspace = create_workspace(agent_id)
    identity = create_identity(agent_id, args.name, args.model, args.role, args.parent)
    identity_path = Path(f"agents/{agent_id}/identity.yaml")
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    save_identity(identity, identity_path)

    print(f"\n{'='*50}")
    print(f"Registration successful!")
    print(f"{'='*50}")
    print(f"Agent ID: {agent_id}")
    print(f"API Key:  {api_key}")
    print(f"Workspace: {workspace}")
    print(f"Identity: {identity_path}")
    print(f"\nSave your API key — it won't be shown again.")
    print(f"Run: modelswarm login")


if __name__ == "__main__":
    main()
