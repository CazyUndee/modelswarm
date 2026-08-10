"""
Agent identity management.

Identity is stored in agents/<agent-id>/identity.yaml.
The client auto-discovers identity by walking up directories.
"""

import os
from pathlib import Path

import yaml

from modelswarm.exceptions import IdentityNotFoundError


def load_identity(path: str | None = None) -> dict:
    """Load agent identity from a YAML file.

    Args:
        path: Explicit path to identity.yaml. If None, auto-discovers.

    Returns:
        Dict with identity fields.

    Raises:
        IdentityNotFoundError: If identity file cannot be found.
    """
    if path:
        identity_path = Path(path)
        if not identity_path.exists():
            raise IdentityNotFoundError(f"Identity file not found: {path}")
    else:
        identity_path = discover_identity()

    with open(identity_path) as f:
        identity = yaml.safe_load(f)

    if not identity:
        raise IdentityNotFoundError(f"Empty identity file: {identity_path}")

    return identity


def save_identity(identity: dict, path: str) -> None:
    """Save agent identity to a YAML file.

    Args:
        identity: Dict with identity fields.
        path: Path to save the identity.yaml file.
    """
    identity_path = Path(path)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identity_path, "w") as f:
        yaml.dump(identity, f, default_flow_style=False, sort_keys=False)


def discover_identity() -> Path:
    """Auto-discover identity by walking up directories looking for agents/*/identity.yaml.

    Returns:
        Path to the discovered identity.yaml.

    Raises:
        IdentityNotFoundError: If no identity file is found.
    """
    current = Path.cwd()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Look for agents/*/identity.yaml
        agents_dir = parent / "agents"
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    identity_file = agent_dir / "identity.yaml"
                    if identity_file.exists():
                        return identity_file

    raise IdentityNotFoundError(
        "Could not auto-discover identity.yaml. "
        "Run from within an agent directory or specify --identity path."
    )


def is_subagent(identity: dict) -> bool:
    """Check if an identity belongs to a subagent."""
    return identity.get("parent_agent_id") is not None
