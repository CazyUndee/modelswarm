"""
Agent workspace management.

Each agent has a private workspace for scripts, experiments, notes, artifacts, and scratch.
"""

from pathlib import Path


WORKSPACE_DIRS = ["scripts", "experiments", "notes", "artifacts", "scratch"]


def init_workspace(agent_id: str, base_dir: str | None = None) -> Path:
    """Create agent workspace directory structure.

    Args:
        agent_id: The agent's unique ID.
        base_dir: Base directory (defaults to current directory).

    Returns:
        Path to the created workspace.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    workspace = base / "agents" / agent_id / "workspace"

    for d in WORKSPACE_DIRS:
        (workspace / d).mkdir(parents=True, exist_ok=True)

    return workspace


def get_workspace_path(agent_id: str, base_dir: str | None = None) -> Path:
    """Get the path to an agent's workspace.

    Args:
        agent_id: The agent's unique ID.
        base_dir: Base directory (defaults to current directory).

    Returns:
        Path to the workspace.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    return base / "agents" / agent_id / "workspace"


def create_subagent_workspace(parent_agent_id: str, subagent_id: str, base_dir: str | None = None) -> Path:
    """Create a workspace for a subagent.

    Args:
        parent_agent_id: The parent agent's ID.
        subagent_id: The subagent's unique ID.
        base_dir: Base directory (defaults to current directory).

    Returns:
        Path to the created subagent workspace.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    subagent_dir = base / "agents" / parent_agent_id / "subagents" / subagent_id

    # Create identity.yaml placeholder
    subagent_dir.mkdir(parents=True, exist_ok=True)

    # Create workspace
    workspace = subagent_dir / "workspace"
    for d in WORKSPACE_DIRS:
        (workspace / d).mkdir(parents=True, exist_ok=True)

    return workspace
