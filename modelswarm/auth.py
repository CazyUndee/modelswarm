"""
Authentication management for ModelSwarm.

Credentials are stored in ~/.modelswarm/credentials.json.
Never store credentials in the repository directory.
"""

import json
import os
from pathlib import Path

from modelswarm.exceptions import AuthError


def get_modelswarm_dir() -> Path:
    """Get the .modelswarm directory path.

    Uses MODELSWARM_HOME env var if set (for testing),
    otherwise defaults to ~/.modelswarm.
    """
    env_home = os.environ.get("MODELSWARM_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".modelswarm"


def _credentials_path() -> Path:
    """Get the credentials file path (computed at runtime)."""
    return get_modelswarm_dir() / "credentials.json"


def save_credentials(api_url: str, api_key: str, agent_id: str) -> None:
    """Save credentials to the credentials file."""
    dir_path = get_modelswarm_dir()
    dir_path.mkdir(parents=True, exist_ok=True)

    creds = {
        "api_url": api_url,
        "api_key": api_key,
        "agent_id": agent_id,
    }
    with open(_credentials_path(), "w") as f:
        json.dump(creds, f, indent=2)


def load_credentials() -> dict:
    """Load credentials from the credentials file.

    Raises:
        AuthError: If credentials file does not exist.
    """
    cred_path = _credentials_path()
    if not cred_path.exists():
        raise AuthError(
            "No credentials found. Run 'modelswarm login' or 'modelswarm register' first."
        )

    with open(cred_path) as f:
        creds = json.load(f)

    if "api_key" not in creds:
        raise AuthError("Invalid credentials file: missing 'api_key'.")

    return creds


def clear_credentials() -> None:
    """Remove stored credentials."""
    cred_path = _credentials_path()
    if cred_path.exists():
        cred_path.unlink()


def get_api_key() -> str:
    """Get just the API key.

    Raises:
        AuthError: If no credentials exist.
    """
    return load_credentials()["api_key"]


def get_agent_id() -> str:
    """Get the agent ID from credentials.

    Raises:
        AuthError: If no credentials exist.
    """
    return load_credentials()["agent_id"]
