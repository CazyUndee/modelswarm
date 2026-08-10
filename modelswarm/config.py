"""
Configuration management for ModelSwarm.

Config is stored in ~/.modelswarm/config.json.
API URL can be overridden with MODELSWARM_API_URL env var.
"""

import json
import os
from pathlib import Path

DEFAULT_API_URL = "https://modelswarm.workers.dev"
CONFIG_DIR = Path.home() / ".modelswarm"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_config_dir() -> Path:
    """Get the config directory, creating it if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def get_api_url() -> str:
    """Get the API URL from env var, config file, or default."""
    # 1. Environment variable takes precedence
    env_url = os.environ.get("MODELSWARM_API_URL")
    if env_url:
        return env_url

    # 2. Config file
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        return config.get("api_url", DEFAULT_API_URL)

    # 3. Default
    return DEFAULT_API_URL


def set_api_url(url: str) -> None:
    """Save the API URL to the config file."""
    get_config_dir()
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    config["api_url"] = url
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_config() -> dict:
    """Get the full configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"api_url": DEFAULT_API_URL}


def set_config(config: dict) -> None:
    """Save the full configuration."""
    get_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
