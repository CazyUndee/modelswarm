"""
Shared test fixtures for ModelSwarm tests.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_agent_identity(temp_dir):
    """Create a mock agent identity file."""
    identity = {
        "agent_id": "agent-test",
        "name": "TestAgent",
        "model": "test-model",
        "role": "research",
        "parent_agent_id": None,
        "registered_at": "2026-08-10T00:00:00Z",
        "status": "active",
    }
    agent_dir = temp_dir / "agents" / "agent-test"
    agent_dir.mkdir(parents=True)
    identity_path = agent_dir / "identity.yaml"
    with open(identity_path, "w") as f:
        yaml.dump(identity, f)
    return {"identity": identity, "path": identity_path, "agent_dir": agent_dir}


@pytest.fixture
def mock_subagent_identity(temp_dir):
    """Create a mock subagent identity file."""
    identity = {
        "agent_id": "agent-test-01",
        "name": "TestSubagent",
        "model": "test-model",
        "role": "research",
        "parent_agent_id": "agent-test",
        "registered_at": "2026-08-10T01:00:00Z",
        "status": "active",
    }
    subagent_dir = temp_dir / "agents" / "agent-test" / "subagents" / "agent-test-01"
    subagent_dir.mkdir(parents=True)
    identity_path = subagent_dir / "identity.yaml"
    with open(identity_path, "w") as f:
        yaml.dump(identity, f)
    return {"identity": identity, "path": identity_path, "subagent_dir": subagent_dir}


@pytest.fixture
def mock_credentials(temp_dir):
    """Create mock credentials file."""
    creds_dir = temp_dir / ".modelswarm"
    creds_dir.mkdir(parents=True)
    creds = {
        "api_url": "https://test.workers.dev",
        "api_key": "ms_test_key_12345",
        "agent_id": "agent-test",
    }
    creds_path = creds_dir / "credentials.json"
    import json
    with open(creds_path, "w") as f:
        json.dump(creds, f)
    return {"credentials": creds, "path": creds_path, "creds_dir": creds_dir}


@pytest.fixture
def mock_experiment():
    """Provide a mock experiment dict."""
    return {
        "experiment_id": "EXP-001",
        "hypothesis": "Test hypothesis",
        "agent_id": "agent-test",
        "executing_agent_id": "agent-test",
        "parent_agent_id": None,
        "parent_experiment_id": None,
        "competition_id": "s6e8",
        "phase": 1,
        "features": ["feature_a", "feature_b"],
        "model": "lightgbm",
        "validation_protocol": "stratified_5_fold",
        "status": "queued",
        "created_at": "2026-08-10T00:00:00Z",
    }
