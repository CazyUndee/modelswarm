"""
Tests for authentication and credentials management.
"""

import json

import pytest

from modelswarm.auth import (
    save_credentials,
    load_credentials,
    clear_credentials,
    AuthError,
)


class TestCredentials:
    """Test credential storage and retrieval."""

    def test_save_and_load_credentials(self, temp_dir, monkeypatch):
        """Test saving and loading credentials."""
        monkeypatch.setenv("MODELSWARM_HOME", str(temp_dir))

        creds = {
            "api_url": "https://test.workers.dev",
            "api_key": "ms_test_key_12345",
            "agent_id": "agent-test",
        }
        save_credentials(**creds)

        loaded = load_credentials()
        assert loaded["api_key"] == "ms_test_key_12345"
        assert loaded["agent_id"] == "agent-test"

    def test_load_missing_credentials(self, temp_dir, monkeypatch):
        """Test loading when no credentials exist raises error."""
        monkeypatch.setenv("MODELSWARM_HOME", str(temp_dir / "nonexistent"))

        with pytest.raises(AuthError):
            load_credentials()

    def test_clear_credentials(self, temp_dir, monkeypatch):
        """Test clearing stored credentials."""
        monkeypatch.setenv("MODELSWARM_HOME", str(temp_dir))

        save_credentials(
            api_url="https://test.workers.dev",
            api_key="ms_test_key",
            agent_id="agent-test",
        )
        clear_credentials()

        with pytest.raises(AuthError):
            load_credentials()

    def test_credentials_not_in_repo(self, temp_dir, monkeypatch):
        """Test that credentials are stored in home directory, not repo."""
        monkeypatch.setenv("MODELSWARM_HOME", str(temp_dir))

        save_credentials(
            api_url="https://test.workers.dev",
            api_key="ms_test_key",
            agent_id="agent-test",
        )

        # Credentials should be in MODELSWARM_HOME/.modelswarm/
        creds_path = temp_dir / "credentials.json"
        assert creds_path.exists()

        with open(creds_path) as f:
            data = json.load(f)
        assert data["api_key"] == "ms_test_key"


import os  # noqa: E402 - needed for os.name check
