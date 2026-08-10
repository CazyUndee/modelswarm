"""
Tests for agent identity management.
"""

import pytest
import yaml

from modelswarm.identity import load_identity, save_identity, discover_identity


class TestIdentityLoading:
    """Test identity file loading and discovery."""

    def test_load_identity_from_path(self, mock_agent_identity):
        """Test loading identity from an explicit path."""
        identity = load_identity(str(mock_agent_identity["path"]))
        assert identity["agent_id"] == "agent-test"
        assert identity["name"] == "TestAgent"
        assert identity["role"] == "research"

    def test_load_identity_missing_file(self, temp_dir):
        """Test loading from a non-existent path raises error."""
        from modelswarm.exceptions import IdentityNotFoundError
        with pytest.raises(IdentityNotFoundError):
            load_identity(str(temp_dir / "nonexistent.yaml"))

    def test_save_identity(self, temp_dir):
        """Test saving identity to YAML."""
        identity = {
            "agent_id": "agent-new",
            "name": "NewAgent",
            "model": "claude-opus-5",
            "role": "research",
            "parent_agent_id": None,
            "registered_at": "2026-08-10T00:00:00Z",
            "status": "active",
        }
        path = temp_dir / "identity.yaml"
        save_identity(identity, str(path))
        assert path.exists()

        # Verify content
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["agent_id"] == "agent-new"
        assert loaded["name"] == "NewAgent"

    def test_discover_identity(self, mock_agent_identity):
        """Test auto-discovery of identity by walking up directories."""
        import os
        original = os.getcwd()
        try:
            os.chdir(mock_agent_identity["agent_dir"])
            path = discover_identity()
            # discover_identity returns a Path; load it to get the dict
            assert path.exists()
            identity = load_identity(str(path))
            assert identity["agent_id"] == "agent-test"
        finally:
            os.chdir(original)

    def test_subagent_identity(self, mock_subagent_identity):
        """Test loading subagent identity with parent_agent_id."""
        identity = load_identity(str(mock_subagent_identity["path"]))
        assert identity["agent_id"] == "agent-test-01"
        assert identity["parent_agent_id"] == "agent-test"

    def test_identity_required_fields(self, mock_agent_identity):
        """Test that required fields are present."""
        identity = load_identity(str(mock_agent_identity["path"]))
        required = ["agent_id", "name", "model", "role", "registered_at", "status"]
        for field in required:
            assert field in identity, f"Missing required field: {field}"
