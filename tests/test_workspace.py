"""
Tests for agent workspace management.
"""

import pytest

from modelswarm.workspace import init_workspace, get_workspace_path, create_subagent_workspace


class TestWorkspace:
    """Test workspace creation and management."""

    def test_init_workspace(self, temp_dir, monkeypatch):
        """Test workspace directory creation."""
        monkeypatch.chdir(temp_dir)

        workspace = init_workspace("agent-test")
        assert workspace.exists()
        assert (workspace / "scripts").exists()
        assert (workspace / "experiments").exists()
        assert (workspace / "notes").exists()
        assert (workspace / "artifacts").exists()
        assert (workspace / "scratch").exists()

    def test_get_workspace_path(self, temp_dir, monkeypatch):
        """Test getting workspace path."""
        monkeypatch.chdir(temp_dir)

        init_workspace("agent-test")
        path = get_workspace_path("agent-test")
        assert path.exists()
        assert path.name == "workspace"

    def test_create_subagent_workspace(self, temp_dir, monkeypatch):
        """Test subagent workspace creation."""
        monkeypatch.chdir(temp_dir)

        init_workspace("agent-test")
        sub_ws = create_subagent_workspace("agent-test", "agent-test-01")
        assert sub_ws.exists()
        assert "subagents" in str(sub_ws)
        assert "agent-test-01" in str(sub_ws)

    def test_workspace_isolation(self, temp_dir, monkeypatch):
        """Test that different agents have isolated workspaces."""
        monkeypatch.chdir(temp_dir)

        ws1 = init_workspace("agent-1")
        ws2 = init_workspace("agent-2")

        # Write to agent-1's workspace
        (ws1 / "notes" / "test.md").write_text("Agent 1 notes")

        # Agent-2 should not see it
        assert not (ws2 / "notes" / "test.md").exists()
